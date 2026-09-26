"""
PTM-CIL: when does an expert bank earn its keep over an analytic readout?

    docs/PTM_CIL_PREREG.md

Per (benchmark, backbone, seed), on frozen ViT-B/16 features with the literature's
protocol (`pal_moe.data.ptm_benchmarks`):

    ncm     SimpleCIL: cosine to class means, training-free
    ridge   closed-form ridge on the raw features (ACIL / RanPAC without RP)
    rp      RanPAC without PETL: ridge on relu(z @ W), W ~ N(0, 1) [768, M]

The ridge penalties are selected once, on task 0 only (RanPAC's rule), and pinned.
Each readout is reported as average incremental accuracy and final accuracy.

Every bank - this repository's L3 bank (the S11 recipe) and any external bank given
as an `ExpertDump` (EASE, MOS, MoTE, ... exported under every forced expert) - is
routed by each readout (the owner of its argmax class) and decomposed:
P2 = m * rho + P(not r, not tau) * rho' - P(r) * beta (`pal_moe.eval.decomposition`).

`--api` also runs the rp readout through `PalMoE` (one guarded write per task) and
vetoes any prediction difference from the direct computation; it records the guard
reports and the wall time per write.

    python experiments/ptm_cil.py --synthetic                       # CI smoke
    python experiments/ptm_cil.py --benchmarks cifar100,imagenet_r \\
        --backbones in21k_1k --seeds 0,1,2,3,4 --device cuda --api
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.api import Batch, GuardConfig, PalMoE  # noqa: E402
from pal_moe.arch.readouts import mask_unseen  # noqa: E402
from pal_moe.core.random_features import RandomProjection  # noqa: E402
from pal_moe.data.ptm_benchmarks import (  # noqa: E402
    BENCHMARKS,
    class_order,
    load_raw_cache,
    split_tasks,
)
from pal_moe.edit.stats import LinearStats, one_hot, select_ridge  # noqa: E402
from pal_moe.eval.decomposition import ExpertDump, decompose  # noqa: E402
from pal_moe.eval.stats import paired_stats  # noqa: E402
from pal_moe.experts.ladder import train_model  # noqa: E402

PREREG = "docs/PTM_CIL_PREREG.md"
READOUTS = ("ncm", "ridge", "rp")
S11_RECIPE = {"epochs": 10, "lr": 1e-3, "batch_size": 128, "lambda_func": 1.0}
IDENTITY_BAND = 1e-12
API_BAND = 1e-8


def git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def synthetic_cache(
    seed: int = 0, C: int = 20, D: int = 32, n: int = 60, scale: float = 0.7
) -> dict:
    """Overlapping Gaussian class clusters: a pipeline check, never a result."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(C, D, generator=g) * scale

    def draw(k):
        y = torch.arange(C).repeat_interleave(k)
        return mu[y] + torch.randn(y.numel(), D, generator=g), y

    return {
        "meta": {"benchmark": "synthetic", "feature_dim": D},
        "train": draw(n),
        "test": draw(n // 2),
    }


def seen_mask(logits: torch.Tensor, n_seen: int) -> torch.Tensor:
    return mask_unseen(logits, list(range(n_seen)))


# -- readouts ---------------------------------------------------------------------------


class NCM:
    def __init__(self, C: int, D: int, device):
        self.sums = torch.zeros(C, D, dtype=torch.float64, device=device)
        self.counts = torch.zeros(C, dtype=torch.float64, device=device)

    def add(self, z, y):
        self.sums.index_add_(0, y, z.double())
        self.counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float64))

    def logits(self, z):
        protos = self.sums / self.counts.clamp_min(1).unsqueeze(1)
        protos = torch.nn.functional.normalize(protos, dim=1)
        return torch.nn.functional.normalize(z.double(), dim=1) @ protos.t()


@torch.no_grad()
def analytic_arms(tasks, C, lam, fm, device) -> dict:
    D = tasks[0]["splits"]["train"][0].size(1)
    ncm = NCM(C, D, device)
    raw = LinearStats(D, C, ridge=lam["ridge"], device=device)
    rp = LinearStats(D, C, ridge=lam["rp"], device=device, feature_map=fm)
    steps = {k: [] for k in READOUTS}
    n_seen = 0
    for t, task in enumerate(tasks):
        z, y = [v.to(device) for v in task["splits"]["train"]]
        ncm.add(z, y)
        raw.add(raw.contribution(f"t{t}", z, one_hot(y, C)))
        rp.add(rp.contribution(f"t{t}", z, one_hot(y, C)))
        n_seen += len(task["classes"])
        zt = torch.cat([k["splits"]["test"][0] for k in tasks[: t + 1]]).to(device)
        yt = torch.cat([k["splits"]["test"][1] for k in tasks[: t + 1]]).to(device)
        for name, f in (
            ("ncm", ncm.logits),
            ("ridge", raw.predict),
            ("rp", rp.predict),
        ):
            pred = seen_mask(f(zt), n_seen).argmax(-1)
            steps[name].append(float((pred == yt).float().mean()))
    z_all = torch.cat([k["splits"]["test"][0] for k in tasks]).to(device)
    final_logits = {
        "ncm": seen_mask(ncm.logits(z_all), C),
        "ridge": seen_mask(raw.predict(z_all), C),
        "rp": seen_mask(rp.predict(z_all), C),
    }
    summary = {
        k: {
            "final": steps[k][-1],
            "avg_incremental": sum(steps[k]) / len(steps[k]),
            "steps": steps[k],
        }
        for k in READOUTS
    }
    return {"summary": summary, "final_logits": final_logits, "rp_stats": rp}


# -- banks ------------------------------------------------------------------------------


@torch.no_grad()
def own_bank_dump(tasks, C, seed, recipe, device) -> ExpertDump:
    """This repository's L3 bank (S11 recipe), exported under every forced expert."""
    args = argparse.Namespace(**recipe, seed=seed)
    cell = {"seed": seed, "rank": 8, "protos": 1, "top_k": 1}
    with torch.enable_grad():
        model = train_model("L3_per_task", tasks, cell, args, device)
    z = torch.cat([k["splits"]["test"][0] for k in tasks]).to(device)
    y = torch.cat([k["splits"]["test"][1] for k in tasks])
    T = len(tasks)
    cols = []
    for t in range(T):
        ids = torch.full((z.size(0),), t, dtype=torch.long, device=device)
        logits = mask_unseen(
            model.readout.predict(model.apply_experts(z, ids)), model.seen
        )
        cols.append(logits.argmax(-1).cpu())
    toc = torch.empty(C, dtype=torch.long)
    for k in tasks:
        toc[torch.tensor(k["classes"])] = k["task_id"]
    native = model.logits(z).argmax(-1).cpu()
    return ExpertDump(
        y=y,
        task_of_class=toc,
        expert_pred=torch.stack(cols, 1),
        native_pred=native,
        meta={"method": "pal_l3", "recipe": recipe, "rank": 8, "seed": seed},
    )


def external_dumps(ext_dir, benchmark, backbone, seed) -> dict:
    """`{method}__{benchmark}__{backbone}__seed{seed}.npz` files in `ext_dir`."""
    if not ext_dir:
        return {}
    out = {}
    for p in sorted(Path(ext_dir).glob(f"*__{benchmark}__{backbone}__seed{seed}.npz")):
        out[p.name.split("__")[0]] = ExpertDump.load(p)
    return out


# -- the API arm ------------------------------------------------------------------------


@torch.no_grad()
def api_arm(tasks, C, lam_rp, M, rf_seed, device, direct_logits) -> dict:
    D = tasks[0]["splits"]["train"][0].size(1)
    g = torch.Generator().manual_seed(rf_seed)
    pool = torch.cat([k["splits"]["train"][0] for k in tasks])
    canary = pool[torch.randperm(pool.size(0), generator=g)[:200]].to(device)
    model = PalMoE(
        dim=D,
        num_classes=C,
        ridge=lam_rp,
        random_features=M,
        rf_seed=rf_seed,
        canary=canary,
        guards=GuardConfig(epsilon_medium=None),
        device=device,
    )
    times, reports = [], []
    for task in tasks:
        z, y = task["splits"]["train"]
        t0 = time.time()
        rec = model.write(Batch(z.to(device), y.to(device), task=task["task_id"]))
        times.append(time.time() - t0)
        reports.append(
            {
                "reversibility": rec.reversibility_report.get("pass"),
                "order": rec.order_report.get("pass"),
                "order_max_abs_dW": rec.order_report.get("max_abs_dW_permutation"),
                "purity": rec.purity_report.get("pass"),
                "locality_flip_rate": rec.locality_report.get("flip_rate"),
            }
        )
    z_all = torch.cat([k["splits"]["test"][0] for k in tasks]).to(device)
    api_logits = seen_mask(model.stats.predict(z_all), C)
    # Forget the last task and compare with a stream that never saw it.
    last = model.log[-1].id
    forget = model.forget(last)
    return {
        "max_abs_dlogit_vs_direct": float((api_logits - direct_logits).abs().max()),
        "argmax_identical": bool(
            torch.equal(api_logits.argmax(-1), direct_logits.argmax(-1))
        ),
        "write_seconds": times,
        "write_reports": reports,
        "forget_last": {k: forget.get(k) for k in ("pass", "downdate")},
        "storage": model.stats.storage_bytes(),
    }


# -- one cell ---------------------------------------------------------------------------


def run_cell(cache, benchmark, backbone, seed, args, device) -> dict:
    spec = BENCHMARKS.get(benchmark)
    C = int(cache["train"][1].max()) + 1
    order = class_order(C, spec.shuffle if spec else True)
    init_cls = args.init_cls or (spec.init_cls if spec else 5)
    increment = args.increment or (spec.increment if spec else 5)
    tasks = split_tasks(cache["train"], cache["test"], order, init_cls, increment)
    fm = RandomProjection(cache["train"][0].size(1), args.M, seed=seed)
    z0, y0 = tasks[0]["splits"]["train"]
    sel = {
        "ridge": select_ridge(z0, y0, C, seed=seed, device=device),
        "rp": select_ridge(z0, y0, C, feature_map=fm, seed=seed, device=device),
    }
    lam = {k: v["ridge"] for k, v in sel.items()}
    arms = analytic_arms(tasks, C, lam, fm, device)
    banks = {}
    if not args.no_bank:
        recipe = dict(S11_RECIPE, epochs=args.bank_epochs)
        banks["pal_l3"] = own_bank_dump(tasks, C, seed, recipe, device)
    banks.update(external_dumps(args.external_dir, benchmark, backbone, seed))
    y_test = torch.cat([k["splits"]["test"][1] for k in tasks])
    decomp = {}
    for name, dump in banks.items():
        if not torch.equal(dump.y, y_test):
            raise SystemExit(f"veto: {name} dump rows are not the cache's test order")
        decomp[name] = {
            r: decompose(dump, arms["final_logits"][r].cpu()) for r in READOUTS
        }
    cell = {
        "benchmark": benchmark,
        "backbone": backbone,
        "seed": seed,
        "num_tasks": len(tasks),
        "split": [init_cls, increment],
        "ridge_selection": {k: {"ridge": v["ridge"]} for k, v in sel.items()},
        "readouts": arms["summary"],
        "banks": decomp,
        "bank_meta": {k: v.meta for k, v in banks.items()},
    }
    if args.api:
        cell["api"] = api_arm(
            tasks, C, lam["rp"], args.M, seed, device, arms["final_logits"]["rp"]
        )
    return cell


def vetoes(cells) -> dict:
    ident = max(
        (
            d["decomposition"]["identity_abs_error"]
            for c in cells
            for b in c["banks"].values()
            for d in b.values()
        ),
        default=0.0,
    )
    r_not_tau = sum(
        d["decomposition"]["counts"]["r_not_tau"]
        for c in cells
        for b in c["banks"].values()
        for d in b.values()
    )
    out = {
        "identity_max": ident,
        "identity_pass": ident <= IDENTITY_BAND,
        "r_not_tau": r_not_tau,
        "r_not_tau_pass": r_not_tau == 0,
    }
    api = [c["api"] for c in cells if "api" in c]
    if api:
        out["api_max_abs_dlogit"] = max(a["max_abs_dlogit_vs_direct"] for a in api)
        out["api_pass"] = all(
            a["argmax_identical"] and a["max_abs_dlogit_vs_direct"] <= API_BAND
            for a in api
        )
        out["guards_pass"] = all(
            r["reversibility"] and r["order"] and r["purity"]
            for a in api
            for r in a["write_reports"]
        ) and all(a["forget_last"]["pass"] for a in api)
    return out


def aggregate(cells) -> dict:
    groups = {}
    for c in cells:
        groups.setdefault((c["benchmark"], c["backbone"]), []).append(c)
    out = {}
    for (bm, bb), cs in groups.items():
        key = f"{bm}__{bb}"
        agg = {"n_seeds": len(cs), "readouts": {}, "contrasts": {}, "banks": {}}
        for r in READOUTS:
            agg["readouts"][r] = {
                m: sum(c["readouts"][r][m] for c in cs) / len(cs)
                for m in ("final", "avg_incremental")
            }
        agg["contrasts"]["rp_minus_ridge_final"] = paired_stats(
            [
                c["readouts"]["rp"]["final"] - c["readouts"]["ridge"]["final"]
                for c in cs
            ],
            "rp - ridge (final)",
        )
        for bank in cs[0]["banks"]:
            agg["banks"][bank] = {}
            for r in READOUTS:
                ds = [c["banks"][bank][r] for c in cs]
                agg["banks"][bank][r] = {
                    "P2": paired_stats(
                        [d["decomposition"]["P2_pooled"] for d in ds],
                        f"P2 {bank} routed by {r}",
                        sesoi=0.01,
                    ),
                    **{
                        k: sum(d["decomposition"][k] for d in ds) / len(ds)
                        for k in ("m", "rho", "beta", "rescue", "break", "P2_max")
                    },
                    **{
                        k: sum(d[k] for d in ds) / len(ds)
                        for k in ("readout_acc", "system_acc", "oracle_acc")
                    },
                }
        out[key] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--benchmarks", default="cifar100")
    ap.add_argument("--backbones", default="in21k_1k")
    ap.add_argument("--cache_dir", default="results/feature_cache/ptm")
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--init_cls", type=int, default=0, help="0: the pinned split")
    ap.add_argument("--increment", type=int, default=0, help="0: the pinned split")
    ap.add_argument("--M", type=int, default=10000)
    ap.add_argument("--bank_epochs", type=int, default=S11_RECIPE["epochs"])
    ap.add_argument("--no_bank", action="store_true")
    ap.add_argument("--external_dir", default=None)
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/ptm_cil/ptm_cil_study.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    cells = []
    if args.synthetic:
        args.init_cls, args.increment = 5, 5
        args.M = min(args.M, 256)
        args.bank_epochs = min(args.bank_epochs, 2)
        for s in seeds:
            cells.append(
                run_cell(synthetic_cache(), "synthetic", "none", s, args, args.device)
            )
    else:
        for bm in args.benchmarks.split(","):
            for bb in args.backbones.split(","):
                cache = load_raw_cache(Path(args.cache_dir) / f"{bm}__{bb}.pt")
                for s in seeds:
                    cells.append(run_cell(cache, bm, bb, s, args, args.device))
                    print(
                        f"[{bm} {bb} seed {s}] done {time.time() - t0:.0f}s", flush=True
                    )
    v = vetoes(cells)
    result = {
        "prereg": PREREG,
        "git": git_rev(),
        "args": vars(args),
        "seconds": round(time.time() - t0, 1),
        "vetoes": v,
        "aggregate": aggregate(cells),
        "cells": cells,
    }
    out = Path(
        args.out if not args.synthetic else "results/ptm_cil/synthetic_smoke.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    failed = [k for k, val in v.items() if k.endswith("_pass") and not val]
    for key, agg in result["aggregate"].items():
        ro = agg["readouts"]
        print(
            key,
            " ".join(
                f"{r}={ro[r]['final']:.4f}/{ro[r]['avg_incremental']:.4f}"
                for r in READOUTS
            ),
        )
        for bank, per in agg["banks"].items():
            for r, d in per.items():
                print(
                    f"  {bank} routed by {r}: P2={d['P2']['mean']:+.4f} m={d['m']:.4f} "
                    f"rho={d['rho']:.3f} beta={d['beta']:.4f}"
                )
    print("vetoes:", json.dumps(v))
    if failed:
        raise SystemExit(f"veto failed: {failed}")


if __name__ == "__main__":
    main()
