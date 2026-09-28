"""
PTM-CIL A3.3: the representation headroom scan - `docs/PTM_CIL_PREREG.md`, amendment 3.

Per (benchmark, backbone, seed), on the pinned splits:

    frozen   the section 2 readouts (`ridge`, `rp`) on the frozen features (penalties
             selected on task 0, pinned; final accuracy)
    joint    the ladder's `L2a_shared_joint` (`joint=True`: every task at once,
             offline - an upper bound, not a stream) and the same two readouts on
             the adapted features

    gap(r) = final(joint, r) - final(frozen, r)

Reading, fixed before the data (the labels go into `PTM_CIL_RESULTS.md`):

    gap(rp) >= 10 pp   the benchmark is representation-limited: routers and banks
                       inside the frozen regime cannot close it; the PETL line owns
                       it, under its own pre-registration and after the ObjectNet
                       license check (`LITERATURE_UPDATE_2026-09-28.md` section 2)
    gap(rp) <  10 pp   readout-limited: the frozen-regime results are the whole story

Descriptive: 3 seeds (`--seeds 0,1,2`), primary backbone; if the projection would
break the study's 48 h veto, seed 0 only, recorded, never hidden (A3.3). Not a PETL
experiment: no streaming, no pretraining data, no per-task order.

    python experiments/ptm_headroom.py --synthetic
    python experiments/ptm_headroom.py --benchmarks cifar100 --backbones in21k_1k \\
        --seeds 0,1,2 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from cerata.arch.readouts import mask_unseen  # noqa: E402
from cerata.core.features import set_seed  # noqa: E402
from cerata.core.random_features import RandomProjection  # noqa: E402
from cerata.data.ptm_benchmarks import (  # noqa: E402
    BENCHMARKS,
    class_order,
    load_raw_cache,
    split_tasks,
)
from cerata.edit.stats import LinearStats, one_hot, select_ridge  # noqa: E402
from cerata.experts.ladder import LEVELS_BY_NAME, LadderModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ptm_cil  # noqa: E402

PREREG = "docs/PTM_CIL_PREREG.md"
READOUTS = ("ridge", "rp")
GAP_PP = 0.10  # A3.3's fixed threshold
JOINT_LEVEL = "L2a_shared_joint"


@torch.no_grad()
def pinned_readouts(features_of, tasks, C, M, seed, device) -> dict:
    """The section 2 readouts (`ridge`, `rp`) on one representation.

    `features_of` maps a feature batch to the representation under test (frozen:
    identity; joint: the shared adapter). Penalties are selected on task 0 with the
    section 2 rule and pinned for the stream, exactly as in `ptm_cil.run_cell`.
    """
    z0, y0 = tasks[0]["splits"]["train"]
    z0r = features_of(z0).to(device)
    fm = RandomProjection(z0r.size(1), M, seed=seed)
    lam = {
        "ridge": select_ridge(z0r.cpu(), y0, C, seed=seed, device=device)["ridge"],
        "rp": select_ridge(z0r.cpu(), y0, C, feature_map=fm, seed=seed, device=device)[
            "ridge"
        ],
    }
    stats = {
        "ridge": LinearStats(z0r.size(1), C, ridge=lam["ridge"], device=device),
        "rp": LinearStats(
            z0r.size(1), C, ridge=lam["rp"], device=device, feature_map=fm
        ),
    }
    for t, task in enumerate(tasks):
        z, y = task["splits"]["train"]
        zr = features_of(z).to(device)
        for name in READOUTS:
            stats[name].add(
                stats[name].contribution(f"t{t}", zr, one_hot(y.to(device), C))
            )
    z_test = torch.cat([k["splits"]["test"][0] for k in tasks])
    y_test = torch.cat([k["splits"]["test"][1] for k in tasks])
    zr_test = features_of(z_test).to(device)
    out = {}
    for name in READOUTS:
        logits = mask_unseen(stats[name].predict(zr_test), list(range(C)))
        out[name] = {
            "final": float((logits.argmax(-1).cpu() == y_test).float().mean()),
            "ridge": lam[name],
        }
    return out


def train_joint(tasks, C, seed, args, device) -> LadderModel:
    """The S2 joint branch of `L2a_shared_joint`: every task at once, offline.

    The same loop `experiments/s2_ladder.py` uses for `spec.joint`, so the level's
    semantics are the recorded ones; the difference is that this script reads the
    adapted representation with closed-form readouts instead of the model's own.
    """
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    set_seed(seed)
    joint_args = argparse.Namespace(
        rank=8,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        lambda_func=1.0,
        max_experts=max(20, len(tasks)),
    )
    model = LadderModel(LEVELS_BY_NAME[JOINT_LEVEL], dim, C, joint_args, device)
    model.seen = sorted({c for task in tasks for c in task["classes"]})
    for i, task in enumerate(tasks):
        model.register_task(task, i)
    all_z = torch.cat([task["splits"]["train"][0] for task in tasks])
    all_y = torch.cat([task["splits"]["train"][1] for task in tasks])
    model.fit_task(None, 0, joint_data=(all_z, all_y))
    return model


def run_cell(cache, benchmark, backbone, seed, args, device) -> dict:
    spec = BENCHMARKS.get(benchmark)
    C = int(cache["train"][1].max()) + 1
    order = class_order(C, spec.shuffle if spec else True)
    init_cls = args.init_cls or (spec.init_cls if spec else 5)
    increment = args.increment or (spec.increment if spec else 5)
    tasks = split_tasks(cache["train"], cache["test"], order, init_cls, increment)
    t0 = time.time()
    frozen = pinned_readouts(lambda z: z, tasks, C, args.M, seed, device)
    model = train_joint(tasks, C, seed, args, device)
    joint = pinned_readouts(
        lambda z: model.apply_experts(z.to(device), None),
        tasks,
        C,
        args.M,
        seed,
        device,
    )
    gap = {r: joint[r]["final"] - frozen[r]["final"] for r in READOUTS}
    return {
        "benchmark": benchmark,
        "backbone": backbone,
        "seed": seed,
        "num_tasks": len(tasks),
        "split": [init_cls, increment],
        "frozen": {
            r: {"final": frozen[r]["final"], "ridge": frozen[r]["ridge"]}
            for r in READOUTS
        },
        "joint": {
            r: {"final": joint[r]["final"], "ridge": joint[r]["ridge"]}
            for r in READOUTS
        },
        "gap": gap,
        "label": (
            "representation-limited" if gap["rp"] >= GAP_PP else "readout-limited"
        ),
        "seconds": round(time.time() - t0, 1),
    }


def aggregate(cells) -> dict:
    groups = {}
    for c in cells:
        groups.setdefault((c["benchmark"], c["backbone"]), []).append(c)
    out = {}
    for (bm, bb), cs in groups.items():
        entry = {"n_seeds": len(cs), "frozen": {}, "joint": {}, "gap": {}}
        for r in READOUTS:
            entry["frozen"][r] = sum(c["frozen"][r]["final"] for c in cs) / len(cs)
            entry["joint"][r] = sum(c["joint"][r]["final"] for c in cs) / len(cs)
            entry["gap"][r] = sum(c["gap"][r] for c in cs) / len(cs)
        entry["label"] = (
            "representation-limited"
            if entry["gap"]["rp"] >= GAP_PP
            else "readout-limited"
        )
        out[f"{bm}__{bb}"] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--benchmarks", default="cifar100")
    ap.add_argument("--backbones", default="in21k_1k")
    ap.add_argument("--cache_dir", default="results/feature_cache/ptm")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--init_cls", type=int, default=0, help="0: the pinned split")
    ap.add_argument("--increment", type=int, default=0, help="0: the pinned split")
    ap.add_argument("--M", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=ptm_cil.S11_RECIPE["epochs"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/ptm_cil/ptm_headroom.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    cells = []
    if args.synthetic:
        args.init_cls, args.increment = 5, 5
        args.M = min(args.M, 256)
        args.epochs = min(args.epochs, 2)
        for s in seeds:
            cells.append(
                run_cell(
                    ptm_cil.synthetic_cache(), "synthetic", "none", s, args, args.device
                )
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
    result = {
        "status": "ok",
        "prereg": PREREG,
        "git": ptm_cil.git_rev(),
        "args": vars(args),
        "seconds": round(time.time() - t0, 1),
        "aggregate": aggregate(cells),
        "cells": cells,
    }
    out = Path(
        args.out
        if not args.synthetic
        else "results/ptm_cil/synthetic_headroom_smoke.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    for key, agg in result["aggregate"].items():
        print(
            key,
            " ".join(
                f"{r}: frozen={agg['frozen'][r]:.4f} joint={agg['joint'][r]:.4f} "
                f"gap={agg['gap'][r]:+.4f}"
                for r in READOUTS
            ),
            agg["label"],
        )


if __name__ == "__main__":
    main()
