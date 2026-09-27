"""
EASE (Zhou et al., CVPR 2024) as an `ExpertDump` (docs/PTM_CIL_PREREG.md, section 8).

Runs the official code (github.com/sun-hailong/CVPR24-Ease; clone it and pass
`--ease_repo`) with its published per-dataset, per-backbone configs (`exps/ease_*.json`),
on the pinned split, class order and data (`common.py`). Nothing in the training loop is
changed: `Learner.incremental_train` runs task by task exactly as the official trainer
calls it, the per-task evaluation is skipped, and after the last task one pass over the
test set records

    native_pred       EASE's own prediction (`forward(test=True)`: the reweighted
                      cosine classifier over all adapters' subspaces)
    expert_pred[:, t] the classifier on the features of adapter t only: cosine between
                      adapter t's [CLS] feature and every class's prototype in subspace t
                      (old classes' prototypes there are EASE's own semantic-guided
                      synthesis), argmax over all classes

That is the pre-registered definition of "forced to expert t". A second dump, written
to `<out_dir>/exploratory/` and not read by the study, records EASE's native logits
restricted to task t's classes (`ease-wp`: EASE's own classifier given the task id).

    python experiments/external/ease_export.py --ease_repo ../CVPR24-Ease \
        --benchmark cifar100 --backbone in21k_1k --seeds 0,1,2,3,4,5 --device cuda

`--check` reads the written dumps back and compares their mean native final accuracy
with the official log (`logs/ease/...`) where the official config's split is the pinned
one: the fidelity veto of amendment 2. `--reproduce` instead runs the config's own
split with seed 1993 against that log; no dump is written. `--micro_batch` accumulates
each published batch over smaller chunks when the GPU cannot hold it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from common import (  # noqa: E402  (common puts the repository on sys.path)
    BENCHMARKS,
    TimmShim,
    dump_name,
    dump_summary,
    make_dump,
    official_final_accuracy,
    pilot_data_manager,
    set_random,
    stream_test_loader,
    task_of_class,
    use_repo,
    write_json,
)

CONFIG_STEM = {
    "cifar100": "cifar",
    "cub": "cub",
    "imagenet_a": "ina",
    "imagenet_r": "inr",
    "objectnet": "obj",
    "omnibenchmark": "omni",
    "vtab": "vtab",
}
BACKBONE_TYPE = {
    "in21k_1k": "vit_base_patch16_224_ease",
    "in21k": "vit_base_patch16_224_in21k_ease",
}
PORT_TOLERANCE = 1e-4


def load_config(repo: Path, benchmark: str, backbone: str) -> tuple[str, dict]:
    name = (
        f"ease_{CONFIG_STEM[benchmark]}{'_in21k' if backbone == 'in21k' else ''}.json"
    )
    cfg = json.loads((repo / "exps" / name).read_text())
    if cfg["backbone_type"] != BACKBONE_TYPE[backbone]:
        raise SystemExit(
            f"{name}: backbone_type {cfg['backbone_type']!r} is not {backbone}"
        )
    return name, cfg


@torch.no_grad()
def port_error(net, reference, device, n=4) -> float:
    """max |EASE backbone with its fresh (zero-output) adapter - stock timm ViT| on
    random images: the weight conversion and the timm 1.x port in one number."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 3, 224, 224, generator=g).to(device)
    net.eval()
    reference = reference.to(device).eval()
    return float(
        (net.backbone.forward_proto(x, adapt_index=0) - reference(x)).abs().max()
    )


@torch.no_grad()
def forced_predictions(net, num_tasks, task_bounds, loader, device):
    """(native_pred [N], expert_pred [N, T], within_task_pred [N, T])."""
    net.eval()
    D, U = net.out_dim, int(bool(net.use_init_ptm))
    W = net.fc.weight
    if W.size(1) != (num_tasks + U) * D:
        raise RuntimeError(f"fc has {W.size(1)} inputs, expected {(num_tasks + U) * D}")
    if net.fc.sigma is not None and float(net.fc.sigma) <= 0:
        raise RuntimeError("non-positive sigma would flip the cosine argmax")
    Wn = [
        F.normalize(W[:, (t + U) * D : (t + U + 1) * D], dim=1)
        for t in range(num_tasks)
    ]
    native, forced, within = [], [], []
    for x, _ in loader:
        out = net(x.to(device), test=True)
        logits, feats = out["logits"], out["features"]
        native.append(logits.argmax(-1).cpu())
        cols = []
        for t in range(num_tasks):
            f = F.normalize(feats[:, (t + U) * D : (t + U + 1) * D], dim=1)
            cols.append(F.linear(f, Wn[t]).argmax(-1).cpu())
        forced.append(torch.stack(cols, 1))
        wp = []
        for lo, hi in task_bounds:
            wp.append((logits[:, lo:hi].argmax(-1) + lo).cpu())
        within.append(torch.stack(wp, 1))
    return torch.cat(native), torch.cat(forced), torch.cat(within)


def accumulating_init_train(learner, micro: int):
    """The official `Learner._init_train` with each batch's gradient accumulated over
    chunks of `micro` samples: sum_i (n_i / n) * grad CE_mean(chunk_i) is the full
    batch's gradient, so the optimiser sees the published batch size (only the adapter
    dropout draws differ). For GPUs that cannot hold the published batch; logging of
    the per-epoch loss is dropped."""

    def _init_train(train_loader, test_loader, optimizer, scheduler):
        if learner.moni_adam and learner._cur_task > learner.adapter_num - 1:
            return
        first = learner._cur_task == 0 or learner.init_cls == learner.inc
        epochs = learner.args["init_epochs" if first else "later_epochs"]
        for _ in range(epochs):
            learner._network.train()
            for _, inputs, targets in train_loader:
                inputs = inputs.to(learner._device)
                targets = targets.to(learner._device)
                aux = torch.where(
                    targets - learner._known_classes >= 0,
                    targets - learner._known_classes,
                    -1,
                )
                optimizer.zero_grad()
                n = inputs.size(0)
                for xs, ys in zip(inputs.split(micro), aux.split(micro)):
                    logits = learner._network(xs, test=False)["logits"]
                    (F.cross_entropy(logits, ys) * (ys.size(0) / n)).backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

    return _init_train


def run_one(a, repo, commit, benchmark, backbone, seed, init_cls, increment):
    import models.ease as ease_mod
    import timm
    from backbone import vit_ease
    from utils.data import build_transform
    from utils.data_manager import DataManager

    cfg_name, cfg = load_config(repo, benchmark, backbone)
    args = dict(cfg)
    args.update(
        seed=seed,
        init_cls=init_cls,
        increment=increment,
        device=[torch.device(a.device)],
    )
    if a.epochs:  # smoke tests only; recorded in meta and in the file name's directory
        args.update(init_epochs=a.epochs, later_epochs=a.epochs)
    shim = TimmShim(timm, random_weights=a.random_weights)
    vit_ease.timm = shim
    ease_mod.num_workers = a.workers

    set_random(seed)
    dm = pilot_data_manager(
        DataManager, build_transform, benchmark, a.data_root, init_cls, increment, args
    )
    args["nb_classes"], args["nb_tasks"] = dm.nb_classes, dm.nb_tasks
    learner = ease_mod.Learner(args)
    if a.micro_batch:
        learner._init_train = accumulating_init_train(learner, a.micro_batch)
    net = learner._network.to(a.device)
    err = port_error(net, shim.created[-1][1], a.device)
    print(
        f"[{benchmark}/{backbone}/seed{seed}] port error {err:.2e} ({shim.created[-1][0]})"
    )
    if err > PORT_TOLERANCE:
        raise SystemExit(
            f"backbone port error {err:.2e} > {PORT_TOLERANCE}: not running"
        )

    t0 = time.time()
    for task in range(dm.nb_tasks):
        learner.incremental_train(dm)
        if task < dm.nb_tasks - 1:  # after the last task the fresh adapter would be
            learner.after_task()  # appended to the test features; the trainer evaluates first
        print(f"  task {task + 1}/{dm.nb_tasks} done ({time.time() - t0:.0f}s)")
    train_s = time.time() - t0

    loader, y = stream_test_loader(
        benchmark, a.data_root, init_cls, increment, a.eval_batch_size, a.workers
    )
    bounds, lo = [], 0
    for n in dm._increments:
        bounds.append((lo, lo + n))
        lo += n
    native, forced, within = forced_predictions(
        net, dm.nb_tasks, bounds, loader, a.device
    )
    meta = {
        "method": "ease",
        "repository": "github.com/sun-hailong/CVPR24-Ease",
        "commit": commit,
        "config_file": cfg_name,
        "config": {k: v for k, v in args.items() if k != "device"},
        "seed": seed,
        "class_order_seed": 1993,
        "timm_tag": shim.created[-1][0],
        "timm_version": str(timm.__version__),
        "torch_version": str(torch.__version__),
        "port_error": err,
        "forced_expert": "cosine classifier on adapter t's subspace only, all classes",
        "train_seconds": round(train_s, 1),
        "micro_batch": a.micro_batch or None,
        "smoke": bool(a.random_weights or a.epochs),
    }
    toc = task_of_class(benchmark, init_cls, increment)
    return (
        make_dump(y, toc, forced, native, meta),
        make_dump(
            y,
            toc,
            within,
            native,
            dict(
                meta,
                method="ease-wp",
                forced_expert="native logits restricted to task t",
            ),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ease_repo", required=True)
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--backbone", default="in21k_1k", choices=sorted(BACKBONE_TYPE))
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--data_root", default="data/ptm")
    ap.add_argument("--out_dir", default="results/ptm_cil/external")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--init_cls", type=int, default=0, help="0: the pinned split")
    ap.add_argument("--increment", type=int, default=0)
    ap.add_argument(
        "--reproduce",
        action="store_true",
        help="the config's own split, seed 1993, compared with the official log",
    )
    ap.add_argument(
        "--micro_batch",
        type=int,
        default=0,
        help="accumulate each published batch over chunks of this size (GPU memory)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare the written dumps' mean native accuracy with the official log",
    )
    ap.add_argument("--epochs", type=int, default=0, help="smoke tests only")
    ap.add_argument("--random_weights", action="store_true", help="smoke tests only")
    a = ap.parse_args()

    repo = Path(a.ease_repo).resolve()
    commit = use_repo(repo)
    spec = BENCHMARKS[a.benchmark]
    out_dir = Path(a.out_dir)
    if a.epochs or a.random_weights:
        out_dir = out_dir / "smoke"

    if a.reproduce:
        _, cfg = load_config(repo, a.benchmark, a.backbone)
        init_cls, increment = cfg["init_cls"], cfg["increment"]
        dump, _ = run_one(
            a, repo, commit, a.benchmark, a.backbone, 1993, init_cls, increment
        )
        log = official_log(repo, cfg, a.backbone)
        ours = dump_summary(dump)["native_acc"]
        theirs = official_final_accuracy(log)
        rep = {
            "benchmark": a.benchmark,
            "backbone": a.backbone,
            "split": [init_cls, increment],
            "native_final_acc": ours,
            "official_final_acc": theirs,
            "official_log": str(log.relative_to(repo)),
            "within_2pp": abs(ours - theirs) <= 0.02,
            "meta": dump.meta,
        }
        write_json(
            out_dir / "reproduce" / f"ease__{a.benchmark}__{a.backbone}.json", rep
        )
        print(
            f"native {ours:.4f} vs official {theirs:.4f}: within 2 pp = {rep['within_2pp']}"
        )
        return

    init_cls = a.init_cls or spec.init_cls
    increment = a.increment or spec.increment
    if (init_cls, increment) != (spec.init_cls, spec.increment):
        # never under the name the runner reads for the pinned split
        out_dir = out_dir / f"split_{init_cls}_{increment}"
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.check:
        rep = check(repo, a.benchmark, a.backbone, seeds, out_dir, init_cls, increment)
        write_json(out_dir / "checks" / f"ease__{a.benchmark}__{a.backbone}.json", rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "per_seed"}, indent=1))
        return
    for seed in seeds:
        path = out_dir / dump_name("ease", a.benchmark, a.backbone, seed)
        if path.exists():
            print(f"{path} exists; skipping")
            continue
        dump, wp = run_one(
            a, repo, commit, a.benchmark, a.backbone, seed, init_cls, increment
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        dump.save(path)
        (out_dir / "exploratory").mkdir(exist_ok=True)
        wp.save(
            out_dir
            / "exploratory"
            / dump_name("ease-wp", a.benchmark, a.backbone, seed)
        )
        s = dump_summary(dump)
        print(
            f"wrote {path}: native {s['native_acc']:.4f}, forced-expert oracle "
            f"{s['oracle_acc']:.4f}"
        )


def official_log(repo: Path, cfg: dict, backbone: str) -> Path:
    """The official run log of a config (its own split, seed 1993)."""
    split = 0 if cfg["init_cls"] == cfg["increment"] else cfg["init_cls"]
    d = repo / "logs" / "ease" / cfg["dataset"] / str(split) / str(cfg["increment"])
    found = sorted(d.glob(f"*_1993_{BACKBONE_TYPE[backbone]}.log"))
    if not found:
        raise FileNotFoundError(f"no official log for {backbone} in {d}")
    return found[0]


def check(repo, benchmark, backbone, seeds, out_dir, init_cls, increment) -> dict:
    """The EASE fidelity veto (docs/PTM_CIL_PREREG.md, amendment 2): the mean native
    final accuracy over the written seeds against the official log, where the
    official config's split equals the one run here; reported, not vetoed, elsewhere."""
    from cerata.eval.decomposition import ExpertDump

    _, cfg = load_config(repo, benchmark, backbone)
    per_seed = {}
    for seed in seeds:
        path = out_dir / dump_name("ease", benchmark, backbone, seed)
        if path.exists():
            d = ExpertDump.load(path)
            per_seed[seed] = dict(dump_summary(d), smoke=d.meta.get("smoke"))
    if not per_seed:
        raise SystemExit(f"no ease dumps for {benchmark}/{backbone} in {out_dir}")
    mean = sum(v["native_acc"] for v in per_seed.values()) / len(per_seed)
    log = official_log(repo, cfg, backbone)
    official = official_final_accuracy(log)
    comparable = (cfg["init_cls"], cfg["increment"]) == (init_cls, increment)
    return {
        "benchmark": benchmark,
        "backbone": backbone,
        "seeds": sorted(per_seed),
        "native_final_acc_mean": mean,
        "official_final_acc": official,
        "official_split": [cfg["init_cls"], cfg["increment"]],
        "split": [init_cls, increment],
        "comparable": comparable,
        "veto": (
            ("pass" if abs(mean - official) <= 0.02 else "fail")
            if comparable
            else "not applicable"
        ),
        "any_smoke": any(v["smoke"] for v in per_seed.values()),
        "per_seed": per_seed,
    }


if __name__ == "__main__":
    main()
