"""
Shared plumbing for the external-bank exporters (docs/PTM_CIL_PREREG.md, section 8).

The published banks (EASE, MOS) ship as LAMDA-style repositories: a `DataManager` that
reads hard-coded dataset paths and derives the class order from the run seed, and
backbones that call `timm.create_model` with timm 0.6 names. The exporters run the
official training code unchanged and replace only what the protocol pins:

- the data: our processed splits and our class order (seed 1993, identity for VTAB),
  whatever the run seed, fed to the official `DataManager` class with its own
  transforms (`pilot_data_manager`);
- the task split: the pinned one, not the repository config's (`--init_cls` /
  `--increment` still override it for a reproduction check);
- the weights: the timm 0.6 names resolve to the timm 1.x tags they meant then
  (`TimmShim`), checked numerically against a stock timm model (`port_error`);
- the evaluation: our own pass over the test set in `stream_test_order`, so the dump's
  rows line up with the feature cache the runner reads.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from cerata.data.ptm_benchmarks import (  # noqa: E402
    BENCHMARKS,
    class_order,
    eval_transform,
    image_datasets,
    stream_test_order,
    task_increments,
)
from cerata.eval.decomposition import ExpertDump  # noqa: E402
from extract_ptm_features import BACKBONES  # noqa: E402

# timm 0.6.12 (the version the official repositories pin) resolved these untagged
# names to the tags below; timm 1.x resolves `vit_base_patch16_224` to a different
# checkpoint (augreg2), so the name is mapped explicitly, never left to the default.
LEGACY_TIMM = {
    "vit_base_patch16_224": BACKBONES["in21k_1k"],
    "vit_base_patch16_224_in21k": BACKBONES["in21k"],
}


class TimmShim:
    """Stands in for the `timm` module inside an official backbone file.

    Every `create_model` goes through `LEGACY_TIMM` (an unknown name fails instead of
    loading timm 1.x's default tag) and is kept in `created`, so the port can be
    checked against the exact reference model. `random_weights` is for offline smoke
    tests only; a dump written with it is marked as such.
    """

    def __init__(self, timm_module, random_weights: bool = False):
        self._timm = timm_module
        self.random_weights = random_weights
        self.created: list[tuple[str, torch.nn.Module]] = []

    def create_model(self, name, pretrained=False, **kwargs):
        if name not in LEGACY_TIMM:
            raise KeyError(f"no pinned timm tag for {name!r}; add it to LEGACY_TIMM")
        tag = LEGACY_TIMM[name]
        model = self._timm.create_model(
            tag, pretrained=pretrained and not self.random_weights, **kwargs
        )
        self.created.append((tag, model))
        return model

    def __getattr__(self, item):
        return getattr(self._timm, item)


def use_repo(repo: str | Path) -> str:
    """Put an official repository first on `sys.path`; return its commit."""
    repo = Path(repo).resolve()
    if not (repo / "models").is_dir():
        raise SystemExit(f"{repo} does not look like a LAMDA-style repository")
    sys.path.insert(0, str(repo))
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def set_random(seed: int) -> None:
    """The official trainers' `_set_random`."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _raw(ds):
    """(images or paths, original labels, use_path) of a torchvision dataset."""
    if hasattr(ds, "samples"):  # ImageFolder, as the official loaders build it
        return np.array([p for p, _ in ds.samples]), np.array(ds.targets), True
    return ds.data, np.array(ds.targets), False  # CIFAR-100: uint8 arrays


def label_positions(benchmark: str) -> tuple[list[int], np.ndarray]:
    spec = BENCHMARKS[benchmark]
    order = class_order(spec.num_classes, spec.shuffle)
    pos = np.empty(spec.num_classes, dtype=np.int64)
    pos[order] = np.arange(spec.num_classes)
    return order, pos


def pilot_data_manager(
    DataManager, build_transform, benchmark, data_root, init_cls, increment, args
):
    """An instance of the official `DataManager` class holding our data.

    `_setup_data` is bypassed (it reads fixed paths and seeds the class order with the
    run seed); everything it would set is set here from the pinned protocol, with the
    repository's own `build_transform` for the train and test transforms.
    """
    spec = BENCHMARKS[benchmark]
    order, pos = label_positions(benchmark)
    train, test = image_datasets(benchmark, data_root, None)
    dm = DataManager.__new__(DataManager)
    dm.args = args
    dm.dataset_name = benchmark
    dm._train_data, ytr, dm.use_path = _raw(train)
    dm._test_data, yte, _ = _raw(test)
    dm._train_targets, dm._test_targets = pos[ytr], pos[yte]
    dm._train_trsf = build_transform(True, args)
    dm._test_trsf = build_transform(False, args)
    dm._common_trsf = []
    dm._class_order = order
    dm._increments = task_increments(spec.num_classes, init_cls, increment)
    return dm


def stream_test_loader(benchmark, data_root, init_cls, increment, batch_size, workers):
    """The test set in `stream_test_order`, through the evaluation transform, and its
    labels in class-order positions."""
    order, pos = label_positions(benchmark)
    _, test = image_datasets(benchmark, data_root, eval_transform())
    y_orig = torch.as_tensor(np.array(test.targets))
    idx = stream_test_order(y_orig, order, init_cls, increment)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test, idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    y = torch.from_numpy(pos[y_orig[idx].numpy()])
    return loader, y


def task_of_class(benchmark, init_cls, increment) -> torch.Tensor:
    incs = task_increments(BENCHMARKS[benchmark].num_classes, init_cls, increment)
    return torch.cat(
        [torch.full((n,), t, dtype=torch.long) for t, n in enumerate(incs)]
    )


def make_dump(y, toc, expert_pred, native_pred, meta) -> ExpertDump:
    dump = ExpertDump(
        y=y.long(),
        task_of_class=toc.long(),
        expert_pred=expert_pred.long(),
        native_pred=native_pred.long(),
        meta=meta,
    )
    dump.validate()
    return dump


def dump_summary(dump: ExpertDump) -> dict:
    """Native accuracy, forced-expert oracle accuracy (expert = true task), and the
    accuracy of every forced expert on its own task's samples."""
    y, toc = dump.y, dump.task_of_class
    true_task = toc[y]
    oracle = dump.expert_pred.gather(1, true_task.unsqueeze(1)).squeeze(1)
    T = dump.expert_pred.size(1)
    own = [
        float(
            (dump.expert_pred[true_task == t, t] == y[true_task == t]).double().mean()
        )
        for t in range(T)
    ]
    return {
        "native_acc": float((dump.native_pred == y).double().mean()),
        "oracle_acc": float((oracle == y).double().mean()),
        "own_task_acc": own,
    }


_CURVE = re.compile(r"CNN top1 curve: \[([^\]]*)\]")


def official_final_accuracy(log_path: str | Path) -> float:
    """The last entry of the last "CNN top1 curve" in an official run log, as a
    fraction."""
    curves = _CURVE.findall(Path(log_path).read_text(errors="replace"))
    if not curves:
        raise ValueError(f"no 'CNN top1 curve' in {log_path}")
    return float(curves[-1].split(",")[-1]) / 100.0


def dump_name(method, benchmark, backbone, seed) -> str:
    return f"{method}__{benchmark}__{backbone}__seed{seed}.npz"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))
