"""The seven pre-trained-model class-incremental benchmarks, as the PTM-CIL literature
runs them (SimpleCIL / APER, RanPAC, EASE, MOS; the LAMDA-PILOT toolbox).

    CIFAR-100      100 classes   torchvision download
    CUB-200        200 classes   ImageFolder  cub/{train,test}
    ImageNet-R     200 classes   ImageFolder  imagenet-r/{train,test}
    ImageNet-A     200 classes   ImageFolder  imagenet-a/{train,test}
    ObjectNet      200 classes   ImageFolder  objectnet/{train,test}
    OmniBenchmark  300 classes   ImageFolder  omnibenchmark/{train,test}
    VTAB            50 classes   ImageFolder  vtab-cil/vtab/{train,test}   (not shuffled)

The ImageFolder splits are the processed releases linked from the RevisitingCIL
repository (github.com/zhoudw-zdw/RevisitingCIL); their md5 sums are in its issue #5.

Protocol details mirrored from LAMDA-PILOT (`utils/data.py`, `utils/data_manager.py`):

- class order: `np.random.seed(1993); np.random.permutation(C)` when shuffled, the
  identity for VTAB; labels are remapped to their position in that order;
- the evaluation transform `Resize(256, bicubic) -> CenterCrop(224) -> ToTensor()`
  with **no normalisation** (PILOT feeds [0, 1] pixels to the timm ViT); frozen-feature
  methods (SimpleCIL, RanPAC without PETL) extract train features with it as well.

Task splits differ between papers; each spec carries the split
`archive/docs/PTM_CIL_PREREG.md` pins (ten tasks, five for VTAB), and the runner can override it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    num_classes: int
    subdir: str | None  # None: torchvision CIFAR-100
    init_cls: int = 10  # the split pinned by archive/docs/PTM_CIL_PREREG.md
    increment: int = 10
    shuffle: bool = True


BENCHMARKS = {
    "cifar100": BenchmarkSpec("cifar100", 100, None, 10, 10),
    "cub": BenchmarkSpec("cub", 200, "cub", 20, 20),
    "imagenet_r": BenchmarkSpec("imagenet_r", 200, "imagenet-r", 20, 20),
    "imagenet_a": BenchmarkSpec("imagenet_a", 200, "imagenet-a", 20, 20),
    "objectnet": BenchmarkSpec("objectnet", 200, "objectnet", 20, 20),
    "omnibenchmark": BenchmarkSpec("omnibenchmark", 300, "omnibenchmark", 30, 30),
    "vtab": BenchmarkSpec("vtab", 50, "vtab-cil/vtab", 10, 10, shuffle=False),
}

PILOT_SEED = 1993


def class_order(num_classes: int, shuffle: bool, seed: int = PILOT_SEED) -> list[int]:
    """PILOT's `_setup_data`: a seeded numpy permutation, or the identity."""
    if not shuffle:
        return list(range(num_classes))
    np.random.seed(seed)
    return np.random.permutation(num_classes).tolist()


def task_increments(num_classes: int, init_cls: int, increment: int) -> list[int]:
    """PILOT's `DataManager.__init__`: the first task, then `increment` each, with any
    remainder as a final smaller task."""
    if init_cls <= 0 or increment <= 0:
        raise ValueError("init_cls and increment must be positive")
    if init_cls > num_classes:
        raise ValueError("not enough classes")
    incs = [init_cls]
    while sum(incs) + increment < num_classes:
        incs.append(increment)
    rest = num_classes - sum(incs)
    if rest > 0:
        incs.append(rest)
    return incs


def eval_transform(normalize: str = "none"):
    from torchvision import transforms

    t = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    if normalize == "imagenet":
        t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    elif normalize == "inception":
        t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif normalize != "none":
        raise KeyError(f"unknown normalisation {normalize!r}")
    return transforms.Compose(t)


def image_datasets(name: str, root: str | Path, transform):
    """(train, test) torchvision datasets for a benchmark under `root`."""
    from torchvision import datasets

    spec = BENCHMARKS[name]
    root = Path(root)
    if spec.subdir is None:
        return (
            datasets.CIFAR100(
                str(root), train=True, download=True, transform=transform
            ),
            datasets.CIFAR100(
                str(root), train=False, download=True, transform=transform
            ),
        )
    base = root / spec.subdir
    train = datasets.ImageFolder(str(base / "train"), transform=transform)
    test = datasets.ImageFolder(str(base / "test"), transform=transform)
    if train.classes != test.classes or len(train.classes) != spec.num_classes:
        raise ValueError(
            f"{base}: expected {spec.num_classes} matching train/test classes, got "
            f"{len(train.classes)} / {len(test.classes)}"
        )
    return train, test


def split_tasks(
    train: tuple[torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor],
    order: list[int],
    init_cls: int,
    increment: int,
) -> list[dict]:
    """Features + original labels -> tasks in the `core.features.load_tasks` format.

    Labels are remapped to their position in `order` (PILOT's `_map_new_class_index`),
    so task t owns a contiguous block of class indices. There is no validation split in
    this protocol: `val` is the training split, and nothing here selects on it.
    """
    pos = torch.empty(len(order), dtype=torch.long)
    pos[torch.tensor(order)] = torch.arange(len(order))
    (ztr, ytr), (zte, yte) = train, test
    ytr, yte = pos[ytr.long()], pos[yte.long()]
    tasks, lo = [], 0
    for t, n in enumerate(task_increments(len(order), init_cls, increment)):
        classes = list(range(lo, lo + n))
        mtr = (ytr >= lo) & (ytr < lo + n)
        mte = (yte >= lo) & (yte < lo + n)
        tr = (ztr[mtr].float(), ytr[mtr])
        tasks.append(
            {
                "task_id": t,
                "classes": classes,
                "splits": {
                    "train": tr,
                    "val": tr,
                    "test": (zte[mte].float(), yte[mte]),
                },
            }
        )
        lo += n
    return tasks


def stream_test_order(
    y_test: torch.Tensor, order: list[int], init_cls: int, increment: int
) -> torch.Tensor:
    """Indices into the original test set, in the order `split_tasks` concatenates its
    tasks' test splits (task by task, original order within a task).

    An external method's `ExpertDump` must list its test samples in this order; an
    exporter reorders its per-sample outputs with `outputs[stream_test_order(...)]`.
    """
    pos = torch.empty(len(order), dtype=torch.long)
    pos[torch.tensor(order)] = torch.arange(len(order))
    y = pos[y_test.long()]
    idx, lo = [], 0
    for n in task_increments(len(order), init_cls, increment):
        idx.append(torch.nonzero((y >= lo) & (y < lo + n)).squeeze(1))
        lo += n
    return torch.cat(idx)


def load_raw_cache(path: str | Path) -> dict:
    """A cache written by `experiments/extract_ptm_features.py`.

    Caches written before 2026-09-27 store `torch.__version__` as a `TorchVersion`
    (a str subclass) in their meta; it is allow-listed so they load with
    `weights_only=True` instead of being re-extracted.
    """
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    for split in ("train", "test"):
        z, y = payload[split]
        payload[split] = (z.float(), y.long())
    return payload
