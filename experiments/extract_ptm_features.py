"""
Extract frozen ViT features for the seven PTM-CIL benchmarks (docs/PTM_CIL_PREREG.md).

One cache per (benchmark, backbone): raw features with the dataset's ORIGINAL labels,
so the class order and task split are applied at run time
(`pal_moe.data.ptm_benchmarks.split_tasks`) and a protocol change never needs a
re-extraction. Train and test both go through the evaluation transform, as the
frozen-feature methods (SimpleCIL, RanPAC without PETL) do.

Backbones are timm names; the two the literature uses for ViT-B/16:

    in21k       vit_base_patch16_224.augreg_in21k
    in21k_1k    vit_base_patch16_224.augreg_in21k_ft_in1k   (timm 0.6's vit_base_patch16_224)

Needs the `ptm` extra (timm). Usage:

    python experiments/extract_ptm_features.py --benchmark cifar100 \
        --data_root data/ptm --backbone in21k_1k --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.core.hashing import digest, module_digest  # noqa: E402
from pal_moe.data.ptm_benchmarks import (  # noqa: E402
    BENCHMARKS,
    eval_transform,
    image_datasets,
)

BACKBONES = {
    "in21k": "vit_base_patch16_224.augreg_in21k",
    "in21k_1k": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}


def cache_path(out_dir, benchmark, backbone) -> Path:
    return Path(out_dir) / f"{benchmark}__{backbone}.pt"


@torch.no_grad()
def extract(model, dataset, device, batch_size, workers):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).float().cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels).long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    ap.add_argument("--data_root", default="data/ptm")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="in21k_1k")
    ap.add_argument(
        "--normalize", default="none", choices=["none", "imagenet", "inception"]
    )
    ap.add_argument("--out_dir", default="results/feature_cache/ptm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import timm

    t0 = time.time()
    model = timm.create_model(BACKBONES[args.backbone], pretrained=True, num_classes=0)
    model = model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    tf = eval_transform(args.normalize)
    train, test = image_datasets(args.benchmark, args.data_root, tf)
    ztr, ytr = extract(model, train, args.device, args.batch_size, args.workers)
    zte, yte = extract(model, test, args.device, args.batch_size, args.workers)
    out = cache_path(args.out_dir, args.benchmark, args.backbone)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "benchmark": args.benchmark,
        "backbone": args.backbone,
        "timm_name": BACKBONES[args.backbone],
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "normalize": args.normalize,
        "transform": repr(tf),
        "feature_dim": int(ztr.size(1)),
        "num_classes": BENCHMARKS[args.benchmark].num_classes,
        "n_train": int(ztr.size(0)),
        "n_test": int(zte.size(0)),
        "weights_digest": module_digest(model),
        "classes": getattr(train, "classes", None),
        "file_list_digest": digest(
            [s[0] for s in getattr(train, "samples", [])],
            [s[0] for s in getattr(test, "samples", [])],
        ),
        "seconds": round(time.time() - t0, 1),
    }
    torch.save({"meta": meta, "train": (ztr, ytr), "test": (zte, yte)}, out)
    print(f"wrote {out}: train {tuple(ztr.shape)} test {tuple(zte.shape)}")


if __name__ == "__main__":
    main()
