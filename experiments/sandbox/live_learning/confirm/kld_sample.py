"""Canary KLD sample over a saved store ckpt (review threshold; short GPU run).

Loads a ckpt (deltas/keys), samples deltas, computes _expert_kld against the
canaries (the same measure the commits gate on) and reports mean/max/p95 vs the
cap (2.0) and the review threshold (0.05).

    python kld_sample.py --ckpt results/live_learning/confirm/cf_n1000.ckpt --sample 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from pilot import CANARIES, PilotStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    store = PilotStore(args.model, steps=16)
    store.kl_prompts = list(CANARIES)
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    store.deltas = st["deltas"]
    store.keys = st["keys"]
    ids = sorted(store.deltas)[: args.sample]
    klds = [float(store._expert_kld(store.deltas[fid])) for fid in ids]
    out = {
        "ckpt": str(args.ckpt),
        "sample": len(ids),
        "total": len(store.deltas),
        "kld_mean": round(float(np.mean(klds)), 4),
        "kld_max": round(float(np.max(klds)), 4),
        "kld_p95": round(float(np.percentile(klds, 95)), 4),
        "cap": 2.0,
        "threshold": 0.05,
    }
    path = Path(args.out or args.ckpt.replace(".ckpt", f"_kld{args.sample}.json"))
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    print("saved:", path)


if __name__ == "__main__":
    main()
