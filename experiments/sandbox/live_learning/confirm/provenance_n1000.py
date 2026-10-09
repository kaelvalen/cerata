"""Provenance (G5 leave-one-out) at N=1000 on the saved store (pinned SS62).

Loads cf_n1000.ckpt, samples facts, and for each: serve with its expert, suspend
the expert (remove + materialize), serve again, restore. Attribution = the answer
is present with the expert and absent without it. The state hash must be identical
before and after the sweep.

    python provenance_n1000.py --ckpt results/live_learning/confirm/cf_n1000.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from pilot import PilotStore, base_answer, hit, serve_expert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--facts",
        default="results/live_learning/confirm/external_counterfact_n1000.json",
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    data = json.loads(Path(args.facts).read_text())
    facts = data["facts"] if isinstance(data, dict) else data
    store = PilotStore(args.model, steps=16)
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    store.deltas = st["deltas"]
    store.keys = st["keys"]
    store.materialize()
    h0 = store.state_hash()

    sample = facts[: args.sample]
    attributed, times = [], []
    for f in sample:
        t = time.time()
        a1 = serve_expert(store, f["id"], f["probe"])
        saved = dict(store.deltas)
        del store.deltas[f["id"]]
        store.materialize()
        a2 = base_answer(store, f["probe"])
        store.deltas = saved
        store.materialize()
        times.append(time.time() - t)
        attributed.append(hit(a1, f["answer"]) and not hit(a2, f["answer"]))
        if len(attributed) % 25 == 0:
            print(f"prov: {len(attributed)}/{len(sample)}", flush=True)

    out = {
        "ckpt": str(args.ckpt),
        "sample": len(sample),
        "total": len(store.deltas),
        "attribution_rate": round(float(np.mean(attributed)), 4),
        "hash_stable": bool(store.state_hash() == h0),
        "state_hash": h0,
        "mean_seconds": round(float(np.mean(times)), 3),
    }
    path = Path(args.out or args.ckpt.replace(".ckpt", f"_prov{args.sample}.json"))
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    print("saved:", path)


if __name__ == "__main__":
    main()
