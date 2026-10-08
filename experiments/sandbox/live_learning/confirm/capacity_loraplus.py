"""Capacity with the adopted LoRA+ recipe (pinned, prereg §47).

CounterFact k in {64, 128, 256}: one grouped LoRA+ delta; in-group efficacy and
paraphrase (delta active), leakage on 10 held-out facts, KLD mean/max, train seconds.

    python capacity_loraplus.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from lora_ablation import eval_variant, train_variant  # noqa: E402
from pilot import PilotStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="results/live_learning/confirm/external_counterfact_n1000.json")
    ap.add_argument("--ks", default="64,128,256")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/confirm/capacity_loraplus_n1000.json")
    args = ap.parse_args()
    data = json.loads(Path(args.facts).read_text())
    facts = data["facts"] if isinstance(data, dict) else data
    outside = facts[900:910]
    store = PilotStore(args.model, steps=16)
    store.kl_prompts = ["What is the capital of France?", "What is 7 times 8?"]
    store.model.to("cpu")
    torch.cuda.empty_cache()

    report = {}
    for k in [int(x) for x in args.ks.split(",")]:
        group = facts[:k]
        pairs = [p for f in group for p in fact_pairs(f)]
        t0 = time.time()
        delta = train_variant(store, "lora_plus", pairs)
        secs = time.time() - t0
        ing = eval_variant(store, "lora_plus", {f["id"]: delta for f in group}, group)
        out = eval_variant(store, "lora_plus", {f["id"]: delta for f in outside}, outside)
        report[k] = {
            "efficacy": ing["efficacy"],
            "paraphrase": ing["paraphrase"],
            "kld_mean": ing["kld_mean"],
            "kld_max": ing["kld_max"],
            "leakage": out["efficacy"],
            "train_seconds": round(secs, 1),
        }
        print(k, json.dumps(report[k]), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"facts": args.facts, "groups": report}, ensure_ascii=False, indent=1))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
