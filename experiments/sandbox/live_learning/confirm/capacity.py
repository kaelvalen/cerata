"""Multi-fact delta capacity (review item 5; pinned).

One delta trained on the union of k facts' pairs, k in {1, 4, 16, 64}; readings:
in-group efficacy and paraphrase (delta active alone), leakage onto 10 held-out facts,
canary KLD of the group delta, training seconds.

    python capacity.py --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from facts import generate  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from pilot import SYSTEM, PilotStore, hit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--facts", default=None, help="external facts JSON (else nonce)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/confirm/capacity_n100.json")
    args = ap.parse_args()
    if args.facts:
        data = json.loads(Path(args.facts).read_text())
        facts = data["facts"] if isinstance(data, dict) else data
    else:
        facts = generate(args.n, args.n)
    outside = facts[64:74]
    store = PilotStore(args.model, steps=16)
    store.kl_prompts = ["What is the capital of France?", "What is 7 times 8?"]

    report = {}
    for k in (1, 4, 16, 64):
        group = facts[:k]
        pairs = [p for f in group for p in fact_pairs(f)]
        t0 = time.time()
        store.add("grp", pairs, key=group[0]["probe"])
        secs = time.time() - t0
        delta = store.deltas["grp"]
        kld = store._expert_kld(delta)
        full = dict(store.deltas)
        store.deltas = {"grp": delta}
        store.materialize()
        eff, para = [], []
        for f in group:
            eff.append(hit(store.answer(f["probe"], system=SYSTEM), f["answer"]))
            para.append(hit(store.answer(f["paraphrase"], system=SYSTEM), f["answer"]))
        leak = [
            hit(store.answer(f["probe"], system=SYSTEM), f["answer"]) for f in outside
        ]
        store.deltas = full
        store.materialize()
        report[k] = {
            "efficacy": round(sum(eff) / len(eff), 3),
            "paraphrase": round(sum(para) / len(para), 3),
            "leakage": round(sum(leak) / len(leak), 3),
            "kld": round(kld, 3),
            "train_seconds": round(secs, 1),
        }
        print(k, json.dumps(report[k]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": args.n, "groups": report}, ensure_ascii=False, indent=1)
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
