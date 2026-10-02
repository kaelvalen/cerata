"""Controller v0: promote a fact from memory to an expert when it recurs (pinned).

Stream: 6 facts are queried 3 times each, 12 facts once. The memory serves every
query (by construction the probe retrieves its own fact); on the second hit the
controller calls propose_and_commit with the KLD cap 2.0 - the promotion is refused
if its footprint is over budget. Readings: which facts got experts, the expert
count versus always-promote (18) and never-promote (0), the promoted facts' routed
recall, and the refusal count.

    python ledger_controller.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger import DeltaStore  # noqa: E402
from stream import FACTS  # noqa: E402

RECURRING = 6
HITS = 3
THRESHOLD = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_controller_15b.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=16)
    facts = {f.id: f for f in FACTS}
    order = list(facts.values())
    counts: dict[str, int] = {f.id: 0 for f in order}
    promotions, refusals = {}, {}
    stream = []
    for i, f in enumerate(order):
        hits = HITS if i < RECURRING else 1
        stream += [f.id] * hits
    for fid in stream:
        f = facts[fid]
        counts[fid] += 1
        if counts[fid] == THRESHOLD and fid not in promotions:
            res = store.propose_and_commit(
                fid,
                [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
                key=f.probe,
                kld_limit=2.0,
            )
            (promotions if res["committed"] else refusals)[fid] = res["kld"]
    promoted = set(promotions)
    recurring = {f.id for f in order[:RECURRING]}
    recall = {
        fid: facts[fid].answer.lower() in store.answer_routed(facts[fid].probe).lower()
        for fid in sorted(promoted)
    }
    result = {
        "model": args.model,
        "threshold": THRESHOLD,
        "stream_len": len(stream),
        "recurring": sorted(recurring),
        "promoted": sorted(promoted),
        "refused": refusals,
        "promoted_recall": recall,
        "promoted_recall_rate": sum(recall.values()) / max(1, len(recall)),
        "experts": len(store.deltas),
        "experts_always": len(order),
        "experts_never": 0,
        "policy_matches_recurring": promoted == recurring,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"promoted {sorted(promoted)} | refused {sorted(refusals)} | "
        f"recall {sum(recall.values())}/{len(recall)} | experts {len(store.deltas)} "
        f"vs {len(order)} always | policy match {result['policy_matches_recurring']}"
    )


if __name__ == "__main__":
    main()
