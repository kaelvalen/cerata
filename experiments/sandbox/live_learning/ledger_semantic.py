"""Semantic router keys: paraphrase margin and recall (pinned in the slice prereg).

Replaces the TF-IDF similarity with the base model's last-token hidden state for the
same six paraphrases; pins tau at the midpoint of the measured margins and reports
routed recall under it.

    python ledger_semantic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


from ledger import DeltaStore  # noqa: E402
from stream import CAPABILITY, FACTS  # noqa: E402

PARAPHRASES = {
    "p1": "Adım neydi?",
    "p2": "Köpeğimin adı neydi?",
    "p4": "Hangi şehirde yaşıyorum?",
    "w1": "Zeta-9 ne zaman fırlatıldı acaba?",
    "c1": "Kırmızı dosyanın kodu nedir?",
    "c2": "İptal etmeden önce hangi komut?",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_semantic_15b.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=16)
    facts = {f.id: f for f in FACTS}
    for f in facts.values():
        store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )
    ids = sorted(store.deltas)
    own, other = {}, {}
    for fid, q in PARAPHRASES.items():
        qv = store.embed(q)
        sims = {i: float(qv @ store.key_vecs[i]) for i in ids}
        own[fid] = sims[fid]
        other[fid] = max(v for i, v in sims.items() if i != fid)
    tau = (min(own.values()) + max(other.values())) / 2
    store.tau = tau
    recall = {
        fid: facts[fid].answer.lower() in store.answer_routed(q).lower()
        for fid, q in PARAPHRASES.items()
    }
    caps = [q for _, q, _ in CAPABILITY]
    abstain = {
        cap: max(float(store.embed(cap) @ store.key_vecs[i]) for i in ids) < tau
        for cap in caps
    }
    result = {
        "model": args.model,
        "tau": tau,
        "own": own,
        "best_other": other,
        "paraphrase_recall": recall,
        "paraphrase_recall_rate": sum(recall.values()) / len(recall),
        "abstention": abstain,
        "abstention_rate": sum(abstain.values()) / len(abstain),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"tau {tau:.3f} | own min {min(own.values()):.3f} | other max "
        f"{max(other.values()):.3f} | recall {sum(recall.values())}/{len(recall)} | "
        f"abstain {sum(abstain.values())}/{len(abstain)}"
    )


if __name__ == "__main__":
    main()
