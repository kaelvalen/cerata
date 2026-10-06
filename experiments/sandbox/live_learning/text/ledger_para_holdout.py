"""Paraphrase augmentation vs a held-out set (pinned in the slice prereg).

Training: each delta keeps its exact probe pair and, for six facts, one paraphrase
pair (set A). Test: six never-trained paraphrases (set B), routed by the MiniLM key
router; metrics: routing accuracy (chosen == fact) and answer recall on B.

    python ledger_para_holdout.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from ledger import DeltaStore  # noqa: E402
from stream import FACTS  # noqa: E402

TRAIN_PARAPHRASES = {  # set A (seen in training)
    "p1": "Adım neydi?",
    "p2": "Köpeğimin adı neydi?",
    "p4": "Hangi şehirde yaşıyorum?",
    "w1": "Zeta-9 ne zaman fırlatıldı acaba?",
    "c1": "Kırmızı dosyanın kodu nedir?",
    "c2": "İptal etmeden önce hangi komut?",
}
HOLDOUT = {  # set B (never trained)
    "p1": "İsmim ne?",
    "p2": "Köpeğim hangi isimle çağrılıyor?",
    "p4": "Yaşadığım şehir hangisi?",
    "w1": "Zeta-9'un fırlatma yılı nedir?",
    "c1": "Kırmızı dosya kodu kaç?",
    "c2": "İptal komutundan önce ne çalıştırılır?",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_para_holdout_15b.json")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    store = DeltaStore(args.model, steps=16)
    facts = {f.id: f for f in FACTS}
    for f in facts.values():
        pairs = [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)]
        if f.id in TRAIN_PARAPHRASES:
            pairs.append((TRAIN_PARAPHRASES[f.id], f.answer))
        store.add(f.id, pairs, key=f.probe)

    ids = sorted(store.deltas)
    K = enc.encode([store.keys[i] for i in ids], normalize_embeddings=True)
    V = enc.encode(list(HOLDOUT.values()), normalize_embeddings=True)
    sims = V @ K.T
    chosen, recall = {}, {}
    for row, (fid, q) in enumerate(HOLDOUT.items()):
        j = int(sims[row].argmax())
        chosen[fid] = ids[j]
        full = dict(store.deltas)
        store.deltas = {ids[j]: full[ids[j]]}
        store.materialize()
        resp = store.answer(q)
        store.deltas = full
        store.materialize()
        recall[fid] = facts[fid].answer.lower() in resp.lower()
    result = {
        "model": args.model,
        "holdout": HOLDOUT,
        "chosen": chosen,
        "routing_correct": sum(chosen[f] == f for f in HOLDOUT),
        "recall": recall,
        "recall_rate": sum(recall.values()) / len(recall),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"routing {result['routing_correct']}/{len(HOLDOUT)} | "
        f"recall {sum(recall.values())}/{len(recall)}"
    )


def _unused() -> None:
    return np


if __name__ == "__main__":
    main()
