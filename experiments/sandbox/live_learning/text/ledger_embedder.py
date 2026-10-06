"""Dedicated sentence-embedding router keys (pinned in the slice prereg).

Same six paraphrases as the two previous router runs; keys and queries encoded by a
small multilingual MiniLM. Reads: own-key min, best-other max, tau midpoint, recall
under tau (expert chosen by the embedding router, others suspended), abstention on
the capability questions.

    python ledger_embedder.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
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
ENC_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_embedder_15b.json")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(ENC_NAME)
    store = DeltaStore(args.model, steps=16)
    facts = {f.id: f for f in FACTS}
    for f in facts.values():
        store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )
    ids = sorted(store.deltas)
    K = enc.encode([store.keys[i] for i in ids], normalize_embeddings=True)
    own, other, chosen = {}, {}, {}
    for fid, q in PARAPHRASES.items():
        v = enc.encode([q], normalize_embeddings=True)[0]
        sims = K @ v
        j = int(sims.argmax())
        own[fid] = float(sims[ids.index(fid)])
        other[fid] = float(np.max(np.delete(sims, ids.index(fid))))
        chosen[fid] = ids[j]
    tau = (min(own.values()) + max(other.values())) / 2

    recall = {}
    for fid, q in PARAPHRASES.items():
        if float(np.max([float(K[i] @ enc.encode([q], normalize_embeddings=True)[0]) for i in range(len(ids))])) < tau:
            recall[fid] = False  # abstained
            continue
        full = dict(store.deltas)
        store.deltas = {chosen[fid]: full[chosen[fid]]}
        store.materialize()
        resp = store.answer(q)
        store.deltas = full
        store.materialize()
        recall[fid] = facts[fid].answer.lower() in resp.lower()

    caps = [q for _, q, _ in CAPABILITY]
    C = enc.encode(caps, normalize_embeddings=True) @ K.T
    abstain = {cap: bool(C[j].max() < tau) for j, cap in enumerate(caps)}
    result = {
        "model": args.model,
        "encoder": ENC_NAME,
        "tau": tau,
        "own": own,
        "best_other": other,
        "chosen": chosen,
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


def _unused():
    return np


if __name__ == "__main__":
    main()
