"""Paraphrase margin + G1 atomic rollback (docs/LIVING_MODEL_SLICE_PREREG.md).

Paraphrase margin: similarities of six hand-written paraphrases to their own key and
the best other key, and routed recall under the recorded tau (0.659).
G1: propose_and_commit with a deliberately strong candidate (kld_limit 2.0) must
refuse the write; a normal candidate must commit.

    python ledger_checks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from ledger import DeltaStore  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from stream import FACTS  # noqa: E402

TAU = 0.659  # recorded in the prereg (router separation run)
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
    ap.add_argument("--out", default="results/live_learning/ledger_checks_15b.json")
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
    keys = [store.keys[i] for i in ids]
    key_index = {fid: j for j, fid in enumerate(ids)}
    queries = list(PARAPHRASES.values())
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x = vec.fit_transform(keys + queries)
    sim = cosine_similarity(x[len(keys) :], x[: len(keys)])
    margin = {}
    for row, fid in enumerate(PARAPHRASES):
        own = float(sim[row, key_index[fid]])
        best_other = float(np.max(np.delete(sim[row], key_index[fid])))
        margin[fid] = {"own": own, "best_other": best_other}
    store.tau = TAU
    recall = {
        fid: facts[fid].answer.lower() in store.answer_routed(q).lower()
        for fid, q in PARAPHRASES.items()
    }

    # -- G1: strong candidate refused, normal candidate committed ----------
    p = facts["p1"]
    pairs1 = [(p.teach, f"Not aldım: {p.teach}"), (p.probe, p.answer)]
    store.lr, store.steps, store.kl_lambda = 1e-3, 32, 0.0
    bad = store.propose_and_commit("bad", pairs1, key=p.probe, kld_limit=2.0)
    q = facts["w1"]
    pairs2 = [(q.teach, f"Not aldım: {q.teach}"), (q.probe, q.answer)]
    store.lr, store.steps, store.kl_lambda = 3e-4, 16, 1.0
    good = store.propose_and_commit("good", pairs2, key=q.probe, kld_limit=2.0)
    result = {
        "model": args.model,
        "tau": TAU,
        "paraphrase_margin": margin,
        "paraphrase_recall": recall,
        "paraphrase_recall_rate": sum(recall.values()) / len(recall),
        "g1_rollback": {"strong_candidate": bad, "normal_candidate": good},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"paraphrase recall {sum(recall.values())}/{len(recall)} | "
        f"own sim min {min(v['own'] for v in margin.values()):.3f} | "
        f"other sim max {max(v['best_other'] for v in margin.values()):.3f} | "
        f"G1 rollback strong {bad['committed']} (kld {bad['kld']:.2f}) "
        f"normal {good['committed']} (kld {good['kld']:.2f})"
    )


if __name__ == "__main__":
    main()
