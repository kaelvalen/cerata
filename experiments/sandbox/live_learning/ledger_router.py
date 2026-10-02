"""Router separation and abstention (docs/LIVING_MODEL_SLICE_PREREG.md).

Builds the 18-fact store, measures TF-IDF similarity margins (in-scope vs
out-of-scope), pins tau at the midpoint, and checks fact recall under tau plus
abstention on the capability questions.

    python ledger_router.py --facts 18
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
from stream import CAPABILITY, FACTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--facts", type=int, default=18)
    ap.add_argument("--out", default="results/live_learning/ledger_router_15b.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=16)
    facts = FACTS[: args.facts]
    for f in facts:
        store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )
    ids = sorted(store.deltas)
    keys = [store.keys[i] for i in ids]
    caps = [q for _, q, _ in CAPABILITY]
    queries = [f.probe for f in facts] + caps
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x = vec.fit_transform(keys + queries)
    sim = cosine_similarity(x[len(keys) :], x[: len(keys)])  # [queries, keys]

    key_index = {fid: j for j, fid in enumerate(ids)}
    in_scope = [float(sim[i, key_index[facts[i].id]]) for i in range(len(facts))]
    confusion = [
        float(np.max(np.delete(sim[i], key_index[facts[i].id]))) if len(ids) > 1 else 0.0
        for i in range(len(facts))
    ]
    out_scope = [float(sim[len(facts) + j].max()) for j in range(len(caps))]
    tau = (min(in_scope) + max(out_scope)) / 2

    store.tau = tau
    recall = {
        f.id: f.answer.lower() in store.answer_routed(f.probe).lower() for f in facts
    }
    abstain = {
        cap: float(sim[len(facts) + j].max()) < tau for j, cap in enumerate(caps)
    }
    result = {
        "model": args.model,
        "facts": len(facts),
        "tau": tau,
        "in_scope_min": min(in_scope),
        "confusion_max": max(confusion),
        "out_scope_max": max(out_scope),
        "recall_under_tau": recall,
        "recall_rate": sum(recall.values()) / len(recall),
        "abstention": abstain,
        "abstention_rate": sum(abstain.values()) / len(abstain),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"tau {tau:.3f} | in-scope min {min(in_scope):.3f} | out-scope max "
        f"{max(out_scope):.3f} | recall {sum(recall.values())}/{len(recall)} | "
        f"abstain {sum(abstain.values())}/{len(abstain)}"
    )


if __name__ == "__main__":
    main()
