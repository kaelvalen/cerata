"""Router stress: aliases, typos, pronouns (review item 2; CPU-only; pinned).

Nonce N=50 set. For every probe, perturbations of the subject: typo swap, typo drop,
lowercase, hyphen->space, partial (first token), and a pronoun form with the subject
removed. Metrics per perturbation: entity-gated route accuracy and abstain rate (the
current router), plus the semantic-only argmax accuracy (what an embedding-based
entity match would see).

    python router_stress.py --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from facts import generate  # noqa: E402
from pilot import TAU, Router  # noqa: E402

KINDS = ["typo_swap", "typo_drop", "lower", "space", "partial", "pronoun"]


def perturb_subject(subject: str, kind: str) -> str:
    if kind == "typo_swap" and len(subject) > 4:
        k = len(subject) // 2
        return subject[: k - 1] + subject[k] + subject[k - 1] + subject[k + 1 :]
    if kind == "typo_drop":
        return subject[:-1]
    if kind == "lower":
        return subject.lower()
    if kind == "space":
        return subject.replace("-", " ")
    if kind == "partial":
        return subject.split("-")[0] if "-" in subject else subject[: len(subject) // 2]
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    facts = generate(args.n, args.n)

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
    )
    router = Router(enc, facts)

    keys = enc.encode([f["probe"] for f in facts], normalize_embeddings=True)
    report = {}
    for kind in KINDS:
        routed, abstained, semantic_ok = [], [], []
        for i, f in enumerate(facts):
            if kind == "pronoun":
                q = f["probe"].replace(f["subject"], "it")
            else:
                q = f["probe"].replace(f["subject"], perturb_subject(f["subject"], kind))
            j, _ = router.route(q, enc)
            routed.append(j == i)
            abstained.append(j is None)
            v = enc.encode([q], normalize_embeddings=True)[0]
            semantic_ok.append(int((keys @ v).argmax()) == i)
        report[kind] = {
            "route_accuracy": round(float(np.mean(routed)), 4),
            "abstain_rate": round(float(np.mean(abstained)), 4),
            "semantic_only_accuracy": round(float(np.mean(semantic_ok)), 4),
        }
        print(kind, json.dumps(report[kind]), flush=True)

    out = Path(args.out or f"results/live_learning/confirm/router_stress_n{args.n}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": args.n, "tau": TAU, "kinds": report}, ensure_ascii=False, indent=1)
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
