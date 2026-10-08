"""Router stress: aliases, typos, pronouns (review item 2; CPU-only; pinned).

Nonce N=50 set. For every probe, perturbations of the subject: typo swap, typo drop,
lowercase, hyphen->space, partial (first token), and a pronoun form with the subject
removed. Metrics per perturbation: entity-gated route accuracy and abstain rate for
the substring gate (v2) and the fuzzy entity match (v3), plus the router-independent
semantic-only argmax accuracy.

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
from pilot import TAU, Router, RouterV2  # noqa: E402

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


def perturbed_probe(f, kind: str) -> str:
    if kind == "pronoun":
        return f["probe"].replace(f["subject"], "it")
    return f["probe"].replace(f["subject"], perturb_subject(f["subject"], kind))


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
    routers = {"v2_substring": RouterV2(enc, facts), "v3_fuzzy": Router(enc, facts)}
    keys = enc.encode([f["probe"] for f in facts], normalize_embeddings=True)

    report = {}
    for kind in KINDS:
        queries = [perturbed_probe(f, kind) for f in facts]
        emb = enc.encode(queries, normalize_embeddings=True)
        semantic_ok = [(keys @ v).argmax() == i for i, v in enumerate(emb)]
        sem = round(float(np.mean(semantic_ok)), 4)
        for name, router in routers.items():
            routed, abstained = [], []
            for i, q in enumerate(queries):
                j, _ = router.route(q, enc)
                routed.append(j == i)
                abstained.append(j is None)
            report.setdefault(name, {})[kind] = {
                "route_accuracy": round(float(np.mean(routed)), 4),
                "abstain_rate": round(float(np.mean(abstained)), 4),
                "semantic_only_accuracy": sem,
            }
            print(name, kind, json.dumps(report[name][kind]), flush=True)

    out = Path(args.out or f"results/live_learning/confirm/router_stress_v2v3_n{args.n}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": args.n, "tau": TAU, "kinds": report}, ensure_ascii=False, indent=1)
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
