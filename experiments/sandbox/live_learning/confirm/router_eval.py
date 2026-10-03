"""Router-only evaluation (CPU, MiniLM only; pinned): entity-aware abstention.

Compares router v1 (pure semantic, pilot A) with v2 (entity-aware, pinned in the
prereg) on the nonce set: probe route accuracy, paraphrase route accuracy, distractor
abstention. No model, no GPU.

    python router_eval.py --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import generate  # noqa: E402
from pilot import TAU, Router  # noqa: E402


class RouterV1:
    """Pilot A's router: pure semantic, argmax + tau, no entity gate."""

    def __init__(self, enc, facts, tau=TAU):
        self.tau = tau
        self.keys = enc.encode([f["probe"] for f in facts], normalize_embeddings=True)

    def route(self, q, enc):
        v = enc.encode([q], normalize_embeddings=True)[0]
        sims = self.keys @ v
        j = int(sims.argmax())
        return (j if float(sims[j]) >= self.tau else None), float(sims[j])


def evaluate(router, facts, enc):
    probe = [router.route(f["probe"], enc)[0] == i for i, f in enumerate(facts)]
    para = [router.route(f["paraphrase"], enc)[0] == i for i, f in enumerate(facts)]
    abst = [router.route(f["distractor"], enc)[0] is None for f in facts]
    return {
        "probe_route_accuracy": round(float(np.mean(probe)), 4),
        "paraphrase_route_accuracy": round(float(np.mean(para)), 4),
        "distractor_abstention": round(float(np.mean(abst)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    facts = generate(args.n, args.n)
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    out = {
        "n": args.n,
        "tau": TAU,
        "v1_semantic": evaluate(RouterV1(enc, facts), facts, enc),
        "v2_entity": evaluate(Router(enc, facts), facts, enc),
    }
    path = Path(args.out or f"results/live_learning/confirm/router_eval_n{args.n}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
