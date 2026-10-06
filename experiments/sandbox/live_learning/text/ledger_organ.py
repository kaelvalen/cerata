"""Expert-organ invariants under ledger edits (pinned in the slice prereg).

MiniLM router; 18 facts; revoke p4 and add a new fact; check that no remaining
fact's routing decision moves, the new fact routes to itself, and the revoked fact's
queries no longer land in the revoked expert.

    python ledger_organ.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger import DeltaStore  # noqa: E402
from ledger_embedder import ENC_NAME  # noqa: E402
from stream import FACTS  # noqa: E402

NEW_FACT = ("x1", "Deneme olgusu: mor anahtar 8 numaradır.", "Mor anahtar kaç numaradır?", "8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_organ_15b.json")
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

    def route(queries: dict[str, str]) -> dict[str, str | None]:
        ids = sorted(store.deltas)
        if not ids:
            return {q: None for q in queries}
        K = enc.encode([store.keys[i] for i in ids], normalize_embeddings=True)
        V = enc.encode(list(queries.values()), normalize_embeddings=True)
        sims = V @ K.T
        out = {}
        for row, (qid, _) in enumerate(queries.items()):
            j = int(sims[row].argmax())
            out[qid] = ids[j] if float(sims[row][j]) >= store.tau else None
        return out

    store.tau = 0.656
    r0 = route({fid: f.probe for fid, f in facts.items()})
    store.revoke("p4")
    remaining = {fid: f.probe for fid, f in facts.items() if fid != "p4"}
    r1 = route(remaining)
    moved_on_revoke = [q for q in remaining if r1[q] != r0[q]]
    revoked_still = [q for q, v in route({"p4": facts["p4"].probe}).items() if v == "p4"]

    fid, teach, probe, answer = NEW_FACT
    store.add(fid, [(teach, f"Not aldım: {teach}"), (probe, answer)], key=probe)
    r2 = route(remaining)
    moved_on_add = [q for q in remaining if r2[q] != r0[q]]
    self_route = route({fid: probe})[fid]

    result = {
        "model": args.model,
        "tau": store.tau,
        "revoke_moved_others": moved_on_revoke,
        "add_moved_others": moved_on_add,
        "new_fact_self_route": self_route,
        "revoked_fact_queries_still_on_revoked": revoked_still,
        "pass": not moved_on_revoke
        and not moved_on_add
        and self_route == fid
        and not revoked_still,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"revoke moved others: {len(moved_on_revoke)} | add moved others: "
        f"{len(moved_on_add)} | new self-route {self_route} | revoked still routed: "
        f"{len(revoked_still)} | pass {result['pass']}"
    )


if __name__ == "__main__":
    main()
