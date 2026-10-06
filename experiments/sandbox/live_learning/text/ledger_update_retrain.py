"""Text update retrain path with a promoted fact (pinned).

add p3 (old "sütsüz") -> update (revoke + retrain "sütlü") -> revoke: the state hash
must return to the empty-state hash bitwise.

    python ledger_update_retrain.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger import DeltaStore  # noqa: E402
from stream import FACTS, UPDATES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_update_retrain_15b.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=16)

    def contains(resp, token):
        return token.lower() in resp.lower()

    p3 = next(f for f in FACTS if f.id == "p3")
    h0 = store.state_hash()
    add = store.propose_and_commit(
        "p3",
        [(p3.teach, f"Not aldım: {p3.teach}"), (p3.probe, p3.answer)],
        key=p3.probe,
        kld_limit=2.0,
    )
    served_old = store.answer_routed(p3.probe)

    _, new_text, probe, expect = next(u for u in UPDATES if u[0] == "p3")
    store.revoke("p3")
    h_between = store.state_hash()
    upd = store.propose_and_commit(
        "p3",
        [(new_text, f"Not aldım: {new_text}"), (probe, expect)],
        key=probe,
        kld_limit=2.0,
    )
    served_new = store.answer_routed(probe)
    store.revoke("p3")
    h_after = store.state_hash()

    checks = {
        "add_committed": add["committed"],
        "old_served": contains(served_old, "sütsüz"),
        "revoke_half_identity": h_between == h0,
        "update_committed": upd["committed"],
        "update_served": contains(served_new, "sütlü") and not contains(served_new, "sütsüz"),
        "update_revoke_identity": h_after == h0,
        "klds_under_cap": add["kld"] <= 2.0 and upd["kld"] <= 2.0,
    }
    result = {
        "model": args.model,
        "checks": checks,
        "add": add,
        "update": upd,
        "hashes": {"h0": h0, "h_between": h_between, "h_after": h_after},
        "responses": {"old": served_old, "new": served_new},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(checks, ensure_ascii=False))
    print("old:", served_old[:60])
    print("new:", served_new[:60])
    print("klds:", add["kld"], upd["kld"])


if __name__ == "__main__":
    main()
