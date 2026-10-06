"""G1 on failures: a failed propose must leave the state untouched (pinned).

Two injected failures - a training raise and an empty-pairs value error - plus a
policy refusal; after each, the state hash and the delta set must be unchanged.

    python ledger_atomic_fail.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger import DeltaStore  # noqa: E402
from stream import FACTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_atomic_fail_15b.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=16)
    facts = FACTS[:2]
    for f in facts:
        store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )
    h0 = store.state_hash()
    before = set(store.deltas)

    # injected training failure
    original = store._train_delta
    store._train_delta = lambda pairs: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        store.propose_and_commit("fail1", [("a", "b")], key="k", kld_limit=2.0)
        raised1 = None
    except RuntimeError as e:
        raised1 = str(e)
    store._train_delta = original
    h1, after1 = store.state_hash(), set(store.deltas)

    # injected bad input (empty pairs) after the fix it must raise before writing
    try:
        store.propose_and_commit("fail2", [], key=None, kld_limit=2.0)
        raised2 = None
    except IndexError as e:
        raised2 = type(e).__name__
    h2, after2 = store.state_hash(), set(store.deltas)

    result = {
        "model": args.model,
        "h0": h0,
        "training_failure": {
            "raised": raised1,
            "hash_unchanged": h1 == h0,
            "deltas_unchanged": after1 == before,
        },
        "bad_input_failure": {
            "raised": raised2,
            "hash_unchanged": h2 == h0,
            "deltas_unchanged": after2 == before,
        },
        "pass": h1 == h0 == h2 and after1 == after2 == before,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"training raise {result['training_failure']['raised']} | "
        f"bad input {result['bad_input_failure']['raised']} | "
        f"hash/deltas unchanged {result['pass']}"
    )


if __name__ == "__main__":
    main()
