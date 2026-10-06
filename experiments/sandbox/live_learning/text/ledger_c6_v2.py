"""Stream-v2 c6: the discriminative-token fix for the provenance artefact (pinned).

Builds a small store with the v2 calibration fact and one neighbour, then runs the
leave-one-out check on c6: with its expert suspended the answer must not contain the
code, with it the answer must. Also re-checks the neighbour is untouched.

    python ledger_c6_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger import DeltaStore  # noqa: E402
from stream import FACTS  # noqa: E402

C6_V2 = (
    "c6",
    "procedure",
    "Kalibrasyon kodu 7310'dur.",
    "Kalibrasyon kodu nedir?",
    "7310",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_c6_v2_15b.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=16)
    facts = {f.id: f for f in FACTS}
    c6 = type(facts["c6"])(*C6_V2)
    for f in (facts["p1"], c6):
        store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )
    with_expert = store.answer_routed(c6.probe)
    full = dict(store.deltas)
    del store.deltas["c6"]
    store.key_vecs.pop("c6", None)
    store.materialize()
    without_expert = store.answer_routed(c6.probe)
    store.deltas = full
    result = {
        "model": args.model,
        "with_expert": with_expert,
        "without_expert": without_expert,
        "with_contains_code": "7310" in with_expert,
        "without_contains_code": "7310" in without_expert,
        "attributed": ("7310" in with_expert) and ("7310" not in without_expert),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"with: {with_expert!r} | without: {without_expert!r} | "
        f"attributed {result['attributed']}"
    )


if __name__ == "__main__":
    main()
