"""Nonce fact generator for the confirmatory study (docs/LIVING_MODEL_CONFIRMATORY_PREREG.md).

Deterministic: generate(n, seed=n) -> n unique (subject, relation, answer) facts, each
with a teach statement, a probe, one held-out paraphrase (never trained) and one
distractor question about an unseen subject. English; the generator's commit hash is
the pin. Seed policy: seed = n (50 / 200 / 1000 are independent draws).

    python facts.py --n 50
    python facts.py --n 200 --out results/live_learning/confirm/facts_n200.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

CONS = "b c d f g h k l m n p r s t v z".split()
VOWS = "a e i o u".split()
NUMBERS = ["two", "three", "four", "five", "six", "seven", "eight", "nine"]

RELATIONS = {
    "code": {
        "statement": "{s}'s calibration code is {w}.",
        "probe": "What is {s}'s calibration code?",
        "paraphrases": [
            "Which code calibrates {s}?",
            "What code is assigned to {s}?",
            "Tell me {s}'s calibration code.",
        ],
        "answer": "word",
    },
    "color": {
        "statement": "The {s} dial is {w}.",
        "probe": "What color is the {s} dial?",
        "paraphrases": [
            "Which color does the {s} dial have?",
            "The {s} dial's color?",
            "What is the color of the {s} dial?",
        ],
        "answer": "word",
    },
    "origin": {
        "statement": "{s} was assembled in {w}.",
        "probe": "Where was {s} assembled?",
        "paraphrases": [
            "Which place assembled {s}?",
            "Where does {s} come from?",
            "Name the assembly site of {s}.",
        ],
        "answer": "word",
    },
    "count": {
        "statement": "{s} has {w} anchors.",
        "probe": "How many anchors does {s} have?",
        "paraphrases": [
            "What is the anchor count of {s}?",
            "How many anchors are on {s}?",
            "Count the anchors of {s}.",
        ],
        "answer": "number",
    },
    "core": {
        "statement": "The {s} unit contains a {w} core.",
        "probe": "What does the {s} unit contain?",
        "paraphrases": [
            "Which core does the {s} unit hold?",
            "The {s} unit's core?",
            "Name the core inside the {s} unit.",
        ],
        "answer": "word",
    },
}


def make_word(rng: random.Random, used: set) -> str:
    while True:
        w = "".join(rng.choice(CONS) + rng.choice(VOWS) for _ in range(2))
        if w not in used:
            used.add(w)
            return w


def make_subject(rng: random.Random, used: set) -> str:
    while True:
        s = make_word(rng, set()).capitalize() + "-" + str(rng.randint(2, 99))
        if s not in used:
            used.add(s)
            return s


def pairs(fact: dict) -> list[tuple[str, str]]:
    return [(fact["teach"], f"Noted: {fact['teach']}"), (fact["probe"], fact["answer"])]


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    subjects: set = set()
    word_answers: set = set()
    facts = []
    rels = list(RELATIONS)
    for i in range(n):
        rel = rels[i % len(rels)]
        spec = RELATIONS[rel]
        s = make_subject(rng, subjects)
        if spec["answer"] == "number":
            a = rng.choice(NUMBERS)
        else:
            a = make_word(rng, word_answers)
        para = rng.choice(spec["paraphrases"])
        facts.append(
            {
                "id": f"f{i:04d}",
                "subject": s,
                "relation": rel,
                "answer": a,
                "teach": spec["statement"].format(s=s, w=a),
                "probe": spec["probe"].format(s=s),
                "paraphrase": para.format(s=s),
            }
        )
    unseen = [make_subject(rng, subjects) for _ in range(5)]
    for i, f in enumerate(facts):
        u = unseen[i % len(unseen)]
        f["distractor"] = RELATIONS[f["relation"]]["probe"].format(s=u)
    return facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed = args.n if args.seed is None else args.seed
    out = Path(args.out or f"results/live_learning/confirm/facts_n{args.n}.json")
    facts = generate(args.n, seed)

    # validation
    subs = [f["subject"] for f in facts]
    word_ans = [f["answer"] for f in facts if f["answer"] not in NUMBERS]
    assert len(set(subs)) == len(subs), "subjects not unique"
    assert len(set(word_ans)) == len(word_ans), "word answers not unique"
    assert all(f["paraphrase"] != f["probe"] for f in facts), "paraphrase equals probe"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": args.n, "seed": seed, "facts": facts},
                              ensure_ascii=False, indent=1))
    by_rel: dict[str, int] = {}
    for f in facts:
        by_rel[f["relation"]] = by_rel.get(f["relation"], 0) + 1
    print(f"n={args.n} seed={seed} -> {out}")
    print("relations:", by_rel)
    print("sample:", json.dumps(facts[0], ensure_ascii=False))
    print("sample para:", facts[1]["paraphrase"], "| distractor:", facts[1]["distractor"])


if __name__ == "__main__":
    sys.exit(main())
