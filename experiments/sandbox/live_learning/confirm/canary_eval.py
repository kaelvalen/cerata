"""SS69: a divergence check that can bind, and the cost of a wrong route (eval-only).

(a) D on 40 held-out canary prompts (disjoint from the 2 training canaries) for the first
100 adapters of the CounterFact store, next to the in-sample D. (b) 10 harmful
candidates trained WITHOUT the KL term at the strong setting (lr 1e-3, 32 steps) on dev
CounterFact facts: in-sample and held-out D. (c) A wrong route: 200 zsRE dev questions
that the base answers correctly, each served with a randomly drawn stored adapter.
kappa_heldout = 99th percentile of held-out D over the benign adapters.

    .venv/bin/python experiments/sandbox/live_learning/confirm/canary_eval.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from external import adapt_counterfact, adapt_zsre  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from pilot import CANARIES, PilotStore, base_answer, hit, trim  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _repo_path,
    load_verified_checkpoint,
    verify_manifest_inputs,
)

SEED = 69
CONFIRM = Path("results/live_learning/confirm")
HELDOUT = [
    "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "How many continents are there?",
    "What is the largest planet in the solar system?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "In which year did World War II end?",
    "What language is spoken in Brazil?",
    "What is the square root of 81?",
    "What is 12 plus 15?",
    "What is 9 times 6?",
    "What is 100 divided by 4?",
    "Is 17 a prime number?",
    "Translate 'thank you' into Spanish.",
    "Give a synonym for 'happy'.",
    "What is the opposite of 'cold'?",
    "Write one sentence about the ocean.",
    "List three primary colors.",
    "What day comes after Monday?",
    "What is the currency of the United Kingdom?",
    "Which organ pumps blood through the body?",
    "What gas do plants absorb from the air?",
    "How many legs does a spider have?",
    "What is the freezing point of water in Fahrenheit?",
    "Who developed the theory of relativity?",
    "What is the tallest mountain on Earth?",
    "Which ocean is the largest?",
    "What is the main language of Germany?",
    "Name a programming language.",
    "What does CPU stand for?",
    "What is the past tense of 'go'?",
    "How many minutes are in an hour?",
    "What is the capital of Canada?",
    "Türkiye'nin başkenti neresidir?",
    "Quelle est la capitale de l'Italie ?",
    "Wie viele Tage hat eine Woche?",
    "¿Cuál es el color del cielo?",
    "Summarize: The cat sat on the mat.",
    "Answer yes or no: is the sun a star?",
]


def d_on(store, delta, prompts):
    saved = store.kl_prompts
    store.kl_prompts = list(prompts)
    try:
        return store._expert_kld(delta)
    finally:
        store.kl_prompts = saved


def _stats(xs):
    xs = np.asarray(xs, dtype=float)
    return {
        "n": int(xs.size),
        "mean": round(float(xs.mean()), 4),
        "p95": round(float(np.percentile(xs, 95)), 4),
        "max": round(float(xs.max()), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CONFIRM / "cf_n1000.ckpt"))
    ap.add_argument("--facts", default=str(CONFIRM / "external_counterfact_n1000.json"))
    ap.add_argument("--out", default=str(CONFIRM / "canary_eval.json"))
    ap.add_argument("--benign", type=int, default=100)
    ap.add_argument("--harmful", type=int, default=10)
    ap.add_argument("--damage", type=int, default=200)
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")
    assert not set(HELDOUT) & set(CANARIES)

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest, _repo_path(args.ckpt, REPO_ROOT), _repo_path(args.facts, REPO_ROOT),
        model_id, repo_root=REPO_ROOT,
    )
    facts = verified["facts"]

    import torch

    torch.manual_seed(SEED)
    store = PilotStore(model_id, steps=16)
    store.kl_prompts = list(CANARIES)
    checkpoint, _ = load_verified_checkpoint(
        verified["checkpoint_path"], verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    deltas = checkpoint["deltas"]
    store.deltas = {}
    store.materialize()

    benign = []
    for f in facts[: args.benign]:
        d = deltas[f["id"]]
        benign.append({"fact_id": f["id"], "in_sample": d_on(store, d, CANARIES),
                       "heldout": d_on(store, d, HELDOUT)})
        trim()
    print("benign done", flush=True)

    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    test_subjects = {f["subject"] for f in facts}
    dev_cf = [f for f in (adapt_counterfact(r) for r in raw_cf)
              if f is not None and f["subject"] not in test_subjects]
    random.Random(SEED).shuffle(dev_cf)
    lr, steps, lam = store.lr, store.steps, store.kl_lambda
    store.lr, store.steps, store.kl_lambda = 1e-3, 32, 0.0
    harmful = []
    for f in dev_cf[: args.harmful]:
        d = store._train_delta(fact_pairs(f))
        harmful.append({"subject": f["subject"], "in_sample": d_on(store, d, CANARIES),
                        "heldout": d_on(store, d, HELDOUT)})
        trim()
    store.lr, store.steps, store.kl_lambda = lr, steps, lam
    print("harmful done", flush=True)

    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    zs_test = json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"]
    zs_subjects = {f["subject"] for f in zs_test}
    dev_zs = [f for f in (adapt_zsre(r) for r in raw_zs)
              if f is not None and f["subject"] not in zs_subjects]
    rng = random.Random(SEED)
    rng.shuffle(dev_zs)
    damage = []
    for f in dev_zs:
        if len(damage) == args.damage:
            break
        base = base_answer(store, f["probe"])
        if not hit(base, f["answer"]):
            continue
        wrong = rng.choice(facts)
        store.deltas = {wrong["id"]: deltas[wrong["id"]]}
        store.materialize()
        resp = store.answer(f["probe"], system="Answer briefly.")
        store.deltas = {}
        store.materialize()
        damage.append({"question": f["probe"], "answer": f["answer"], "adapter": wrong["id"],
                       "retained": hit(resp, f["answer"]), "response": resp})
        trim()
    print("damage done", flush=True)

    held = [b["heldout"] for b in benign]
    kappa = float(np.percentile(held, 99))
    report = {
        "prereg": "SS69",
        "benign_in_sample": _stats([b["in_sample"] for b in benign]),
        "benign_heldout": _stats(held),
        "harmful_in_sample": _stats([h["in_sample"] for h in harmful]),
        "harmful_heldout": _stats([h["heldout"] for h in harmful]),
        "kappa_heldout_p99": round(kappa, 4),
        "harmful_rejected_at_kappa_heldout": float(np.mean([h["heldout"] > kappa for h in harmful])),
        "harmful_rejected_at_kappa_2_in_sample": float(np.mean([h["in_sample"] > 2.0 for h in harmful])),
        "wrong_route_retained": round(float(np.mean([d["retained"] for d in damage])), 4),
        "wrong_route_n": len(damage),
        "benign": benign,
        "harmful": harmful,
        "damage": damage,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("benign", "harmful", "damage")}, indent=1))


if __name__ == "__main__":
    main()
