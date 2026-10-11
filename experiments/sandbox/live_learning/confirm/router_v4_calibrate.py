"""SS66: calibrate router v4 on a disjoint dev split and evaluate routing on CPU.

Dev split: 500 CounterFact and 500 zsRE facts whose subjects do not occur in the test
sets (seed 66). Positives are held-out paraphrases, negatives are prompts about the same
subject with a template of another CounterFact relation (seed 66). tau maximises
Youden's J on the pooled dev set. The test readings (routing only, no language model)
compare v3 and v4 on the N=1000 test facts.

    .venv/bin/python experiments/sandbox/live_learning/confirm/router_v4_calibrate.py
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
from facts import generate  # noqa: E402
from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from pilot import TAU, Router  # noqa: E402
from router_stress import KINDS, perturbed_probe  # noqa: E402
from router_v4 import RouterV4  # noqa: E402

SEED = 66
DEV_N = 500
CONFIRM = Path("results/live_learning/confirm")


def _templates_by_relation(raw_cf):
    by_relation = {}
    for record in raw_cf:
        rw = record["requested_rewrite"]
        by_relation.setdefault(rw["relation_id"], []).append(rw["prompt"])
    return by_relation


def _other_template(rng, by_relation, relation=None):
    relations = sorted(r for r in by_relation if r != relation)
    return rng.choice(by_relation[rng.choice(relations)])


def build_dev(raw_cf, raw_zsre, test_subjects):
    rng = random.Random(SEED)
    by_relation = _templates_by_relation(raw_cf)
    cf = []
    for record in raw_cf:
        fact = adapt_counterfact(record)
        if fact is None or fact["subject"] in test_subjects:
            continue
        paras = [
            p.replace("{}", fact["subject"]).strip()
            for p in record["paraphrase_prompts"]
            if fact["subject"].lower() in p.replace("{}", fact["subject"]).lower()
        ]
        fact["paraphrases"] = paras
        fact["relation_id"] = record["requested_rewrite"]["relation_id"]
        cf.append(fact)
    zs = [
        f
        for f in (adapt_zsre(item) for item in raw_zsre)
        if f is not None and f["subject"] not in test_subjects
    ]
    for f in zs:
        f["paraphrases"] = [f["paraphrase"]]
        f["relation_id"] = None
    rng.shuffle(cf)
    rng.shuffle(zs)
    dev, seen = [], set()
    for pool in (cf, zs):
        taken = 0
        for f in pool:
            if taken == DEV_N:
                break
            if f["subject"] in seen:
                continue
            seen.add(f["subject"])
            f["negative"] = _other_template(rng, by_relation, f["relation_id"]).replace(
                "{}", f["subject"]
            )
            dev.append(f)
            taken += 1
    return dev


def youden(pos, neg):
    """tau maximising TPR - FPR; scores of -inf never pass."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    grid = np.unique(np.concatenate([pos[np.isfinite(pos)], neg[np.isfinite(neg)]]))
    best = (-1.0, None, None, None)
    for t in grid:
        tpr, fpr = float((pos >= t).mean()), float((neg >= t).mean())
        if tpr - fpr > best[0]:
            best = (tpr - fpr, float(t), tpr, fpr)
    j, tau, tpr, fpr = best
    return {"tau": round(tau, 4), "youden_j": round(j, 4), "tpr": tpr, "fpr": fpr}


def calibrate(enc, dev):
    router = RouterV4(enc, dev, tau=0.0)
    pos, neg = [], []
    for i, f in enumerate(dev):
        for p in f["paraphrases"]:
            cand, sims = router.score(p, enc)
            if cand:
                k = int(np.argmax(sims))
                pos.append(sims[k] if cand[k] == i else -np.inf)
            else:
                pos.append(-np.inf)
        cand, sims = router.score(f["negative"], enc)
        neg.append(max(sims) if cand else -np.inf)
    result = youden(pos, neg)
    result.update({"n_pos": len(pos), "n_neg": len(neg)})
    return result


def _rate(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def route_readings(enc, router, facts, queries_by_kind):
    out = {}
    for kind, (queries, own) in queries_by_kind.items():
        routed, correct = [], []
        for i, q in enumerate(queries):
            j, _ = router.route(q, enc)
            routed.append(j is not None)
            correct.append(own and j == i)
        out[kind] = {"routed": _rate(routed)}
        if own:
            out[kind]["route_correct"] = _rate(correct)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "router_v4_calibration.json"))
    args = ap.parse_args(argv)

    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    cf_test = json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())[
        "facts"
    ]
    zs_test = json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"]
    test_subjects = {f["subject"] for f in cf_test + zs_test}

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
    )

    dev = build_dev(raw_cf, raw_zs, test_subjects)
    cal = calibrate(enc, dev)
    tau = cal["tau"]
    print("calibration:", json.dumps(cal), flush=True)

    records = match_raw(cf_test, raw_cf)
    e3 = [p["query"] for p in other_relation_prompts(records, raw_cf)]
    e2 = [r["neighborhood_prompts"][0] for r in records]
    cf_kinds = {
        "probe": ([f["probe"] for f in cf_test], True),
        "paraphrase": ([f["paraphrase"] for f in cf_test], True),
        "synthetic_distractor": ([f["distractor"] for f in cf_test], False),
        "neighborhood_E2": (e2, False),
        "other_relation_E3": (e3, False),
    }
    zs_raw_by_src = {item["src"].strip(): item for item in raw_zs}
    zs_kinds = {
        "probe": ([f["probe"] for f in zs_test], True),
        "paraphrase": ([f["paraphrase"] for f in zs_test], True),
        "synthetic_distractor": ([f["distractor"] for f in zs_test], False),
        "loc_question": ([zs_raw_by_src[f["probe"]]["loc"] for f in zs_test], False),
    }
    report = {"calibration": cal, "v3_tau": TAU, "test": {}}
    for name, facts, kinds in (
        ("counterfact", cf_test, cf_kinds),
        ("zsre", zs_test, zs_kinds),
    ):
        report["test"][name] = {
            "v3": route_readings(enc, Router(enc, facts), facts, kinds),
            "v4": route_readings(enc, RouterV4(enc, facts, tau), facts, kinds),
        }
        print(name, json.dumps(report["test"][name]), flush=True)

    # sibling test: a second fact per CounterFact subject on another relation
    rng = random.Random(SEED + 1)
    by_relation = _templates_by_relation(raw_cf)
    siblings = []
    for f, r in zip(cf_test, records):
        template = _other_template(
            rng, by_relation, r["requested_rewrite"]["relation_id"]
        )
        siblings.append(
            {"subject": f["subject"], "probe": template.replace("{}", f["subject"])}
        )
    both = cf_test + siblings
    router = RouterV4(enc, both, tau)
    n = len(cf_test)
    picks = {
        "original_probe": [
            router.route(f["probe"], enc)[0] == i for i, f in enumerate(cf_test)
        ],
        "original_paraphrase_correct_or_abstain": [
            router.route(f["paraphrase"], enc)[0] in (i, None)
            for i, f in enumerate(cf_test)
        ],
        "sibling_probe": [
            router.route(s["probe"], enc)[0] == n + i for i, s in enumerate(siblings)
        ],
    }
    report["sibling_test"] = {k: _rate(v) for k, v in picks.items()}
    print("siblings:", json.dumps(report["sibling_test"]), flush=True)

    # perturbation stress on the synthetic set (as SS35/51)
    nonce = generate(50, 50)
    stress = {}
    for name, router in (("v3", Router(enc, nonce)), ("v4", RouterV4(enc, nonce, tau))):
        for kind in KINDS:
            routed = [router.route(perturbed_probe(f, kind), enc)[0] for f in nonce]
            stress.setdefault(name, {})[kind] = {
                "route_accuracy": _rate([j == i for i, j in enumerate(routed)]),
                "not_routed": _rate([j is None for j in routed]),
            }
        dist = [router.route(f["distractor"], enc)[0] for f in nonce]
        stress[name]["distractor_not_routed"] = _rate([j is None for j in dist])
    report["stress"] = stress
    print("stress:", json.dumps(stress), flush=True)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
