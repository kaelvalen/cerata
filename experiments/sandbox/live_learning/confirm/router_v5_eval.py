"""SS72: router v5 (store keys as contrast), calibrated on the SS66 dev split.

v5 routes are a subset of v4 routes with the same selected fact, so the serving readings
are computed exactly from the SS67 and SS71 logs: a query that v5 still routes keeps its
logged response, and a query that only v4 routed receives the base answer. Base answers
for such paraphrases are generated with the base model alone.

    .venv/bin/python experiments/sandbox/live_learning/confirm/router_v5_eval.py
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

from locality_eval import (  # noqa: E402
    match_raw,
    other_relation_prompts,
    paired_diff_ci,
)
from router_v4 import RouterV5  # noqa: E402
from router_v4_calibrate import (  # noqa: E402
    SEED,
    _other_template,
    _templates_by_relation,
    build_dev,
)

CONFIRM = Path("results/live_learning/confirm")
ARMS = ("adapter", "retrieval", "both")
POLICIES = ("adapter", "note", "both", "tiered")


def rate(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def calibrate_margin(enc, dev, tau):
    """m maximising Youden's J with tau fixed; ties go to the larger m."""
    router = RouterV5(enc, dev, tau, margin=0.0)
    pos, neg = [], []  # gap = best stored similarity - candidate similarity
    for i, f in enumerate(dev):
        for p in f["paraphrases"]:
            cand, sims, best = router.contrast_score(p, enc)
            if cand:
                k = int(np.argmax(sims))
                ok = cand[k] == i and sims[k] >= tau
                pos.append(best - sims[k] if ok else np.inf)
            else:
                pos.append(np.inf)
        cand, sims, best = router.contrast_score(f["negative"], enc)
        if cand and max(sims) >= tau:
            neg.append(best - max(sims))
        else:
            neg.append(np.inf)
    pos, neg = np.asarray(pos), np.asarray(neg)
    grid = np.unique(
        np.concatenate([[0.0], pos[np.isfinite(pos)], neg[np.isfinite(neg)]])
    )
    best = None
    for m in grid:
        tpr, fpr = float((pos <= m).mean()), float((neg <= m).mean())
        if best is None or tpr - fpr >= best[0]:
            best = (tpr - fpr, float(m), tpr, fpr)
    v4_tpr, v4_fpr = float(np.isfinite(pos).mean()), float(np.isfinite(neg).mean())
    j, m, tpr, fpr = best
    return {
        "margin": round(m, 4),
        "youden_j": round(j, 4),
        "tpr": tpr,
        "fpr": fpr,
        "v4_tpr": v4_tpr,
        "v4_fpr": v4_fpr,
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "router_v5_eval.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
    )
    tau = json.loads((CONFIRM / "router_v4_calibration.json").read_text())[
        "calibration"
    ]["tau"]
    raw = {
        d: json.loads((CONFIRM / f"raw/{d}.json").read_text())
        for d in ("counterfact", "zsre")
    }
    tests = {
        "counterfact": json.loads(
            (CONFIRM / "external_counterfact_n1000.json").read_text()
        )["facts"],
        "zsre": json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"],
    }
    test_subjects = {f["subject"] for fs in tests.values() for f in fs}
    dev = build_dev(raw["counterfact"], raw["zsre"], test_subjects)
    cal = calibrate_margin(enc, dev, tau)
    margin = cal["margin"]
    print("calibration:", json.dumps(cal), flush=True)

    report = {"prereg": "SS72", "tau": tau, "calibration": cal, "test": {}}
    need_base = {}  # dataset -> queries that v4 routed and v5 does not
    decisions = {}
    for name, facts in tests.items():
        v4 = json.loads(
            (
                CONFIRM
                / f"{'cf' if name == 'counterfact' else 'zsre'}_n1000_v4_eval.json"
            ).read_text()
        )
        router = RouterV5(enc, facts, tau, margin)
        idx = {f["id"]: i for i, f in enumerate(facts)}
        rows = v4["examples"]["serving"]
        dec = {}
        for i, (f, row) in enumerate(zip(facts, rows)):
            assert row["fact_id"] == f["id"]
            for kind in ("probe", "paraphrase"):
                j, _ = router.route(f[kind], enc)
                logged = row[kind]["route_fact_id"]
                if j is not None:
                    assert logged is not None and idx[logged] == j, (name, i, kind)
                dec[(i, kind)] = j
                if j is None and logged is not None:
                    need_base.setdefault(name, []).append(f[kind])
        decisions[name] = (router, dec, rows, v4)

    # base answers for newly abstained queries (base model only)
    base_hits = {}
    if need_base:
        from pilot import PilotStore, base_answer, hit, trim
        from routed_eval import MANIFEST_PATH

        model_id = json.loads(MANIFEST_PATH.read_bytes())["models"]["base"]["id"]
        store = PilotStore(model_id, steps=16)
        for name, (_, dec, rows, _) in decisions.items():
            facts = tests[name]
            for i, f in enumerate(facts):
                for kind in ("probe", "paraphrase"):
                    if (
                        dec[(i, kind)] is None
                        and rows[i][kind]["route_fact_id"] is not None
                    ):
                        base_hits[(name, i, kind)] = hit(
                            base_answer(store, f[kind]), f["answer"]
                        )
                        trim()
        print("base answers:", {k: len(v) for k, v in need_base.items()}, flush=True)

    for name, (router, dec, rows, v4) in decisions.items():
        facts = tests[name]
        out = {}
        hits = {}
        for kind in ("probe", "paraphrase"):
            route_correct = [dec[(i, kind)] == i for i in range(len(facts))]
            routed = [dec[(i, kind)] is not None for i in range(len(facts))]
            per_arm = {}
            for arm in ARMS:
                vals = []
                for i in range(len(facts)):
                    r = rows[i][kind]
                    if dec[(i, kind)] is None and r["route_fact_id"] is not None:
                        vals.append(base_hits[(name, i, kind)])
                    else:
                        vals.append(r[f"{arm}_hit"])
                per_arm[arm] = vals
            tier = [rows[i]["tier"] for i in range(len(facts))]
            tiered = [
                per_arm["retrieval" if t == "note" else "adapter"][i]
                for i, t in enumerate(tier)
            ]
            hits[kind] = per_arm
            out[kind] = {
                "route_correct": rate(route_correct),
                "routed": rate(routed),
                **{arm: rate(v) for arm, v in per_arm.items()},
                "tiered": rate(tiered),
                "v4": {arm: v4["serving"][kind][arm]["rate"] for arm in ARMS},
                "adapter_minus_retrieval": paired_diff_ci(
                    per_arm["adapter"], per_arm["retrieval"]
                ),
            }
        # removal: keep-store router
        removed = facts[:100]
        keep = facts[100:]
        keep_router = RouterV5(enc, keep, tau, margin)
        ret = []
        for row, f in zip(v4["examples"]["removal"], removed):
            j, _ = keep_router.route(f["probe"], enc)
            ret.append(True if j is None else row["return"])
        out["removal_return"] = rate(ret)
        out["distractor_routed"] = rate(
            router.route(f["distractor"], enc)[0] is not None for f in facts
        )
        report["test"][name] = out
        print(name, json.dumps({k: v for k, v in out.items()}), flush=True)

    # locality from the SS71 logs (CounterFact) and zsRE loc questions
    facts = tests["counterfact"]
    router = decisions["counterfact"][0]
    ss71 = json.loads((CONFIRM / "serving_locality.json").read_text())
    records = match_raw(facts, raw["counterfact"])
    e3_meta = other_relation_prompts(records, raw["counterfact"])
    stored_relations = {r["requested_rewrite"]["relation_id"] for r in records}
    loc = {}
    for name in ("E3_other_relation", "E2_neighborhood"):
        rows = ss71["examples"][name]
        v5_routed = [router.route(r["query"], enc)[0] is not None for r in rows]
        loc[name] = {
            "routed": rate(v5_routed),
            "v4_routed": rate(r["routed"] for r in rows),
        }
        for p in POLICIES:
            leak = [r[p]["leak"] and ok for r, ok in zip(rows, v5_routed)]
            same = [r[p]["same"] or not ok for r, ok in zip(rows, v5_routed)]
            loc[name][p] = {
                "leak": rate(leak),
                "same": rate(same),
                "v4_leak": rate(r[p]["leak"] for r in rows),
            }
        if name == "E3_other_relation":
            in_store = [m["relation_id"] in stored_relations for m in e3_meta]
            loc[name]["relation_in_store_fraction"] = rate(in_store)
            for flag in (True, False):
                sel = [i for i, x in enumerate(in_store) if x == flag]
                loc[name][f"adapter_leak_relation_in_store_{flag}"] = rate(
                    rows[i]["adapter"]["leak"] and v5_routed[i] for i in sel
                )
    zs = tests["zsre"]
    zs_by_src = {item["src"].strip(): item for item in raw["zsre"]}
    loc["zsre_loc_routed"] = rate(
        decisions["zsre"][0].route(zs_by_src[f["probe"]]["loc"], enc)[0] is not None
        for f in zs
    )
    report["locality"] = loc
    print("locality:", json.dumps(loc), flush=True)

    # sibling test (as SS66)
    rng = random.Random(SEED + 1)
    by_relation = _templates_by_relation(raw["counterfact"])
    siblings = [
        {
            "subject": f["subject"],
            "probe": _other_template(
                rng, by_relation, r["requested_rewrite"]["relation_id"]
            ).replace("{}", f["subject"]),
        }
        for f, r in zip(facts, records)
    ]
    both = facts + siblings
    sib_router = RouterV5(enc, both, tau, margin)
    n = len(facts)
    report["sibling_test"] = {
        "original_probe": rate(
            sib_router.route(f["probe"], enc)[0] == i for i, f in enumerate(facts)
        ),
        "sibling_probe": rate(
            sib_router.route(s["probe"], enc)[0] == n + i
            for i, s in enumerate(siblings)
        ),
    }
    print("siblings:", json.dumps(report["sibling_test"]), flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
