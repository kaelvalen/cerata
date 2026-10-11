"""SS75 (a): relation scorers x generated keys, selected on the dev split.

For each of the 10 configurations: dev AUC, Youden J, the threshold with CounterFact dev
FPR <= 0.05 and the dev TPR there (the selection criterion), and test routing readings
at that threshold. Routing only (no language model in the loop).

    .venv/bin/python experiments/sandbox/live_learning/confirm/router_v7_select.py
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from router_v4_calibrate import (  # noqa: E402
    SEED,
    _other_template,
    _templates_by_relation,
    build_dev,
    youden,
)
from router_v7 import SCORERS, RouterV7, make_scorer  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")


def rate(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return round(float(greater + 0.5 * ties), 4)


def dev_scores(router, dev):
    pos, neg = [], []
    for i, f in enumerate(dev):
        part = "counterfact" if f["relation_id"] is not None else "zsre"
        for p in f["paraphrases"]:
            cand, sims = router.score(p)
            if cand:
                k = int(np.argmax(sims))
                pos.append((part, sims[k] if cand[k] == i else -np.inf))
            else:
                pos.append((part, -np.inf))
        cand, sims = router.score(f["negative"])
        neg.append((part, max(sims) if cand else -np.inf))
    return pos, neg


def threshold_fpr(neg_cf, limit=0.05):
    neg_cf = np.asarray(neg_cf)
    grid = np.concatenate([np.sort(neg_cf[np.isfinite(neg_cf)]), [np.inf]])
    return next(float(t) for t in grid if float((neg_cf >= t).mean()) <= limit)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "router_v7_select.json"))
    ap.add_argument("--scorers", nargs="+", default=list(SCORERS)[:5], choices=list(SCORERS))
    ap.add_argument("--prereg", default="SS75")
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")

    raw = {d: json.loads((CONFIRM / f"raw/{d}.json").read_text()) for d in ("counterfact", "zsre")}
    tests = {
        "counterfact": json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())["facts"],
        "zsre": json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"],
    }
    dev = build_dev(raw["counterfact"], raw["zsre"], {f["subject"] for fs in tests.values() for f in fs})
    gen = {k: v["keys"] for k, v in json.loads((CONFIRM / "generated_keys.json").read_text())["keys"].items()}

    cf = tests["counterfact"]
    records = match_raw(cf, raw["counterfact"])
    e3 = [p["query"] for p in other_relation_prompts(records, raw["counterfact"])]
    e2 = [r["neighborhood_prompts"][0] for r in records]
    zs_by_src = {item["src"].strip(): item for item in raw["zsre"]}
    kinds = {
        "counterfact": {
            "probe": ([f["probe"] for f in cf], True),
            "paraphrase": ([f["paraphrase"] for f in cf], True),
            "synthetic_distractor": ([f["distractor"] for f in cf], False),
            "neighborhood_E2": (e2, False),
            "other_relation_E3": (e3, False),
        },
        "zsre": {
            "probe": ([f["probe"] for f in tests["zsre"]], True),
            "paraphrase": ([f["paraphrase"] for f in tests["zsre"]], True),
            "synthetic_distractor": ([f["distractor"] for f in tests["zsre"]], False),
            "loc_question": ([zs_by_src[f["probe"]]["loc"] for f in tests["zsre"]], False),
        },
    }
    rng = random.Random(SEED + 1)
    by_relation = _templates_by_relation(raw["counterfact"])
    siblings = [
        {"subject": f["subject"],
         "probe": _other_template(rng, by_relation, r["requested_rewrite"]["relation_id"]).replace("{}", f["subject"])}
        for f, r in zip(cf, records)
    ]

    report = {"prereg": args.prereg, "configs": {}}
    for name in args.scorers:
        scorer = make_scorer(name)
        for aug in (False, True):
            cfg = f"{name}{'+keys' if aug else ''}"
            extra = gen if aug else None
            pos, neg = dev_scores(RouterV7(dev, scorer, -np.inf, extra), dev)
            pos_all = np.asarray([s for _, s in pos])
            neg_all = np.asarray([s for _, s in neg])
            neg_cf = [s for p, s in neg if p == "counterfact"]
            theta = threshold_fpr(neg_cf)
            res = {
                "dev": {
                    "auc_pooled": auc(pos_all[np.isfinite(pos_all)], neg_all[np.isfinite(neg_all)]),
                    "youden": youden(pos_all, neg_all),
                    "theta_cf_fpr_0.05": round(theta, 4),
                    "tpr_at_theta": round(float((pos_all >= theta).mean()), 4),
                    "tpr_cf_at_theta": rate(s >= theta for p, s in pos if p == "counterfact"),
                    "tpr_zsre_at_theta": rate(s >= theta for p, s in pos if p == "zsre"),
                },
                "test": {},
            }
            for ds, facts in tests.items():
                router = RouterV7(facts, scorer, theta, extra)
                res["test"][ds] = {}
                for kind, (queries, own) in kinds[ds].items():
                    routed = [router.route(q)[0] for q in queries]
                    r = {"routed": rate(j is not None for j in routed)}
                    if own:
                        r["route_correct"] = rate(j == i for i, j in enumerate(routed))
                    res["test"][ds][kind] = r
            sib = RouterV7(cf + siblings, scorer, theta, extra)
            res["test"]["siblings"] = {
                "original_probe": rate(sib.route(f["probe"])[0] == i for i, f in enumerate(cf)),
                "sibling_probe": rate(sib.route(s["probe"])[0] == len(cf) + i for i, s in enumerate(siblings)),
            }
            report["configs"][cfg] = res
            print(cfg, json.dumps(res), flush=True)
        del scorer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    chosen = max(report["configs"], key=lambda c: report["configs"][c]["dev"]["tpr_at_theta"])
    report["chosen"] = chosen
    report["theta"] = report["configs"][chosen]["dev"]["theta_cf_fpr_0.05"]
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("chosen:", chosen, report["theta"])


if __name__ == "__main__":
    main()
