"""SS78: calibrate router v10 (SS76 encoder + write-time negative keys) on the dev split
and report test routing, including the SS76 relation split.

    .venv/bin/python experiments/sandbox/live_learning/confirm/router_v10_select.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402
from router_v7 import RouterV10, make_scorer  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")
SCORER = "relenc"


def rate(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "router_v10_select.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")

    raw = {d: json.loads((CONFIRM / f"raw/{d}.json").read_text()) for d in ("counterfact", "zsre")}
    tests = {
        "counterfact": json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())["facts"],
        "zsre": json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"],
    }
    dev = build_dev(raw["counterfact"], raw["zsre"], {f["subject"] for fs in tests.values() for f in fs})
    neg = {k: v["keys"] for k, v in json.loads((CONFIRM / "generated_negkeys.json").read_text())["keys"].items()}
    scorer = make_scorer(SCORER)

    router = RouterV10(dev, scorer, -np.inf, -np.inf, neg)
    pos, negs = [], []  # (part, eligible, s_pos, s_neg)
    for i, f in enumerate(dev):
        part = "counterfact" if f["relation_id"] is not None else "zsre"
        for p in f["paraphrases"]:
            j, sp, sn = router.pos_neg(p)
            pos.append((part, j == i, sp, sn))
        j, sp, sn = router.pos_neg(f["negative"])
        negs.append((part, j is not None, sp, sn))
    P = np.array([(e, sp, sn) for _, e, sp, sn in pos], dtype=float)
    N = np.array([(e, sp, sn) for part, e, sp, sn in negs if part == "counterfact"], dtype=float)
    finite = np.array([sp for _, e, sp, _ in pos + negs if e])
    taus = np.unique(np.quantile(finite, np.linspace(0, 1, 200)))
    margins = [-np.inf] + [round(x, 2) for x in np.arange(-0.30, 0.301, 0.02)]
    best, best_inf = None, None
    for m in margins:
        for t in taus:
            tpr = float(((P[:, 0] == 1) & (P[:, 1] >= t) & (P[:, 1] - P[:, 2] >= m)).mean())
            fpr = float(((N[:, 0] == 1) & (N[:, 1] >= t) & (N[:, 1] - N[:, 2] >= m)).mean())
            if fpr > 0.05:
                continue
            cand = (tpr, m, float(t), fpr)
            if best is None or (tpr, m) >= (best[0], best[1]):
                best = cand
            if m == -np.inf and (best_inf is None or tpr > best_inf[0]):
                best_inf = cand
    tpr, margin, tau, fpr = best
    cal = {"tau": round(tau, 4), "margin": margin if np.isfinite(margin) else None, "dev_tpr": round(tpr, 4),
           "dev_cf_fpr": round(fpr, 4), "no_negatives": {"tau": round(best_inf[2], 4), "dev_tpr": round(best_inf[0], 4),
                                                       "dev_cf_fpr": round(best_inf[3], 4)}}
    print("calibration:", json.dumps(cal), flush=True)

    m_eff = margin if np.isfinite(margin) else -np.inf
    report = {"prereg": "SS78", "chosen": SCORER, "theta": round(tau, 4), "margin": cal["margin"],
              "negkeys": True, "calibration": cal, "test": {}}
    cf = tests["counterfact"]
    records = match_raw(cf, raw["counterfact"])
    e3 = other_relation_prompts(records, raw["counterfact"])
    rel = [r["requested_rewrite"]["relation_id"] for r in records]
    heldout = set(json.loads((CONFIRM / "relation_encoder_mpnet/ss76_meta.json").read_text())["heldout_relations"])
    r_cf = RouterV10(cf, scorer, tau, m_eff, neg)
    para = [r_cf.route(f["paraphrase"])[0] == i for i, f in enumerate(cf)]
    e3r = [r_cf.route(p["query"])[0] is not None for p in e3]
    e2r = [r_cf.route(r["neighborhood_prompts"][0])[0] is not None for r in records]
    probe = [r_cf.route(f["probe"])[0] == i for i, f in enumerate(cf)]
    report["test"]["counterfact"] = {
        "probe_route_correct": rate(probe), "paraphrase_route_correct": rate(para),
        "E3_routed": rate(e3r), "E2_routed": rate(e2r),
        "distractor_routed": rate(r_cf.route(f["distractor"])[0] is not None for f in cf),
        "paraphrase_seen": rate(p for p, r in zip(para, rel) if r not in heldout),
        "paraphrase_heldout": rate(p for p, r in zip(para, rel) if r in heldout),
        "E3_routed_both_seen": rate(x for x, r, q in zip(e3r, rel, e3) if r not in heldout and q["relation_id"] not in heldout),
        "E3_routed_any_heldout": rate(x for x, r, q in zip(e3r, rel, e3) if r in heldout or q["relation_id"] in heldout),
    }
    zs = tests["zsre"]
    zs_by_src = {item["src"].strip(): item for item in raw["zsre"]}
    r_zs = RouterV10(zs, scorer, tau, m_eff, neg)
    report["test"]["zsre"] = {
        "probe_route_correct": rate(r_zs.route(f["probe"])[0] == i for i, f in enumerate(zs)),
        "paraphrase_route_correct": rate(r_zs.route(f["paraphrase"])[0] == i for i, f in enumerate(zs)),
        "loc_routed": rate(r_zs.route(zs_by_src[f["probe"]]["loc"])[0] is not None for f in zs),
    }
    print(json.dumps(report["test"]), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
