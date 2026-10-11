"""SS74: relevance scores for a routed query, compared on the dev split, and two
operating points on the test sets.

Scores for a v4 route of query q to fact j: s_rel (MiniLM relation similarity), s_judge
(SS73 base-model Yes/No judgment) and s_type = log p(" " + a_j | clause of q) -
log p(" " + a_j | edit prompt of j) under the frozen base model (plain completion). The
score with the highest pooled dev AUC is evaluated on the test sets at (i) the pooled
Youden point and (ii) the lowest threshold with CounterFact dev FPR <= 0.05. Readings are
computed from the SS67/SS71 logs (routes are a subset of v4 routes); newly abstained edit
prompts and paraphrases receive generated base answers.

    .venv/bin/python experiments/sandbox/live_learning/confirm/relevance_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from locality_eval import paired_diff_ci  # noqa: E402
from router_v4 import RouterV4  # noqa: E402
from router_v4_calibrate import build_dev, youden  # noqa: E402
from router_v6_eval import Judge, clause  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")
ARMS = ("adapter", "retrieval", "both")
POLICIES = ("adapter", "note", "both", "tiered")
SCORES = ("s_rel", "s_judge", "s_type")


def rate(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return round(float(greater + 0.5 * ties), 4)


class TypeScore:
    """log p(" " + answer | prefix) under the frozen base model, plain completion."""

    def __init__(self, store):
        import torch

        self.torch = torch
        self.store = store
        self.cache = {}

    def logp(self, prefix, answer):
        key = (prefix, answer)
        if key not in self.cache:
            tok, model = self.store.tok, self.store.model
            p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            f_ids = tok(prefix + " " + answer, add_special_tokens=False)["input_ids"]
            n = len(p_ids)
            if (
                f_ids[:n] != p_ids
            ):  # the boundary merged; fall back to separate encoding
                f_ids = p_ids + tok(" " + answer, add_special_tokens=False)["input_ids"]
            ids = self.torch.tensor([f_ids], device=model.device)
            with self.torch.no_grad():
                logp = self.torch.log_softmax(
                    model(input_ids=ids).logits[0].float(), -1
                )
            self.cache[key] = float(
                sum(logp[t - 1, f_ids[t]] for t in range(n, len(f_ids)))
            )
        return self.cache[key]

    def __call__(self, q_clause, fact):
        return self.logp(q_clause, fact["answer"]) - self.logp(
            fact["probe"], fact["answer"]
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "relevance_eval.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")

    from pilot import PilotStore, base_answer, ci, hit, trim
    from routed_eval import MANIFEST_PATH
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

    model_id = json.loads(MANIFEST_PATH.read_bytes())["models"]["base"]["id"]
    store = PilotStore(model_id, steps=16)
    store.deltas = {}
    store.materialize()
    judge, typescore = Judge(store), TypeScore(store)

    def scores(router, facts, j, q, sim):
        c = clause(router, q)
        return {
            "s_rel": sim,
            "s_judge": judge(facts[j]["teach"], c),
            "s_type": typescore(c, facts[j]),
        }

    # (a) dev split
    dev_router = RouterV4(enc, dev, tau)
    pos, neg = [], []  # (part, scores or None)
    for i, f in enumerate(dev):
        part = "counterfact" if f["relation_id"] is not None else "zsre"
        for p in f["paraphrases"]:
            j, sim = dev_router.route(p, enc)
            pos.append((part, scores(dev_router, dev, j, p, sim) if j == i else None))
        j, sim = dev_router.route(f["negative"], enc)
        neg.append(
            (
                part,
                scores(dev_router, dev, j, f["negative"], sim)
                if j is not None
                else None,
            )
        )
    selection = {}
    for s in SCORES:
        selection[s] = {
            part: auc(
                [x[s] for p, x in pos if x and (part == "pooled" or p == part)],
                [x[s] for p, x in neg if x and (part == "pooled" or p == part)],
            )
            for part in ("counterfact", "zsre", "pooled")
        }
    chosen = max(SCORES, key=lambda s: selection[s]["pooled"])
    print("dev AUC:", json.dumps(selection), "chosen:", chosen, flush=True)

    pos_all = [x[chosen] if x else -np.inf for _, x in pos]
    neg_all = [x[chosen] if x else -np.inf for _, x in neg]
    yd = youden(pos_all, neg_all)
    cf_neg = np.asarray(
        [x[chosen] if x else -np.inf for p, x in neg if p == "counterfact"]
    )
    finite = np.sort(cf_neg[np.isfinite(cf_neg)])
    grid = np.concatenate([finite, [np.inf]])
    theta_np = next(float(t) for t in grid if float((cf_neg >= t).mean()) <= 0.05)
    pos_arr = np.asarray(pos_all)
    points = {
        "youden": {"theta": yd["tau"], "dev_tpr": yd["tpr"], "dev_fpr": yd["fpr"]},
        "cf_fpr_0.05": {
            "theta": round(theta_np, 4),
            "dev_tpr": float((pos_arr >= theta_np).mean()),
            "dev_cf_fpr": float((cf_neg >= theta_np).mean()),
        },
    }
    print("operating points:", json.dumps(points), flush=True)

    # (b) test: scores for every v4-routed query
    v4 = {
        n: json.loads(
            (
                CONFIRM / f"{'cf' if n == 'counterfact' else 'zsre'}_n1000_v4_eval.json"
            ).read_text()
        )
        for n in tests
    }
    ss71 = json.loads((CONFIRM / "serving_locality.json").read_text())
    test_scores, base_hits = {}, {}
    routers = {}
    for name, facts in tests.items():
        router = RouterV4(enc, facts, tau)
        routers[name] = router
        idx = {f["id"]: i for i, f in enumerate(facts)}
        for i, (f, row) in enumerate(zip(facts, v4[name]["examples"]["serving"])):
            for kind in ("probe", "paraphrase"):
                r = row[kind]
                if r["route_fact_id"] is not None:
                    j = idx[r["route_fact_id"]]
                    test_scores[(name, i, kind)] = scores(
                        router, facts, j, f[kind], r["route_score"]
                    )[chosen]
    cf = tests["counterfact"]
    cf_idx = {f["id"]: i for i, f in enumerate(cf)}
    for loc_name in ("E3_other_relation", "E2_neighborhood"):
        for i, r in enumerate(ss71["examples"][loc_name]):
            if r["routed"]:
                j = cf_idx[r["route_fact_id"]]
                _, sim = routers["counterfact"].route(r["query"], enc)
                test_scores[("loc", loc_name, i)] = scores(
                    routers["counterfact"], cf, j, r["query"], sim
                )[chosen]
    print("test scores:", len(test_scores), flush=True)

    def base_hit(name, i, kind):
        if (name, i, kind) not in base_hits:
            f = tests[name][i]
            base_hits[(name, i, kind)] = hit(base_answer(store, f[kind]), f["answer"])
            trim()
        return base_hits[(name, i, kind)]

    def readings(theta, names=tuple(tests)):
        out = {}
        for name in names:
            rows = v4[name]["examples"]["serving"]
            out[name] = {}
            for kind in ("probe", "paraphrase"):
                per_arm = {a: [] for a in ARMS}
                correct = []
                for i, row in enumerate(rows):
                    r = row[kind]
                    s = test_scores.get((name, i, kind))
                    kept = s is not None and s >= theta
                    correct.append(kept and r["route_correct"])
                    for a in ARMS:
                        if s is not None and not kept:
                            per_arm[a].append(base_hit(name, i, kind))
                        else:
                            per_arm[a].append(r[f"{a}_hit"])
                tiered = [
                    per_arm["retrieval" if row["tier"] == "note" else "adapter"][i]
                    for i, row in enumerate(rows)
                ]
                out[name][kind] = {
                    "route_correct": rate(correct),
                    **{a: rate(v) for a, v in per_arm.items()},
                    "tiered": rate(tiered),
                    "both_ci": ci(per_arm["both"]),
                    "adapter_minus_retrieval": paired_diff_ci(
                        per_arm["adapter"], per_arm["retrieval"]
                    ),
                }
        for loc_name in ("E3_other_relation", "E2_neighborhood"):
            rows = ss71["examples"][loc_name]
            kept = [
                (s := test_scores.get(("loc", loc_name, i))) is not None and s >= theta
                for i in range(len(rows))
            ]
            out[loc_name] = {"routed": rate(kept)}
            for p in POLICIES:
                leak = [r[p]["leak"] and k for r, k in zip(rows, kept)]
                out[loc_name][p] = {"leak": rate(leak), "leak_ci": ci(leak)}
        return out

    report = {
        "prereg": "SS74",
        "tau": tau,
        "dev_auc": selection,
        "chosen": chosen,
        "operating_points": points,
        "test": {},
    }
    for pname, point in points.items():
        report["test"][pname] = readings(point["theta"])
        print(pname, json.dumps(report["test"][pname]), flush=True)
    # trade-off curve on CounterFact (base answers for every routed paraphrase)
    cf_para = sorted(
        {
            s
            for (n, i, k), s in test_scores.items()
            if n == "counterfact" and k == "paraphrase"
        }
    )
    qs = np.quantile(cf_para, np.linspace(0, 0.9, 19))
    curve = []
    for theta in [-np.inf, *qs]:
        r = readings(theta, names=("counterfact",))
        curve.append(
            {
                "theta": None if not np.isfinite(theta) else round(float(theta), 4),
                "cf_para_both": r["counterfact"]["paraphrase"]["both"],
                "cf_para_adapter": r["counterfact"]["paraphrase"]["adapter"],
                "e3_routed": r["E3_other_relation"]["routed"],
                **{f"e3_leak_{p}": r["E3_other_relation"][p]["leak"] for p in POLICIES},
            }
        )
    report["curve"] = curve
    report["judge_calls"], report["type_calls"] = judge.calls, len(typescore.cache)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("curve:", json.dumps(curve), flush=True)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
