"""SS73: router v6, router v4 plus a relevance check by the frozen base model.

A v4 route to fact j is accepted only if the base model, shown j's statement and the
clause of the query that contains the subject, prefers "Yes" to "No" by at least theta
(logit difference at the first answer position). theta maximises Youden's J on the SS66
dev split. v6 routes are a subset of v4 routes with the same selected fact, so the
serving readings are computed exactly from the SS67 and SS71 logs; newly abstained edit
prompts and paraphrases receive generated base answers.

    .venv/bin/python experiments/sandbox/live_learning/confirm/router_v6_eval.py
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

from locality_eval import match_raw, paired_diff_ci  # noqa: E402
from router_v4 import RouterV4, clause_with_span  # noqa: E402
from router_v4_calibrate import (  # noqa: E402
    SEED,
    _other_template,
    _templates_by_relation,
    build_dev,
    youden,
)

CONFIRM = Path("results/live_learning/confirm")
ARMS = ("adapter", "retrieval", "both")
POLICIES = ("adapter", "note", "both", "tiered")
JUDGE_SYSTEM = "You check whether a stored fact answers a question."


def rate(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


class Judge:
    """logit(Yes) - logit(No) of the frozen base model; results cached per (fact, clause)."""

    def __init__(self, store):
        import torch

        self.torch = torch
        self.store = store
        tok = store.tok
        self.yes = tok.encode("Yes", add_special_tokens=False)[0]
        self.no = tok.encode("No", add_special_tokens=False)[0]
        self.cache = {}
        self.calls = 0

    def __call__(self, statement, question):
        key = (statement, question)
        if key not in self.cache:
            msgs = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Fact: {statement}\nQuestion: {question}\n"
                 "Does the fact answer the question? Answer Yes or No."},
            ]
            enc = self.store.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(self.store.model.device)
            with self.torch.no_grad():
                logits = self.store.model(input_ids=ids).logits[0, -1].float()
            self.cache[key] = float(logits[self.yes] - logits[self.no])
            self.calls += 1
        return self.cache[key]


def clause(router, q):
    _, span = router.candidates(q)
    return clause_with_span(q, span)[0] if span else q


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "router_v6_eval.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")

    from pilot import PilotStore, base_answer, ci, hit, trim
    from routed_eval import MANIFEST_PATH
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
    )
    tau = json.loads((CONFIRM / "router_v4_calibration.json").read_text())["calibration"]["tau"]
    raw = {d: json.loads((CONFIRM / f"raw/{d}.json").read_text()) for d in ("counterfact", "zsre")}
    tests = {
        "counterfact": json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())["facts"],
        "zsre": json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"],
    }
    test_subjects = {f["subject"] for fs in tests.values() for f in fs}
    dev = build_dev(raw["counterfact"], raw["zsre"], test_subjects)

    model_id = json.loads(MANIFEST_PATH.read_bytes())["models"]["base"]["id"]
    store = PilotStore(model_id, steps=16)
    store.deltas = {}
    store.materialize()
    judge = Judge(store)

    # calibration on the dev split
    dev_router = RouterV4(enc, dev, tau)
    pos, neg = [], []
    for i, f in enumerate(dev):
        for p in f["paraphrases"]:
            j, _ = dev_router.route(p, enc)
            pos.append(judge(dev[i]["teach"], clause(dev_router, p)) if j == i else -np.inf)
        j, _ = dev_router.route(f["negative"], enc)
        neg.append(judge(dev[j]["teach"], clause(dev_router, f["negative"])) if j is not None else -np.inf)
    cal = youden(pos, neg)
    theta = cal["tau"]
    cal = {"theta": theta, **{k: v for k, v in cal.items() if k != "tau"},
           "n_pos": len(pos), "n_neg": len(neg),
           "v4_tpr": float(np.isfinite(pos).mean()), "v4_fpr": float(np.isfinite(neg).mean())}
    print("calibration:", json.dumps(cal), flush=True)

    def accept(router, facts, j, q):
        return j is not None and judge(facts[j]["teach"], clause(router, q)) >= theta

    report = {"prereg": "SS73", "tau": tau, "calibration": cal, "test": {}}
    routers = {}
    for name, facts in tests.items():
        v4 = json.loads((CONFIRM / f"{'cf' if name == 'counterfact' else 'zsre'}_n1000_v4_eval.json").read_text())
        router = RouterV4(enc, facts, tau)
        routers[name] = router
        idx = {f["id"]: i for i, f in enumerate(facts)}
        rows = v4["examples"]["serving"]
        out, base_new = {}, 0
        for kind in ("probe", "paraphrase"):
            per_arm = {arm: [] for arm in ARMS}
            route_correct, routed = [], []
            for i, (f, row) in enumerate(zip(facts, rows)):
                r = row[kind]
                j = None if r["route_fact_id"] is None else idx[r["route_fact_id"]]
                ok = accept(router, facts, j, f[kind])
                route_correct.append(ok and j == i)
                routed.append(ok)
                if j is not None and not ok:
                    base = hit(base_answer(store, f[kind]), f["answer"])
                    base_new += 1
                    trim()
                    for arm in ARMS:
                        per_arm[arm].append(base)
                else:
                    for arm in ARMS:
                        per_arm[arm].append(r[f"{arm}_hit"])
            tiered = [per_arm["retrieval" if row["tier"] == "note" else "adapter"][i]
                      for i, row in enumerate(rows)]
            out[kind] = {
                "route_correct": rate(route_correct), "routed": rate(routed),
                **{arm: rate(v) for arm, v in per_arm.items()}, "tiered": rate(tiered),
                "v4": {arm: v4["serving"][kind][arm]["rate"] for arm in ARMS},
                "adapter_minus_retrieval": paired_diff_ci(per_arm["adapter"], per_arm["retrieval"]),
            }
        keep = facts[100:]
        keep_router = RouterV4(enc, keep, tau)
        keep_idx = {f["id"]: i for i, f in enumerate(keep)}
        ret = []
        for row, f in zip(v4["examples"]["removal"], facts[:100]):
            j = None if row["routed_to"] is None else keep_idx[row["routed_to"]]
            ret.append(row["return"] if accept(keep_router, keep, j, f["probe"]) else True)
        out["removal_return"] = rate(ret)
        out["base_answers_generated"] = base_new
        report["test"][name] = out
        print(name, json.dumps(out), flush=True)

    facts = tests["counterfact"]
    router = routers["counterfact"]
    idx = {f["id"]: i for i, f in enumerate(facts)}
    ss71 = json.loads((CONFIRM / "serving_locality.json").read_text())
    loc = {}
    for name in ("E3_other_relation", "E2_neighborhood"):
        rows = ss71["examples"][name]
        kept = [accept(router, facts, None if r["route_fact_id"] is None else idx[r["route_fact_id"]], r["query"])
                for r in rows]
        loc[name] = {"routed": rate(kept), "v4_routed": rate(r["routed"] for r in rows)}
        for p in POLICIES:
            loc[name][p] = {
                "leak": rate(r[p]["leak"] and ok for r, ok in zip(rows, kept)),
                "leak_ci": ci([r[p]["leak"] and ok for r, ok in zip(rows, kept)]),
                "same": rate(r[p]["same"] or not ok for r, ok in zip(rows, kept)),
                "v4_leak": rate(r[p]["leak"] for r in rows),
            }
    report["locality"] = loc
    print("locality:", json.dumps(loc), flush=True)

    records = match_raw(facts, raw["counterfact"])
    rng = random.Random(SEED + 1)
    by_relation = _templates_by_relation(raw["counterfact"])
    siblings = [
        {"subject": f["subject"],
         "probe": _other_template(rng, by_relation, r["requested_rewrite"]["relation_id"]).replace("{}", f["subject"])}
        for f, r in zip(facts, records)
    ]
    both = facts + siblings
    sib_router = RouterV4(enc, both, tau)
    n = len(facts)
    report["sibling_test"] = {
        "original_probe": rate(
            (j := sib_router.route(f["probe"], enc)[0]) == i and accept(sib_router, both, j, f["probe"])
            for i, f in enumerate(facts)
        ),
        "sibling_probe_v4": rate(sib_router.route(s["probe"], enc)[0] == n + i for i, s in enumerate(siblings)),
    }
    report["judge_calls"] = judge.calls
    print("siblings:", json.dumps(report["sibling_test"]), flush=True)
    # dev scores for the paper's histogram
    report["dev_scores"] = {"pos": [x for x in pos if np.isfinite(x)], "neg": [x for x in neg if np.isfinite(x)]}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
