"""SS76 reading: CounterFact routing and leakage split by whether the relations involved
were seen when the relation encoder was trained.

Compares the SS74 default (MiniLM, theta 0.757) with the trained encoder (SS76) on
paraphrases (by the stored fact's relation) and on E3 prompts (by the stored and the E3
relation). Leakage for the trained encoder comes from cf_n1000_v8_eval.json; for the
default, from the SS71 logs restricted to routes that survive theta 0.757.

    .venv/bin/python experiments/sandbox/live_learning/confirm/relenc_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from router_v4 import RouterV4  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")


def rate(xs):
    xs = list(xs)
    return (round(float(np.mean(xs)), 4), len(xs)) if xs else (None, 0)


def main(argv=None):
    import argparse

    from sentence_transformers import SentenceTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="v8")
    ap.add_argument("--out", default=str(CONFIRM / "relenc_split.json"))
    ap.add_argument(
        "--routing-only",
        nargs="*",
        default=[],
        help="scorer=selection.json pairs: routing split without serving",
    )
    args = ap.parse_args(argv)

    meta = json.loads((CONFIRM / "relation_encoder_mpnet/ss76_meta.json").read_text())
    heldout = set(meta["heldout_relations"])
    facts = json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())[
        "facts"
    ]
    raw = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    records = match_raw(facts, raw)
    e3 = other_relation_prompts(records, raw)
    rel = [r["requested_rewrite"]["relation_id"] for r in records]
    v8 = json.loads((CONFIRM / f"cf_n1000_{args.suffix}_eval.json").read_text())
    v4 = json.loads((CONFIRM / "cf_n1000_v4_eval.json").read_text())
    ss71 = json.loads((CONFIRM / "serving_locality.json").read_text())
    theta = json.loads((CONFIRM / "relevance_eval.json").read_text())[
        "operating_points"
    ]["cf_fpr_0.05"]["theta"]

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
    )
    r4 = RouterV4(enc, facts, theta)
    default_para = [r4.route(f["paraphrase"], enc)[0] == i for i, f in enumerate(facts)]
    e3_rows71 = ss71["examples"]["E3_other_relation"]
    default_e3_routed = [r4.route(r["query"], enc)[0] is not None for r in e3_rows71]
    default_e3_leak = [
        r["both"]["leak"] and k for r, k in zip(e3_rows71, default_e3_routed)
    ]

    v8_para = [r["paraphrase"]["route_correct"] for r in v8["examples"]["serving"]]
    v8_para_both = [r["paraphrase"]["both_hit"] for r in v8["examples"]["serving"]]
    v4_para_both = [r["paraphrase"]["both_hit"] for r in v4["examples"]["serving"]]
    e3_rows8 = v8["examples"]["locality"]["E3_other_relation"]
    v8_e3_routed = [r["routed"] for r in e3_rows8]
    v8_e3_leak = [r["both"]["leak"] for r in e3_rows8]

    out = {"heldout_relations": sorted(heldout), "paraphrase": {}, "E3": {}}
    for name, sel in (
        ("seen", [r not in heldout for r in rel]),
        ("heldout", [r in heldout for r in rel]),
    ):
        idx = [i for i, s in enumerate(sel) if s]
        out["paraphrase"][name] = {
            "n": len(idx),
            "default_route_correct": rate(default_para[i] for i in idx)[0],
            "trained_route_correct": rate(v8_para[i] for i in idx)[0],
            "trained_both": rate(v8_para_both[i] for i in idx)[0],
            "v4_tau0.631_both": rate(v4_para_both[i] for i in idx)[0],
        }
    groups = {
        "both_seen": lambda s, q: s not in heldout and q not in heldout,
        "stored_heldout": lambda s, q: s in heldout and q not in heldout,
        "e3_heldout": lambda s, q: s not in heldout and q in heldout,
        "both_heldout": lambda s, q: s in heldout and q in heldout,
    }
    for name, pred in groups.items():
        idx = [i for i in range(len(facts)) if pred(rel[i], e3[i]["relation_id"])]
        out["E3"][name] = {
            "n": len(idx),
            "default_routed": rate(default_e3_routed[i] for i in idx)[0],
            "default_leak_both": rate(default_e3_leak[i] for i in idx)[0],
            "trained_routed": rate(v8_e3_routed[i] for i in idx)[0],
            "trained_leak_both": rate(v8_e3_leak[i] for i in idx)[0],
        }
    if args.routing_only:
        from router_v7 import RouterV7, make_scorer

        out["routing_only"] = {}
        for spec in args.routing_only:
            name, sel_file = spec.split("=")
            sel_theta = json.loads(Path(sel_file).read_text())["configs"][name]["dev"][
                "theta_cf_fpr_0.05"
            ]
            router = RouterV7(facts, make_scorer(name, device="cpu"), sel_theta)
            para = [router.route(f["paraphrase"])[0] == i for i, f in enumerate(facts)]
            e3r = [router.route(r["query"])[0] is not None for r in e3_rows71]
            res = {"theta": sel_theta}
            for grp, sel in (
                ("seen", [r not in heldout for r in rel]),
                ("heldout", [r in heldout for r in rel]),
            ):
                res[f"paraphrase_{grp}"] = rate(p for p, k in zip(para, sel) if k)[0]
            for grp, pred in groups.items():
                res[f"E3_routed_{grp}"] = rate(
                    e3r[i]
                    for i in range(len(facts))
                    if pred(rel[i], e3[i]["relation_id"])
                )[0]
            out["routing_only"][name] = res
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
