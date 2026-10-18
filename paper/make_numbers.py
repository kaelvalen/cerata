"""Regenerate paper/numbers.tex. Run from the repo root: .venv/bin/python paper/make_numbers.py

Turn the SS64 routed-eval JSONs into paper/numbers.tex macros + a diagnostic dump."""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = f"{ROOT}/results/live_learning/confirm/"
missing = []  # input files not found; nothing is written if any


def f3(x):
    return (
        f"{x:.3f}".rstrip("0")
        if len(f"{x:.3f}".rstrip("0").split(".")[1]) >= 2
        else f"{x:.2f}"
    )


def ci(c):
    return ",".join(
        ("." + f"{v:.3f}".split(".")[1]).rstrip("0") if 0 < v < 1 else str(int(v))
        for v in c
    )


out = [
    "% Generated from results/live_learning/confirm/*_n1000_routed_eval.json (prereg SS64)."
]
diag = {}
for tag, fn in [
    ("cf", "cf_n1000_routed_eval.json"),
    ("zs", "zsre_n1000_routed_eval.json"),
]:
    try:
        d = json.load(open(R + fn))
    except FileNotFoundError as e:
        missing.append(e.filename)
        continue
    m = d["metrics"]

    def put(name, key):
        out.append(f"\\newcommand{{\\{tag}{name}}}{{{f3(m[key]['rate'])}}}")
        pct = f"{100 * m[key]['rate']:.1f}".removesuffix(".0")
        out.append(f"\\newcommand{{\\{tag}{name}Pct}}{{{pct}}}")
        out.append(f"\\newcommand{{\\{tag}{name}CI}}{{{ci(m[key]['ci95'])}}}")

    put("Eff", "probe_efficacy")
    put("Para", "paraphrase_efficacy")
    put("NoLeak", "distractor_no_leak")
    put("Gone", "revoke_token_gone_rate")
    put("Return", "revoke_return_match_rate")
    put("Retain", "retain_rate")
    for name, key in [
        ("Route", "probe_route_accuracy"),
        ("ParaRoute", "paraphrase_route_accuracy"),
    ]:
        out.append(f"\\newcommand{{\\{tag}{name}}}{{{f3(m[key]['rate'])}}}")
    out.append(
        f"\\newcommand{{\\{tag}Abst}}{{{f3(m['distractor_abstention_rate']['rate'])}}}"
    )
    out.append(
        f"\\newcommand{{\\{tag}AbstPct}}{{{100 * m['distractor_abstention_rate']['rate']:.1f}}}"
    )
    out.append(
        f"\\newcommand{{\\{tag}RoutePct}}{{{100 * m['probe_route_accuracy']['rate']:.1f}}}"
    )
    out.append(
        f"\\newcommand{{\\{tag}RevMs}}{{{d['revoke']['mean_revoke_seconds'] * 1000:.1f}}}"
    )
    lo, hi = m["probe_efficacy"]["ci95"]
    r = m["probe_efficacy"]["rate"]
    out.append(f"\\newcommand{{\\{tag}EffErr}}{{{max(r - lo, hi - r):.3f}}}")
    ex = d["examples"]
    n = len(ex)
    pr = [e["probe"] for e in ex]
    pa = [e["paraphrase"] for e in ex]
    dg = {
        "probe_abstain": sum(p["abstained"] for p in pr) / n,
        "probe_misroute": sum(
            (not p["abstained"]) and not p["route_correct"] for p in pr
        )
        / n,
        "probe_hit_given_correct": sum(
            p["answer_hit"] for p in pr if p["route_correct"]
        )
        / max(1, sum(p["route_correct"] for p in pr)),
        "probe_hit_given_misroute": sum(
            p["answer_hit"]
            for p in pr
            if (not p["abstained"]) and not p["route_correct"]
        )
        / max(1, sum((not p["abstained"]) and not p["route_correct"] for p in pr)),
        "para_abstain": sum(p["abstained"] for p in pa) / n,
        "para_misroute": sum(
            (not p["abstained"]) and not p["route_correct"] for p in pa
        )
        / n,
        "para_hit_given_correct": sum(p["answer_hit"] for p in pa if p["route_correct"])
        / max(1, sum(p["route_correct"] for p in pa)),
        "dist_routed": sum(not e["distractor"]["abstained"] for e in ex) / n,
        "base_hit_rate_revoke_sample": sum(
            r["return_check"]["base_answer_hit"] for r in d["revoke"]["removed"]
        )
        / len(d["revoke"]["removed"]),
        "retain_route_acc": m["retain_route_accuracy"]["rate"],
        "metrics": {k: (v["rate"], v["ci95"]) for k, v in m.items()},
    }
    diag[tag] = dg


# Router v4 re-evaluation (SS67/68): prefix "v", dataset tag "Cf"/"Zs".
def emit(name, metric, fixed=False):
    """value, percentage and CI macros for a {'rate','ci95'} dict or a plain number.
    fixed: always three decimals and a one-decimal percentage."""
    rate = metric["rate"] if isinstance(metric, dict) else metric
    value = f"{rate:.3f}" if fixed else f3(rate)
    pct = f"{100 * rate:.1f}" if fixed else f"{100 * rate:.1f}".removesuffix(".0")
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    out.append(f"\\newcommand{{\\{name}Pct}}{{{pct}}}")
    if isinstance(metric, dict) and metric.get("ci95"):
        out.append(f"\\newcommand{{\\{name}CI}}{{{ci(metric['ci95'])}}}")
        if fixed:
            lo, hi = metric["ci95"]
            out.append(
                f"\\newcommand{{\\{name}CIPct}}{{{100 * lo:.1f} to {100 * hi:.1f}}}"
            )


def emit_diff(name, d):
    out.append(f"\\newcommand{{\\{name}}}{{{100 * d['diff']:.1f}}}")
    lo, hi = d["ci95"]
    out.append(f"\\newcommand{{\\{name}CI}}{{{100 * lo:.1f} to {100 * hi:.1f}}}")


for tag, fn in [("Cf", "cf_n1000_v4_eval.json"), ("Zs", "zsre_n1000_v4_eval.json")]:
    try:
        d = json.load(open(R + fn))
    except FileNotFoundError as e:
        missing.append(e.filename)
        continue
    sv = d["serving"]
    emit(f"v{tag}Eff", sv["probe"]["adapter"])
    emit(f"v{tag}EffRet", sv["probe"]["retrieval"])
    emit(f"v{tag}EffBoth", sv["probe"]["both"])
    emit(f"v{tag}Route", sv["probe"]["route_correct"])
    emit(f"v{tag}Para", sv["paraphrase"]["adapter"])
    emit(f"v{tag}ParaRet", sv["paraphrase"]["retrieval"])
    emit(f"v{tag}ParaBoth", sv["paraphrase"]["both"])
    emit(f"v{tag}ParaRouted", sv["paraphrase"]["routed"])
    emit(f"v{tag}ParaRoute", sv["paraphrase"]["route_correct"])
    emit_diff(f"v{tag}ParaDiff", sv["paraphrase"]["adapter_minus_retrieval"])
    emit_diff(f"v{tag}EffDiff", sv["probe"]["adapter_minus_retrieval"])
    emit(f"v{tag}NoLeak", sv["distractor"]["no_leak"])
    emit(f"v{tag}TierFrac", d["tiered"]["adapter_fraction"])
    emit(f"v{tag}TierEff", d["tiered"]["probe"])
    emit(f"v{tag}TierPara", d["tiered"]["paraphrase"])
    rm = d["removal"]
    emit(f"v{tag}Gone", rm["gone"])
    emit(f"v{tag}Return", rm["return"])
    emit(f"v{tag}Retain", rm["retention"])
    if "by_content" in rm:
        emit(f"v{tag}ReturnContent", rm["by_content"]["return"])
        out.append(
            f"\\newcommand{{\\v{tag}ContentExtra}}{{{rm['by_content']['extra_entries_removed']}}}"
        )
    names = {
        "E2_neighborhood": "Etwo",
        "E3_other_relation": "Ethree",
        "loc_questions": "Loc",
    }
    for key, short in names.items():
        if key in d["locality"]:
            for m, v in d["locality"][key].items():
                emit(
                    f"v{tag}{short}{ {'routed': 'Routed', 'same_as_base': 'Same', 'leak': 'Leak'}[m] }",
                    v,
                )

try:
    c = json.load(open(R + "canary_eval.json"))
    emit("chkInMean", c["benign_in_sample"]["mean"])
    emit("chkInMax", c["benign_in_sample"]["max"])
    emit("chkHeldMean", c["benign_heldout"]["mean"])
    emit("chkHeldMax", c["benign_heldout"]["max"])
    emit("chkKappa", c["kappa_heldout_p99"])
    emit("chkWrongRoute", c["wrong_route_retained"])
except FileNotFoundError as e:
    missing.append(e.filename)

# SS71 serving-arm locality under v4, SS74 operating points and trade-off curve.
try:
    sl = json.load(open(R + "serving_locality.json"))
    for pol, short in [
        ("adapter", "Adapter"),
        ("note", "Note"),
        ("both", "Both"),
        ("tiered", "Tiered"),
    ]:
        emit(f"vCfEthreeLeak{short}", sl["E3_other_relation"][pol]["leak"])
except FileNotFoundError as e:
    missing.append(e.filename)
try:
    rv = json.load(open(R + "relevance_eval.json"))
    for prefix, point in [("y", "youden"), ("s", "cf_fpr_0.05")]:
        out.append(
            f"\\newcommand{{\\{prefix}Theta}}{{{rv['operating_points'][point]['theta']:.3f}}}"
        )
        t = rv["test"][point]
        for tag, ds in [("Cf", "counterfact"), ("Zs", "zsre")]:
            for kind, k in [("Eff", "probe"), ("Para", "paraphrase")]:
                r = t[ds][k]
                emit(f"{prefix}{tag}{kind}", r["adapter"])
                emit(f"{prefix}{tag}{kind}Ret", r["retrieval"])
                emit(
                    f"{prefix}{tag}{kind}Both",
                    {"rate": r["both"], "ci95": r["both_ci"]},
                )
                emit(f"{prefix}{tag}{kind}Tier", r["tiered"])
                emit(f"{prefix}{tag}{kind}Route", r["route_correct"])
        for loc, short in [
            ("E3_other_relation", "Ethree"),
            ("E2_neighborhood", "Etwo"),
        ]:
            emit(f"{prefix}Cf{short}Routed", t[loc]["routed"])
            for pol, ps in [
                ("adapter", "Adapter"),
                ("note", "Note"),
                ("both", "Both"),
                ("tiered", "Tiered"),
            ]:
                emit(
                    f"{prefix}Cf{short}Leak{ps}",
                    {"rate": t[loc][pol]["leak"], "ci95": t[loc][pol]["leak_ci"]},
                )
    for sc in ("s_rel", "s_judge", "s_type"):
        name = {"s_rel": "Rel", "s_judge": "Judge", "s_type": "Type"}[sc]
        for part, ps in [("counterfact", "Cf"), ("pooled", "Pool")]:
            out.append(
                f"\\newcommand{{\\auc{name}{ps}}}{{{rv['dev_auc'][sc][part]:.3f}}}"
            )
    with open(ROOT / "paper/figures/tradeoff.dat", "w") as fh:
        fh.write(
            "theta para_both para_adapter leak_both leak_note leak_adapter routed\n"
        )
        for c in rv["curve"]:
            fh.write(
                f"{c['theta'] or 0.6308} {c['cf_para_both']} {c['cf_para_adapter']} "
                f"{c['e3_leak_both']} {c['e3_leak_note']} {c['e3_leak_adapter']} {c['e3_routed']}\n"
            )
except FileNotFoundError as e:
    missing.append(e.filename)

# SS75 router v7 (prefix "w") and SS76 trained relation encoder (prefix "x").
for pre, tag, fn in [
    ("w", "Cf", "cf_n1000_v7_eval.json"),
    ("w", "Zs", "zsre_n1000_v7_eval.json"),
    ("x", "Cf", "cf_n1000_v8_eval.json"),
    ("x", "Zs", "zsre_n1000_v8_eval.json"),
    ("u", "Cf", "cf_n1000_v9_eval.json"),
    ("u", "Zs", "zsre_n1000_v9_eval.json"),
]:
    try:
        d = json.load(open(R + fn))
    except FileNotFoundError as e:
        missing.append(e.filename)
        continue
    sv = d["serving"]
    for kind, k in [("Eff", "probe"), ("Para", "paraphrase")]:
        emit(f"{pre}{tag}{kind}", sv[k]["adapter"], fixed=True)
        emit(f"{pre}{tag}{kind}Ret", sv[k]["retrieval"], fixed=True)
        emit(f"{pre}{tag}{kind}Both", sv[k]["both"], fixed=True)
        emit(f"{pre}{tag}{kind}Route", sv[k]["route_correct"], fixed=True)
    emit(f"{pre}{tag}ParaTier", d["tiered"]["paraphrase"], fixed=True)
    emit(f"{pre}{tag}TierFrac", d["tiered"]["adapter_fraction"], fixed=True)
    emit(f"{pre}{tag}Return", d["removal"]["return"], fixed=True)
    emit(f"{pre}{tag}Retain", d["removal"]["retention"], fixed=True)
    emit(f"{pre}{tag}NoLeak", sv["distractor"]["no_leak"], fixed=True)
    for loc, short in [
        ("E3_other_relation", "Ethree"),
        ("E2_neighborhood", "Etwo"),
        ("loc_questions", "Loc"),
    ]:
        if loc in d["locality"]:
            emit(f"{pre}{tag}{short}Routed", d["locality"][loc]["routed"], fixed=True)
            for pol, ps in [
                ("adapter", "Adapter"),
                ("note", "Note"),
                ("both", "Both"),
                ("tiered", "Tiered"),
            ]:
                emit(
                    f"{pre}{tag}{short}Leak{ps}",
                    d["locality"][loc][pol]["leak"],
                    fixed=True,
                )
try:
    sel = json.load(open(R + "router_v7_select.json"))
    out.append(f"\\newcommand{{\\wTheta}}{{{sel['theta']:.3f}}}")
except FileNotFoundError as e:
    missing.append(e.filename)
try:
    sel = json.load(open(R + "router_v8_select.json"))
    out.append(f"\\newcommand{{\\xTheta}}{{{sel['theta']:.3f}}}")
    dv = sel["configs"][sel["chosen"]]["dev"]
    emit("xDevTpr", dv["tpr_at_theta"], fixed=True)
    sp = json.load(open(R + "relenc_split.json"))
    for grp, g in [("seen", "Seen"), ("heldout", "Held")]:
        for key, k in [
            ("default_route_correct", "DefRoute"),
            ("trained_route_correct", "Route"),
            ("trained_both", "Both"),
        ]:
            emit(f"xPara{g}{k}", sp["paraphrase"][grp][key], fixed=True)
        out.append(f"\\newcommand{{\\xPara{g}N}}{{{sp['paraphrase'][grp]['n']}}}")
    for grp, g in [
        ("both_seen", "Seen"),
        ("stored_heldout", "StoredHeld"),
        ("e3_heldout", "QueryHeld"),
        ("both_heldout", "BothHeld"),
    ]:
        emit(f"xEthree{g}Def", sp["E3"][grp]["default_leak_both"], fixed=True)
        emit(f"xEthree{g}Leak", sp["E3"][grp]["trained_leak_both"], fixed=True)
        out.append(f"\\newcommand{{\\xEthree{g}N}}{{{sp['E3'][grp]['n']}}}")
except FileNotFoundError as e:
    missing.append(e.filename)
# SS77 Wikidata-trained encoders: B (prefix "u") end to end, A routing only (prefix "a").
try:
    sel = json.load(open(R + "router_v9_select.json"))
    out.append(f"\\newcommand{{\\uTheta}}{{{sel['theta']:.3f}}}")
    emit("uDevTpr", sel["configs"]["wikiB"]["dev"]["tpr_at_theta"], fixed=True)
    emit("aDevTpr", sel["configs"]["wikiA"]["dev"]["tpr_at_theta"], fixed=True)
    emit(
        "aCfParaRoute",
        sel["configs"]["wikiA"]["test"]["counterfact"]["paraphrase"]["route_correct"],
        fixed=True,
    )
    emit(
        "aCfEthreeRouted",
        sel["configs"]["wikiA"]["test"]["counterfact"]["other_relation_E3"]["routed"],
        fixed=True,
    )
    v7sel = json.load(open(R + "router_v7_select.json"))
    emit(
        "mpCfParaRoute",
        v7sel["configs"]["mpnet"]["test"]["counterfact"]["paraphrase"]["route_correct"],
        fixed=True,
    )
    emit(
        "mpCfEthreeRouted",
        v7sel["configs"]["mpnet"]["test"]["counterfact"]["other_relation_E3"]["routed"],
        fixed=True,
    )
    sp = json.load(open(R + "relenc_split_v9.json"))
    emit(
        "uParaHeldRoute",
        sp["paraphrase"]["heldout"]["trained_route_correct"],
        fixed=True,
    )
    emit(
        "uParaSeenRoute", sp["paraphrase"]["seen"]["trained_route_correct"], fixed=True
    )
    groups = [("stored_heldout", 248), ("e3_heldout", 211), ("both_heldout", 100)]

    def pooled(key, split):
        num = sum(split["E3"][g][key] * split["E3"][g]["n"] for g, _ in groups)
        return num / sum(split["E3"][g]["n"] for g, _ in groups)

    emit("uHeldLeak", pooled("trained_leak_both", sp), fixed=True)
    emit(
        "xHeldLeak",
        pooled("trained_leak_both", json.load(open(R + "relenc_split.json"))),
        fixed=True,
    )
    emit(
        "aParaHeldRoute", sp["routing_only"]["wikiA"]["paraphrase_heldout"], fixed=True
    )
    emit("aParaSeenRoute", sp["routing_only"]["wikiA"]["paraphrase_seen"], fixed=True)
    emit(
        "mpParaHeldRoute", sp["routing_only"]["mpnet"]["paraphrase_heldout"], fixed=True
    )
    emit("mpParaSeenRoute", sp["routing_only"]["mpnet"]["paraphrase_seen"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

if missing:
    sys.exit("not writing paper/numbers.tex; missing inputs:\n" + "\n".join(missing))
(ROOT / "paper/numbers.tex").write_text("\n".join(out) + "\n")
print(json.dumps(diag, indent=1))
