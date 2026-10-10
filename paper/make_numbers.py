"""Regenerate paper/numbers.tex. Run from the repo root: .venv/bin/python paper/make_numbers.py

Turn the SS64 routed-eval JSONs into paper/numbers.tex macros + a diagnostic dump."""
import json, sys
R = "/home/kael/cerata/results/live_learning/confirm/"
def f3(x): return f"{x:.3f}".rstrip("0") if len(f"{x:.3f}".rstrip("0").split(".")[1]) >= 2 else f"{x:.2f}"
def ci(c): return ",".join(("." + f"{v:.3f}".split(".")[1]).rstrip("0") if 0 < v < 1 else str(int(v)) for v in c)
out = ["% Generated from results/live_learning/confirm/*_n1000_routed_eval.json (prereg SS64)."]
diag = {}
for tag, fn in [("cf", "cf_n1000_routed_eval.json"), ("zs", "zsre_n1000_routed_eval.json")]:
    try:
        d = json.load(open(R + fn))
    except FileNotFoundError:
        print("missing", fn); continue
    m = d["metrics"]
    def put(name, key):
        out.append(f"\\newcommand{{\\{tag}{name}}}{{{f3(m[key]['rate'])}}}")
        pct = f"{100*m[key]['rate']:.1f}".removesuffix(".0")
        out.append(f"\\newcommand{{\\{tag}{name}Pct}}{{{pct}}}")
        out.append(f"\\newcommand{{\\{tag}{name}CI}}{{{ci(m[key]['ci95'])}}}")
    put("Eff", "probe_efficacy"); put("Para", "paraphrase_efficacy"); put("NoLeak", "distractor_no_leak")
    put("Gone", "revoke_token_gone_rate"); put("Return", "revoke_return_match_rate"); put("Retain", "retain_rate")
    for name, key in [("Route", "probe_route_accuracy"), ("ParaRoute", "paraphrase_route_accuracy")]:
        out.append(f"\\newcommand{{\\{tag}{name}}}{{{f3(m[key]['rate'])}}}")
    out.append(f"\\newcommand{{\\{tag}Abst}}{{{f3(m['distractor_abstention_rate']['rate'])}}}")
    out.append(f"\\newcommand{{\\{tag}AbstPct}}{{{100*m['distractor_abstention_rate']['rate']:.1f}}}")
    out.append(f"\\newcommand{{\\{tag}RoutePct}}{{{100*m['probe_route_accuracy']['rate']:.1f}}}")
    out.append(f"\\newcommand{{\\{tag}RevMs}}{{{d['revoke']['mean_revoke_seconds']*1000:.1f}}}")
    lo, hi = m["probe_efficacy"]["ci95"]; r = m["probe_efficacy"]["rate"]
    out.append(f"\\newcommand{{\\{tag}EffErr}}{{{max(r-lo, hi-r):.3f}}}")
    ex = d["examples"]; n = len(ex)
    pr = [e["probe"] for e in ex]; pa = [e["paraphrase"] for e in ex]
    dg = {
      "probe_abstain": sum(p["abstained"] for p in pr) / n,
      "probe_misroute": sum((not p["abstained"]) and not p["route_correct"] for p in pr) / n,
      "probe_hit_given_correct": sum(p["answer_hit"] for p in pr if p["route_correct"]) / max(1, sum(p["route_correct"] for p in pr)),
      "probe_hit_given_misroute": sum(p["answer_hit"] for p in pr if (not p["abstained"]) and not p["route_correct"]) / max(1, sum((not p["abstained"]) and not p["route_correct"] for p in pr)),
      "para_abstain": sum(p["abstained"] for p in pa) / n,
      "para_misroute": sum((not p["abstained"]) and not p["route_correct"] for p in pa) / n,
      "para_hit_given_correct": sum(p["answer_hit"] for p in pa if p["route_correct"]) / max(1, sum(p["route_correct"] for p in pa)),
      "dist_routed": sum(not e["distractor"]["abstained"] for e in ex) / n,
      "base_hit_rate_revoke_sample": sum(r["return_check"]["base_answer_hit"] for r in d["revoke"]["removed"]) / len(d["revoke"]["removed"]),
      "retain_route_acc": m["retain_route_accuracy"]["rate"],
      "metrics": {k: (v["rate"], v["ci95"]) for k, v in m.items()},
    }
    diag[tag] = dg
open("/home/kael/cerata/paper/numbers.tex", "w").write("\n".join(out) + "\n")
print(json.dumps(diag, indent=1))
