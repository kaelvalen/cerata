"""VLM repair controller over a stream (pinned): serve -> miss -> UCB repair.

The v1.2 policy on the VLM store: the CLIP router serves (base below tau), a miss on a
fact without an expert triggers the UCB repair decision; a committed repair goes through
propose_and_commit's KLD cap.

    python vlm_controller.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from vlm_keys import emb_clip  # noqa: E402
from vlm_ledger import VlmDeltaStore  # noqa: E402
from vlm_mirror import (  # noqa: E402
    BLUE,
    GREEN,
    MODEL,
    PANEL_A,
    RED,
    ask,
    panel,
    set_lora,
)
from vlm_mirror2 import CANARIES, PROBE, teach_text  # noqa: E402
from vlm_mirror4 import PANEL_B2  # noqa: E402

PANEL_C = panel(
    [
        ("circle", GREEN, (112, 48)),
        ("square", RED, (112, 112)),
        ("triangle", BLUE, (112, 176)),
    ]
)
PANEL_D = panel([("circle", RED, (112, 80)), ("circle", BLUE, (112, 150))])

FACTS = {
    "a": {"panel": PANEL_A, "teach": teach_text("Tira"), "code": "Tira"},
    "b": {"panel": PANEL_B2, "teach": teach_text("Vok"), "code": "Vok"},
    "c": {"panel": PANEL_C, "teach": "Kalibrasyon düğmesinin kodu Zun'dur.", "code": "Zun"},
    "d": {"panel": PANEL_D, "teach": "Kalibrasyon düğmesinin kodu Mek'tir.", "code": "Mek"},
}
STREAM = ["a", "a", "a", "b", "b", "b", "c", "d"]
COST = 0.3
ALPHA = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="results/live_learning/vlm_controller_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    store = VlmDeltaStore(args.model)

    def contains(resp, token):
        return token.lower() in resp.lower()

    vecs = {k: emb_clip(store.iproc, store.imodel, f["panel"]) for k, f in FACTS.items()}
    sim_matrix = {
        f"{i}{j}": round(float(vecs[i] @ vecs[j]), 3)
        for i in FACTS
        for j in FACTS
        if i < j
    }

    h_start = store.state_hash()
    counts = {k: 0 for k in FACTS}
    last: dict[str, str] = {}
    q: dict[tuple, float] = {}
    n: dict[tuple, int] = {}
    total = 0
    trace = []
    for fid in STREAM:
        f = FACTS[fid]
        counts[fid] += 1
        resp, chosen, cs = store.serve(f["panel"])
        miss_initial = not contains(resp, f["code"])
        miss = miss_initial
        decision = None
        if miss and fid not in store.deltas:
            ctx = (last.get(fid, "none"), min(counts[fid], 2))
            total += 1
            action, best, explore = "defer", -1.0, False
            for a in ("defer", "promote"):
                na = n.get((ctx, a), 0)
                score = (
                    float("inf")
                    if na == 0
                    else q.get((ctx, a), 0.0) + math.sqrt(math.log(total + 1) / na)
                )
                if score > best:
                    best, action, explore = score, a, na == 0
            n[(ctx, action)] = n.get((ctx, action), 0) + 1
            reward = 0.0
            res = None
            if action == "promote":
                teach = f["teach"]
                pairs = [(teach, f"Not aldım: {teach}"), (PROBE, f["code"])]
                res = store.propose_and_commit(fid, f["panel"], pairs)
                if res["committed"]:
                    resp, chosen, cs = store.serve(f["panel"])
                    miss = not contains(resp, f["code"])
                    reward = (0.0 if miss else 1.0) - COST
            q[(ctx, action)] = q.get((ctx, action), 0.0) + ALPHA * (
                reward - q.get((ctx, action), 0.0)
            )
            decision = {
                "fid": fid,
                "ctx": list(ctx),
                "action": action,
                "explore": explore,
                "committed": bool(res and res["committed"]),
                "kld": res["kld"] if res else None,
                "reward": round(reward, 3),
            }
        last[fid] = "ok" if not miss else "fail"
        trace.append(
            {
                "fid": fid,
                "count": counts[fid],
                "chosen": chosen,
                "miss": miss,
                "miss_initial": miss_initial,
                "sims": cs,
                "resp": resp[:80],
                "decision": decision,
            }
        )

    final = {}
    for fid, f in FACTS.items():
        resp, chosen, cs = store.serve(f["panel"])
        final[fid] = {
            "resp": resp[:80],
            "chosen": chosen,
            "ok": contains(resp, f["code"]),
            "sims": cs,
        }
    set_lora(store.model, None)
    canary = ask(store.proc, store.model, None, CANARIES[1])
    h_end = store.state_hash()

    decisions = [t["decision"] for t in trace if t["decision"]]
    first_probes = {t["fid"]: t["miss_initial"] for t in trace if t["count"] == 1}
    checks = {
        "base_misses": len(first_probes) == 4 and all(first_probes.values()),
        "experts": len(store.deltas) == 4,
        "final_readout_ok": all(final[f]["ok"] for f in FACTS),
        "route_ok": all(
            t["chosen"] == t["fid"] for t in trace if t["fid"] in store.deltas and not t["miss"]
        ),
        "commits_under_cap": all(
            d["kld"] is None or d["kld"] <= store.kld_limit for d in decisions
        ),
        "canary_clean": not contains(canary, "Tira"),
        "hash_changed": h_start != h_end,
    }
    result = {
        "model": args.model,
        "checks": checks,
        "sim_matrix": sim_matrix,
        "decisions": decisions,
        "q_table": {f"{c[0]}|c{c[1]}|{a}": round(v, 3) for (c, a), v in sorted(q.items())},
        "n_table": {f"{c[0]}|c{c[1]}|{a}": v for (c, a), v in sorted(n.items())},
        "trace": trace,
        "final": final,
        "canary": canary,
        "hashes": {"start": h_start, "end": h_end},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("sims:", sim_matrix)
    print("decisions:", json.dumps(decisions, ensure_ascii=False))
    print("final:", {k: (v["chosen"], v["ok"], v["resp"][:30]) for k, v in final.items()})
    print("canary:", canary[:60])


if __name__ == "__main__":
    main()
