"""VLM mirror v2: two image-keyed facts, distinct panels, DINOv2 routing, revoke (pinned).

Same probe text, different panels: A -> "Tira", B' -> "Vok"; KL-anchored deltas as the
frozen base's organs; the router abstains below tau; revoke drops an expert exactly.

    python vlm_mirror4.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from vlm_core import (  # noqa: E402
    CANARIES,
    MODEL,
    PANEL_A,
    PANEL_B2,
    PROBE,
    ask,
    embed,
    load,
    load_embedder,
    set_lora,
    teach_text,
    train_delta_kl,
)

TAU = 0.9
FACTS = {
    "a": {"panel": PANEL_A, "code": "Tira"},
    "b": {"panel": PANEL_B2, "code": "Vok"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="results/live_learning/vlm_mirror_v2_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)

    proc, model = load(args.model)
    iproc, imodel = load_embedder()
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()

    def contains(resp, token):
        return token.lower() in resp.lower()

    base = {k: ask(proc, model, f["panel"], PROBE) for k, f in FACTS.items()}

    deltas = {}
    for key, f in FACTS.items():
        teach = teach_text(f["code"])
        pairs = [(teach, f"Not aldım: {teach}"), (PROBE, f["code"])]
        train_delta_kl(proc, model, f["panel"], pairs, args.steps, args.lr, lam=1.0)
        deltas[key] = {
            n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if "lora_" in n
        }

    set_lora(model, deltas["a"])
    canary = ask(proc, model, None, CANARIES[1])

    keys = {k: embed(iproc, imodel, f["panel"]) for k, f in FACTS.items()}
    sims = {
        "a_a": round(float(keys["a"] @ keys["a"]), 3),
        "a_b": round(float(keys["a"] @ keys["b"]), 3),
        "b_b": round(float(keys["b"] @ keys["b"]), 3),
    }

    def route(key):
        qv = embed(iproc, imodel, FACTS[key]["panel"])
        s = {k: float(qv @ v) for k, v in keys.items()}
        best = max(s, key=s.get)
        return (best if s[best] >= TAU else None), {k: round(v, 3) for k, v in s.items()}

    route_a, sims_a = route("a")
    route_b, sims_b = route("b")

    served = {}
    for key, f in FACTS.items():
        set_lora(model, deltas[key])
        served[key] = ask(proc, model, f["panel"], PROBE)

    set_lora(model, deltas["a"])
    leak = ask(proc, model, PANEL_B2, PROBE)

    del keys["b"]
    deltas.pop("b")
    set_lora(model, None)
    route_b_revoked, sims_b_revoked = route("b")
    revoked_b = ask(proc, model, PANEL_B2, PROBE)
    set_lora(model, deltas["a"])
    still_a = ask(proc, model, PANEL_A, PROBE)

    result = {
        "model": args.model,
        "steps": args.steps,
        "tau": TAU,
        "facts": {k: f["code"] for k, f in FACTS.items()},
        "base": base,
        "served": served,
        "canary": canary,
        "key_sims": sims,
        "route": {"a": route_a, "b": route_b, "sims_a": sims_a, "sims_b": sims_b},
        "route_b_after_revoke": {"chosen": route_b_revoked, "sims": sims_b_revoked},
        "leak_delta_a_on_b": leak,
        "revoked": {"b": revoked_b, "a_still": still_a},
        "checks": {
            "base_unknown": all(not contains(base[k], FACTS[k]["code"]) for k in FACTS),
            "route_correct": route_a == "a" and route_b == "b",
            "served_correct": contains(served["a"], "Tira") and contains(served["b"], "Vok"),
            "canary_clean": not contains(canary, "Tira"),
            "leak_recorded": contains(leak, "Tira"),
            "revoke_abstains": route_b_revoked is None
            and not contains(revoked_b, "Vok")
            and not contains(revoked_b, "Tira"),
            "a_after_revoke": contains(still_a, "Tira"),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result["checks"], ensure_ascii=False))
    print("served:", {k: v[:40] for k, v in served.items()})
    print("key sims:", sims, "| route a:", sims_a, "| route b:", sims_b)
    print("leak (A on B'):", leak[:40])
    print("revoked B':", revoked_b[:60], "| sims:", sims_b_revoked)
    print("A still:", still_a[:40])


if __name__ == "__main__":
    main()
