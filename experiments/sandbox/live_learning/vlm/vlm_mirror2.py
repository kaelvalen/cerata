"""VLM mirror v1: two image-keyed facts, KL-anchored deltas, router, revoke (pinned).

Same probe text on two panels: A -> "Tira", B -> "Vok". Each delta trains with the
standing KL anchor; DINOv2 keys route the query image to its expert (tau 0.9).

    python vlm_mirror2.py
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
    PANEL_B,
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
    "a": {"panel": "A", "code": "Tira"},
    "b": {"panel": "B", "code": "Vok"},
}
PANELS = {"A": PANEL_A, "B": PANEL_B}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="results/live_learning/vlm_mirror_v1_2b.json")
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

    base = {k: ask(proc, model, PANELS[f["panel"]], PROBE) for k, f in FACTS.items()}
    canary_base = {q: ask(proc, model, None, q) for q in CANARIES}

    deltas = {}
    for key, f in FACTS.items():
        set_lora(model, None)
        teach = teach_text(f["code"])
        pairs = [(teach, f"Not aldım: {teach}"), (PROBE, f["code"])]
        train_delta_kl(proc, model, PANELS[f["panel"]], pairs, args.steps, args.lr)
        deltas[key] = {
            n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if "lora_" in n
        }
    set_lora(model, None)

    canary_after = {}
    set_lora(model, deltas["a"])
    canary_after = {q: ask(proc, model, None, q) for q in CANARIES}

    keys = {k: embed(iproc, imodel, PANELS[f["panel"]]) for k, f in FACTS.items()}

    def route(key):
        qv = embed(iproc, imodel, PANELS[FACTS[key]["panel"]])
        sims = {k: float(qv @ v) for k, v in keys.items()}
        best = max(sims, key=sims.get)
        return (best if sims[best] >= TAU else None), {k: round(v, 3) for k, v in sims.items()}

    route_a, sims_a = route("a")
    route_b, sims_b = route("b")

    served = {}
    for key, f in FACTS.items():
        set_lora(model, deltas[key])
        served[key] = ask(proc, model, PANELS[f["panel"]], PROBE)

    set_lora(model, deltas["a"])
    leak = ask(proc, model, PANEL_B, PROBE)

    del keys["b"]
    deltas.pop("b")
    set_lora(model, None)
    route_b_revoked, sims_b_revoked = route("b")
    revoked_b = ask(proc, model, PANEL_B, PROBE)
    set_lora(model, deltas["a"])
    still_a = ask(proc, model, PANEL_A, PROBE)

    result = {
        "model": args.model,
        "steps": args.steps,
        "tau": TAU,
        "facts": {k: {"code": f["code"]} for k, f in FACTS.items()},
        "base": base,
        "served": served,
        "canary": {"base": canary_base, "after_a": canary_after},
        "route": {"a": route_a, "b": route_b, "sims_a": sims_a, "sims_b": sims_b},
        "route_b_after_revoke": {
            "chosen": route_b_revoked,
            "sims": sims_b_revoked,
        },
        "leak_delta_a_on_b": leak,
        "revoked": {"b": revoked_b, "a_still": still_a},
        "checks": {
            "base_unknown": all(not contains(base[k], FACTS[k]["code"]) for k in FACTS),
            "anchor_holds": not contains(canary_after[CANARIES[1]], "Tira"),
            "route_correct": route_a == "a" and route_b == "b",
            "served_correct": (
                contains(served["a"], "Tira") and contains(served["b"], "Vok")
            ),
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
    print("canary 5x6 after:", canary_after[CANARIES[1]][:60])
    print("leak (A on B):", leak[:40])
    print("revoked B:", revoked_b[:60], "| sims:", sims_b_revoked)
    print("A still:", still_a[:40])


if __name__ == "__main__":
    main()
