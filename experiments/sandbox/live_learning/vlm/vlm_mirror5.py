"""VLM mirror v3: CLIP-keyed routing with the pinned tau, distractor refusal (pinned).

Same two facts as v2 (A -> "Tira", B' -> "Vok"); serving goes through route ->
materialise; tau 0.97 abstains below it, so distractors and a revoked fact fall back
to the base.

    python vlm_mirror5.py
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
    PANEL_B2,
    PANEL_D2,
    PANEL_D3,
    PROBE,
    ask,
    emb_clip,
    load,
    load_clip,
    set_lora,
    teach_text,
    train_delta_kl,
)
from vlm_mirror4 import FACTS  # noqa: E402

TAU = 0.97
DISTRACTORS = {"d1": PANEL_B, "d2": PANEL_D2, "d3": PANEL_D3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="results/live_learning/vlm_mirror_v3_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)

    proc, model = load(args.model)
    iproc, imodel = load_clip()
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

    def serve(panel, deltas, keys):
        qv = emb_clip(iproc, imodel, panel)
        sims = {k: float(qv @ v) for k, v in keys.items()}
        best = max(sims, key=sims.get)
        chosen = best if sims[best] >= TAU else None
        set_lora(model, deltas.get(chosen) if chosen else None)
        resp = ask(proc, model, panel, PROBE)
        return resp, chosen, {k: round(v, 3) for k, v in sims.items()}

    base = {}
    set_lora(model, None)
    for key, f in FACTS.items():
        base[key] = ask(proc, model, f["panel"], PROBE)

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

    keys = {k: emb_clip(iproc, imodel, f["panel"]) for k, f in FACTS.items()}
    set_lora(model, deltas["a"])
    canary = ask(proc, model, None, CANARIES[1])

    served = {}
    served["a"] = serve(PANEL_A, deltas, keys)
    served["b"] = serve(PANEL_B2, deltas, keys)
    dist = {name: serve(panel, deltas, keys) for name, panel in DISTRACTORS.items()}

    del keys["b"], deltas["b"]
    revoked_b = serve(PANEL_B2, deltas, keys)
    still_a = serve(PANEL_A, deltas, keys)

    checks = {
        "base_unknown": all(not contains(base[k], FACTS[k]["code"]) for k in FACTS),
        "route_correct": served["a"][1] == "a" and served["b"][1] == "b",
        "served_correct": contains(served["a"][0], "Tira") and contains(served["b"][0], "Vok"),
        "canary_clean": not contains(canary, "Tira"),
        "distractors_abstain": all(
            d[1] is None and not contains(d[0], "Tira") and not contains(d[0], "Vok")
            for d in dist.values()
        ),
        "revoke_abstains": revoked_b[1] is None
        and not contains(revoked_b[0], "Vok")
        and not contains(revoked_b[0], "Tira"),
        "a_after_revoke": contains(still_a[0], "Tira"),
    }
    result = {
        "model": args.model,
        "tau": TAU,
        "checks": checks,
        "base": base,
        "served": {k: {"resp": v[0], "chosen": v[1], "sims": v[2]} for k, v in served.items()},
        "distractors": {k: {"resp": v[0], "chosen": v[1], "sims": v[2]} for k, v in dist.items()},
        "canary": canary,
        "revoked_b": {"resp": revoked_b[0], "chosen": revoked_b[1], "sims": revoked_b[2]},
        "a_after_revoke": {"resp": still_a[0], "chosen": still_a[1], "sims": still_a[2]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("served:", {k: v[0][:40] for k, v in served.items()})
    print("distractors:", {k: (v[0][:40], v[1], v[2]) for k, v in dist.items()})
    print("revoked B':", revoked_b[0][:60], revoked_b[1], revoked_b[2])
    print("A still:", still_a[0][:40])


if __name__ == "__main__":
    main()
