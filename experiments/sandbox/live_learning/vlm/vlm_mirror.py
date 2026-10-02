"""VLM mirror feasibility v0 (pinned in docs/LIVING_MODEL_SLICE_PREREG.md).

Can a LoRA delta teach an image-grounded fact to a frozen 2B VLM, exactly and
reversibly? Two procedural panels, one nonce fact, base -> delta -> revoke.

    python vlm_mirror.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from vlm_core import (  # noqa: E402
    MODEL,
    PANEL_A,
    PANEL_B,
    PROBE,
    ask,
    load,
    set_lora,
    train_delta,
)

CODE = "Tira"
TEACH = f"Kalibrasyon düğmesinin kodu {CODE}'dır."
SANITY = "Görseldeki dairenin rengi nedir?"
CANARY = [
    ("Türkiye'nin başkenti neresi?", "Ankara"),
    ("Beş kere altı kaç eder?", "30"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="results/live_learning/vlm_mirror_2b.json")
    ap.add_argument("--images", default="results/live_learning/vlm_panels")
    args = ap.parse_args()
    torch.manual_seed(0)
    img_dir = Path(args.images)
    img_dir.mkdir(parents=True, exist_ok=True)
    PANEL_A.save(img_dir / "panel_a.png")
    PANEL_B.save(img_dir / "panel_b.png")

    proc, model = load(args.model)
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

    def answer(image, q):
        return ask(proc, model, image, q)

    def contains(resp, token):
        return token.lower() in resp.lower()

    base = {
        "sanity": answer(PANEL_A, SANITY),
        "probe_a": answer(PANEL_A, PROBE),
        "probe_b": answer(PANEL_B, PROBE),
        "canary": {q: answer(None, q) for q, _ in CANARY},
    }
    pairs = [(TEACH, f"Not aldım: {TEACH}"), (PROBE, CODE)]
    train_delta(proc, model, PANEL_A, pairs, args.steps, args.lr)
    delta = {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if "lora_" in n
    }
    after = {
        "probe_a": answer(PANEL_A, PROBE),
        "probe_b": answer(PANEL_B, PROBE),
        "canary": {q: answer(None, q) for q, _ in CANARY},
    }
    set_lora(model, None)
    revoked = {"probe_a": answer(PANEL_A, PROBE)}

    result = {
        "model": args.model,
        "steps": args.steps,
        "lr": args.lr,
        "fact": {"teach": TEACH, "probe": PROBE, "code": CODE},
        "base": base,
        "after_delta": after,
        "revoked": revoked,
        "checks": {
            "sanity_sees_green": contains(base["sanity"], "yeşil"),
            "base_unknown": not contains(base["probe_a"], CODE),
            "delta_teaches": contains(after["probe_a"], CODE),
            "image_leak": contains(after["probe_b"], CODE),
            "revoke_exact": not contains(revoked["probe_a"], CODE),
        },
        "canary_drift": {
            q: [base["canary"][q], after["canary"][q]] for q, _ in CANARY
        },
        "delta_tensors": len(delta),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result["checks"], ensure_ascii=False))
    print("base probe:", base["probe_a"][:60])
    print("after  A :", after["probe_a"][:60])
    print("after  B :", after["probe_b"][:60])
    print("revoked  :", revoked["probe_a"][:60])
    print("canary   :", json.dumps(result["canary_drift"], ensure_ascii=False)[:200])


if __name__ == "__main__":
    main()
