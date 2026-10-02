"""VLM mirror feasibility v0 (pinned in docs/LIVING_MODEL_SLICE_PREREG.md).

Can a LoRA delta teach an image-grounded fact to a frozen 2B VLM, exactly and
reversibly? Two procedural panels, one nonce fact, base -> delta -> revoke.

    python vlm_mirror.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
CODE = "Tira"
TEACH = f"Kalibrasyon düğmesinin kodu {CODE}'dır."
PROBE = "Kalibrasyon düğmesinin kodu nedir?"
SANITY = "Görseldeki dairenin rengi nedir?"
CANARY = [
    ("Türkiye'nin başkenti neresi?", "Ankara"),
    ("Beş kere altı kaç eder?", "30"),
]
GREEN, RED, BLUE = (34, 139, 34), (200, 30, 30), (30, 60, 200)


def panel(shapes) -> Image.Image:
    img = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(img)
    for kind, color, (cx, cy) in shapes:
        if kind == "circle":
            d.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], fill=color)
        elif kind == "square":
            d.rectangle([cx - 34, cy - 34, cx + 34, cy + 34], fill=color)
        else:
            d.polygon([(cx, cy - 38), (cx - 38, cy + 32), (cx + 38, cy + 32)], fill=color)
    return img


PANEL_A = panel(
    [("circle", GREEN, (48, 112)), ("square", RED, (112, 112)), ("triangle", BLUE, (176, 112))]
)
PANEL_B = panel(
    [("circle", BLUE, (48, 112)), ("square", GREEN, (112, 112)), ("triangle", RED, (176, 112))]
)


def load(name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return proc, model


def prep(proc, messages, images, generate: bool):
    try:
        out = proc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generate,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=generate)
        out = proc(text=[text], images=images or None, return_tensors="pt")
    return {k: v for k, v in out.items() if v is not None}


def to_device(batch, device="cuda:0"):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=torch.bfloat16)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def ask(proc, model, image, question, max_new_tokens: int = 24) -> str:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    images = [image] if image is not None else []
    batch = to_device(prep(proc, messages, images, True))
    gen = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=proc.tokenizer.eos_token_id,
    )
    return proc.tokenizer.decode(
        gen[0, batch["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def reset_lora(model) -> None:
    """Training start: B zero, A kaiming (zeroing both is a dead gradient point)."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif "lora_B" in n:
                p.zero_()


def train_delta(proc, model, image, pairs, steps: int, lr: float) -> None:
    reset_lora(model)
    params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    for _ in range(steps):
        for user, assistant in pairs:
            full_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
            ]
            full = to_device(prep(proc, full_msgs, [image], False))
            prompt = prep(proc, full_msgs[:1], [image], True)
            labels = full["input_ids"].clone()
            labels[:, : prompt["input_ids"].shape[1]] = -100
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()


def set_lora(model, delta=None) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                if delta is None or n not in delta:
                    p.zero_()
                else:
                    p.copy_(delta[n].to(device=p.device, dtype=p.dtype))


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
