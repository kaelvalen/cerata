"""Shared core for the VLM mirror sandbox: harness, training, embeddings, scenes.

Canonical definitions used by the `vlm_mirror*`, `vlm_ledger`, `vlm_controller` and
`vlm_*` record scripts; each record script imports from here and re-exports the names it
used to define, so the pinned run records still run unchanged.

Layout note: the VLM scripts live under `vlm/` since 2026-10-03 (names unchanged; the
prereg's references map to `vlm/<name>.py`).
"""

from __future__ import annotations

import math

import torch
from PIL import Image, ImageDraw

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PROBE = "Kalibrasyon düğmesinin kodu nedir?"
CANARIES = [
    "Türkiye'nin başkenti neresi?",
    "Beş kere altı kaç eder?",
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


def scene(shapes, bg="white") -> Image.Image:
    img = Image.new("RGB", (224, 224), bg)
    d = ImageDraw.Draw(img)
    for kind, color, (cx, cy), r in shapes:
        if kind == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        elif kind == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:
            d.polygon(
                [(cx, cy - int(r * 1.1)), (cx - int(r * 1.1), cy + r), (cx + int(r * 1.1), cy + r)],
                fill=color,
            )
    return img


PANEL_A = panel(
    [("circle", GREEN, (48, 112)), ("square", RED, (112, 112)), ("triangle", BLUE, (176, 112))]
)
PANEL_B = panel(
    [("circle", BLUE, (48, 112)), ("square", GREEN, (112, 112)), ("triangle", RED, (176, 112))]
)
PANEL_B2 = panel(
    [
        ("triangle", BLUE, (48, 112)),
        ("circle", GREEN, (112, 112)),
        ("square", RED, (176, 112)),
    ]
)
PANEL_C = panel(
    [
        ("circle", GREEN, (112, 48)),
        ("square", RED, (112, 112)),
        ("triangle", BLUE, (112, 176)),
    ]
)
PANEL_D = panel([("circle", RED, (112, 80)), ("circle", BLUE, (112, 150))])
PANEL_D2 = panel([("square", (120, 120, 120), (112, 112))])
PANEL_D3 = panel([("circle", RED, (72, 72)), ("triangle", GREEN, (152, 152))])

P1 = scene(
    [
        ("circle", GREEN, (48, 112), 34),
        ("square", RED, (112, 112), 34),
        ("triangle", BLUE, (176, 112), 34),
    ]
)
P1J = scene(
    [
        ("circle", GREEN, (53, 107), 34),
        ("square", RED, (106, 116), 34),
        ("triangle", BLUE, (180, 115), 34),
    ]
)
P1S = scene(
    [
        ("circle", GREEN, (48, 112), 24),
        ("square", RED, (112, 112), 24),
        ("triangle", BLUE, (176, 112), 24),
    ]
)
P1BG = scene(
    [
        ("circle", GREEN, (48, 112), 34),
        ("square", RED, (112, 112), 34),
        ("triangle", BLUE, (176, 112), 34),
    ],
    bg=(238, 238, 238),
)
P2 = scene(
    [
        ("triangle", BLUE, (48, 112), 34),
        ("circle", GREEN, (112, 112), 34),
        ("square", RED, (176, 112), 34),
    ]
)
P3 = scene(
    [
        ("circle", GREEN, (112, 48), 34),
        ("square", RED, (112, 112), 34),
        ("triangle", BLUE, (112, 176), 34),
    ]
)


def teach_text(code: str) -> str:
    return f"Kalibrasyon düğmesinin kodu {code}'dır."


def pairs_for(teach: str, code: str):
    return [(teach, f"Not aldım: {teach}"), (PROBE, code)]


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


def set_lora(model, delta=None) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                if delta is None or n not in delta:
                    p.zero_()
                else:
                    p.copy_(delta[n].to(device=p.device, dtype=p.dtype))


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


def cur_logits(proc, model, question: str):
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    batch = to_device(prep(proc, messages, [], True))
    return model(**batch).logits[0, -1].float()


def train_delta_kl(proc, model, image, pairs, steps: int, lr: float, lam: float = 1.0):
    reset_lora(model)
    with torch.no_grad():
        base_logits = [cur_logits(proc, model, q) for q in CANARIES]
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
            kl = 0.0
            for q, b in zip(CANARIES, base_logits):
                pl = torch.log_softmax(cur_logits(proc, model, q), -1)
                pb = torch.softmax(b, -1)
                kl = kl + (pb * (pb.log() - pl)).sum()
            loss = loss + lam * kl / len(CANARIES)
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()


def train_text_candidate(proc, model, pairs, steps: int, lr: float) -> None:
    """A text-only candidate (no image, no anchor): the G1 refusal stress test."""
    reset_lora(model)
    params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    for _ in range(steps):
        for user, assistant in pairs:
            msgs = [
                {"role": "user", "content": [{"type": "text", "text": user}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
            ]
            full = to_device(prep(proc, msgs, [], False))
            prompt = prep(proc, msgs[:1], [], True)
            labels = full["input_ids"].clone()
            labels[:, : prompt["input_ids"].shape[1]] = -100
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()


def load_dino():
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained(
        "facebook/dinov2-base", dtype=torch.float32, device_map={"": 0}
    )
    model.eval()
    return proc, model


@torch.no_grad()
def emb_dino(proc, model, image, mode: str = "cls"):
    batch = {
        k: v.to("cuda:0") for k, v in proc(images=image, return_tensors="pt").items()
    }
    h = model(**batch).last_hidden_state
    v = h[:, 0][0] if mode == "cls" else h.mean(dim=1)[0]
    return (v / v.norm()).cpu()


def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    proc = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    model = CLIPModel.from_pretrained(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", dtype=torch.float32, device_map={"": 0}
    )
    model.eval()
    return proc, model


@torch.no_grad()
def emb_clip(proc, model, image):
    batch = {k: v.to("cuda:0") for k, v in proc(images=image, return_tensors="pt").items()}
    out = model.vision_model(pixel_values=batch["pixel_values"])
    v = out.pooler_output
    if v is None:
        v = out.last_hidden_state.mean(dim=1)
    try:
        v = model.visual_projection(v)
    except Exception:
        pass
    v = v.reshape(-1)
    return (v / v.norm()).cpu()


def load_embedder():
    return load_dino()


def embed(iproc, imodel, image):
    return emb_dino(iproc, imodel, image, "cls")
