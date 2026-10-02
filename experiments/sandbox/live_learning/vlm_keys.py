"""Key comparison for the VLM mirror panels (pinned): pure measurement, no VLM.

DINOv2 CLS / mean-patch, CLIP image embedding, and their concatenation: A vs B' cross
similarity and the maximum distractor similarity to either key.

    python vlm_keys.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from vlm_mirror import GREEN, PANEL_A, PANEL_B, RED, panel  # noqa: E402
from vlm_mirror4 import PANEL_B2  # noqa: E402

PANEL_D2 = panel([("square", (120, 120, 120), (112, 112))])
PANEL_D3 = panel([("circle", RED, (72, 72)), ("triangle", GREEN, (152, 152))])
PANELS = {
    "a": PANEL_A,
    "b": PANEL_B2,
    "d1": PANEL_B,
    "d2": PANEL_D2,
    "d3": PANEL_D3,
}


def load_dino():
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained(
        "facebook/dinov2-base", dtype=torch.float32, device_map={"": 0}
    )
    model.eval()
    return proc, model


@torch.no_grad()
def emb_dino(proc, model, image, mode: str):
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


def main() -> None:
    dino_proc, dino = load_dino()
    clip_proc, clip = load_clip()
    vecs = {}
    for key, img in PANELS.items():
        cls = emb_dino(dino_proc, dino, img, "cls")
        mean = emb_dino(dino_proc, dino, img, "mean")
        c = emb_clip(clip_proc, clip, img)
        concat = torch.cat([cls, c])
        concat = concat / concat.norm()
        vecs[key] = {"dinov2_cls": cls, "dinov2_mean": mean, "clip": c, "concat": concat}

    report = {}
    for mod in ("dinov2_cls", "dinov2_mean", "clip", "concat"):
        v = {k: vecs[k][mod] for k in PANELS}
        cross = float(v["a"] @ v["b"])
        dist = {
            k: round(max(float(v["a"] @ v[k]), float(v["b"] @ v[k])), 4)
            for k in ("d1", "d2", "d3")
        }
        worst = max(cross, *dist.values())
        report[mod] = {
            "cross_a_b": round(cross, 4),
            "distractors": dist,
            "worst_offkey": round(worst, 4),
            "margin": round(1.0 - worst, 4),
        }
        print(mod, json.dumps(report[mod]), flush=True)
    out = Path("results/live_learning/vlm_key_compare.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
