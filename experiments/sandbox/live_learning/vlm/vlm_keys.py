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
from vlm_core import (  # noqa: E402
    PANEL_A,
    PANEL_B,
    PANEL_B2,
    PANEL_D2,
    PANEL_D3,
    emb_clip,
    emb_dino,
    load_clip,
    load_dino,
)

PANELS = {
    "a": PANEL_A,
    "b": PANEL_B2,
    "d1": PANEL_B,
    "d2": PANEL_D2,
    "d3": PANEL_D3,
}


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
