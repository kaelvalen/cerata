"""VLM scene variants: key robustness and expert generalisation (pinned).

P1 (row of three shapes) and its variants - jittered, scaled, background-shifted - plus
P2 (a different row). One expert on P1; tau_v from the pinned rule; does the router
accept variants, does the expert answer them, does P2 abstain?

    python vlm_variants.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from vlm_keys import emb_clip  # noqa: E402
from vlm_ledger import VlmDeltaStore  # noqa: E402
from vlm_mirror import BLUE, GREEN, MODEL, RED, ask, set_lora  # noqa: E402
from vlm_mirror2 import CANARIES, PROBE, teach_text  # noqa: E402


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
PANELS = {"p1": P1, "p1j": P1J, "p1s": P1S, "p1bg": P1BG, "p2": P2}


def pairs_for(teach: str, code: str):
    return [(teach, f"Not aldım: {teach}"), (PROBE, code)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="results/live_learning/vlm_variants_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    store = VlmDeltaStore(args.model)

    def contains(resp, token):
        return token.lower() in resp.lower()

    set_lora(store.model, None)
    base_p1 = ask(store.proc, store.model, P1, PROBE)
    add = store.propose_and_commit("p1", P1, pairs_for(teach_text("Tira"), "Tira"))

    emb = {name: emb_clip(store.iproc, store.imodel, img) for name, img in PANELS.items()}
    sims = {name: round(float(emb["p1"] @ emb[name]), 4) for name in PANELS}
    min_own = min(sims[v] for v in ("p1j", "p1s", "p1bg"))
    max_cross = sims["p2"]
    separable = min_own > max_cross
    tau_v = round((min_own + max_cross) / 2, 4) if separable else store.tau
    store.tau = tau_v

    served = {}
    for name, img in PANELS.items():
        resp, chosen, cs = store.serve(img)
        served[name] = {"resp": resp, "chosen": chosen, "sims": cs}

    set_lora(store.model, None)
    canary = ask(store.proc, store.model, None, CANARIES[1])

    checks = {
        "base_unknown": not contains(base_p1, "Tira"),
        "add_committed": add["committed"],
        "tau_separable": separable,
        "p1_served": served["p1"]["chosen"] == "p1" and contains(served["p1"]["resp"], "Tira"),
        "p2_abstains": served["p2"]["chosen"] is None
        and not contains(served["p2"]["resp"], "Tira")
        and not contains(served["p2"]["resp"], "Vok"),
        "canary_clean": not contains(canary, "Tira"),
    }
    variants = {
        v: {
            "routed": served[v]["chosen"] == "p1",
            "answers_tira": contains(served[v]["resp"], "Tira"),
            "resp": served[v]["resp"][:80],
        }
        for v in ("p1j", "p1s", "p1bg")
    }
    result = {
        "model": args.model,
        "checks": checks,
        "add": add,
        "sims": sims,
        "min_own_variant": min_own,
        "max_cross_scene": max_cross,
        "tau_v": tau_v,
        "variants": variants,
        "served": {k: {"resp": v["resp"], "chosen": v["chosen"], "sims": v["sims"]} for k, v in served.items()},
        "canary": canary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("sims:", sims, "| tau_v:", tau_v, "| separable:", separable)
    print("variants:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "resp"} for k, v in variants.items()}))
    print("p1:", served["p1"]["resp"][:40], "| p2:", served["p2"]["resp"][:50])


if __name__ == "__main__":
    main()
