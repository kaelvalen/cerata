"""VLM mirror v1 diagnosis: KL strength sweep + restore verification (pinned).

Trains fact A at several lambdas; per arm: in-memory probe, canary, LoRA norm,
save -> zero -> restore with the exact tensor diff, restored probe.

    python vlm_mirror3.py --lams 0.1,1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from vlm_mirror import MODEL, PANEL_A, ask, load, set_lora  # noqa: E402
from vlm_mirror2 import CANARIES, PROBE, teach_text, train_delta_kl  # noqa: E402


def lora_named(model):
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lams", default="0.1,1.0")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="results/live_learning/vlm_mirror_sweep_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)

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
    teach = teach_text("Tira")
    pairs = [(teach, f"Not aldım: {teach}"), (PROBE, "Tira")]

    def contains(resp, token):
        return token.lower() in resp.lower()

    arms = {}
    for lam in [float(x) for x in args.lams.split(",")]:
        set_lora(model, None)
        train_delta_kl(proc, model, PANEL_A, pairs, args.steps, args.lr, lam=lam)
        in_memory = ask(proc, model, PANEL_A, PROBE)
        canary = ask(proc, model, None, CANARIES[1])
        norm = sum(float((p.detach().float() ** 2).sum()) for _, p in lora_named(model)) ** 0.5
        delta = {n: p.detach().float().cpu().clone() for n, p in lora_named(model)}
        set_lora(model, None)
        set_lora(model, delta)
        diff = max(
            float((p.detach().float().cpu() - delta[n]).abs().max()) for n, p in lora_named(model)
        )
        restored = ask(proc, model, PANEL_A, PROBE)
        arms[str(lam)] = {
            "in_memory": in_memory,
            "in_memory_teaches": contains(in_memory, "Tira"),
            "canary": canary,
            "canary_clean": not contains(canary, "Tira"),
            "lora_norm": round(norm, 4),
            "restore_diff": diff,
            "restored": restored,
            "restored_teaches": contains(restored, "Tira"),
        }
        print(
            f"lam {lam}: memory teaches {arms[str(lam)]['in_memory_teaches']} | "
            f"canary clean {arms[str(lam)]['canary_clean']} | norm "
            f"{arms[str(lam)]['lora_norm']} | diff {diff:.2e} | "
            f"restored teaches {arms[str(lam)]['restored_teaches']}",
            flush=True,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": arms}, ensure_ascii=False, indent=1))
    print(json.dumps({k: [v["in_memory"][:40], v["restored"][:40]] for k, v in arms.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
