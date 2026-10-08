"""LoRA delta improvement ablation (pinned, prereg §44).

CounterFact N=50, per-fact deltas, 16 steps, KL anchor against the true zeroed
base; variants: baseline, olora, lora_plus, mlp, dora. Readings: efficacy,
paraphrase (delta active alone), mean canary KLD (10-delta sample), train seconds.

    python lora_ablation.py --facts results/live_learning/confirm/external_counterfact_n50.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from harness import chat, load_model  # noqa: E402
from pilot import SYSTEM, PilotStore, hit  # noqa: E402

VARIANTS = ["baseline", "olora", "lora_plus", "mlp", "dora"]


def make_variant_lora(base, variant: str):
    from peft import LoraConfig, get_peft_model

    kw = dict(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM")
    targets = ["q_proj", "v_proj"]
    if variant == "mlp":
        targets += ["gate_proj", "up_proj", "down_proj"]
    kw["target_modules"] = targets
    if variant == "olora":
        kw["init_lora_weights"] = "olora"
    if variant == "dora":
        kw["use_dora"] = True
    return get_peft_model(base, LoraConfig(**kw))


def extract_delta(m) -> dict:
    delta: dict = {}
    for n, p in dict(m.named_parameters()).items():
        if "lora_A" in n:
            delta[n] = {"A": p.detach().float().cpu().clone()}
    for n, p in dict(m.named_parameters()).items():
        if "lora_B" in n:
            delta[n.replace("lora_B", "lora_A")]["B"] = p.detach().float().cpu().clone()
    return delta


def train_variant(store, variant: str, pairs):
    base = load_model(store.model_name)[1]
    m = make_variant_lora(base, variant)
    params_a = [p for n, p in m.named_parameters() if "lora_A" in n and p.requires_grad]
    params_b = [p for n, p in m.named_parameters() if "lora_B" in n and p.requires_grad]
    if variant == "lora_plus":
        opt = torch.optim.AdamW(
            [
                {"params": params_a, "lr": store.lr},
                {"params": params_b, "lr": 8 * store.lr},
            ]
        )
    else:
        opt = torch.optim.AdamW(params_a + params_b, lr=store.lr)
    with torch.no_grad():
        m.eval()
        init = {n: p.clone() for n, p in m.named_parameters() if "lora_" in n}
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.zero_()
        store._base_logits = [store._logits(m, q) for q in store.kl_prompts]
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.copy_(init[n])
    m.train()
    for _ in range(16):
        for user, assistant in pairs:
            store._pair_step(m, opt, user, assistant)
    store._base_logits = None
    delta = extract_delta(m)
    del m, base
    torch.cuda.empty_cache()
    return delta


def set_lora_params(model, delta=None):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                if delta is None or n not in delta:
                    p.zero_()
                else:
                    p.copy_(delta[n].to(p.device, p.dtype))


def eval_variant(store, variant: str, deltas, facts) -> dict:
    """Variant-aware evaluation: a fresh model with the variant's LoRA config."""
    base = load_model(store.model_name)[1]
    m = make_variant_lora(base, variant)
    m.eval()
    set_lora_params(m, None)
    with torch.no_grad():
        base_logits = [store._logits(m, q) for q in store.kl_prompts]
    eff, para, klds = [], [], []
    for i, f in enumerate(facts):
        set_lora_params(m, deltas[f["id"]])
        resp = chat(
            store.tok,
            m,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f["probe"]},
            ],
            max_new_tokens=24,
        )
        rp = chat(
            store.tok,
            m,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f["paraphrase"]},
            ],
            max_new_tokens=24,
        )
        eff.append(hit(resp, f["answer"]))
        para.append(hit(rp, f["answer"]))
        if i < 10:
            with torch.no_grad():
                cur = [store._logits(m, q) for q in store.kl_prompts]
            ks = []
            for b, a in zip(base_logits, cur):
                pb, pa = torch.softmax(b, -1), torch.softmax(a, -1)
                ks.append(
                    float((pb * (pb / pa).log()).sum() + (pa * (pa / pb).log()).sum())
                )
            klds.append(sum(ks) / len(ks))
    set_lora_params(m, None)
    del m, base
    torch.cuda.empty_cache()
    return {
        "efficacy": round(float(np.mean(eff)), 3),
        "paraphrase": round(float(np.mean(para)), 3),
        "kld_mean": round(float(np.mean(klds)), 3),
        "kld_max": round(float(np.max(klds)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="results/live_learning/confirm/external_counterfact_n50.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/confirm/lora_ablation_n50.json")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    data = json.loads(Path(args.facts).read_text())
    facts = data["facts"] if isinstance(data, dict) else data
    store = PilotStore(args.model, steps=16)
    store.kl_prompts = ["What is the capital of France?", "What is 7 times 8?"]
    store.model.to("cpu")  # the store model is unused here; free the GPU for the variants
    torch.cuda.empty_cache()

    report = {}
    for variant in args.variants.split(","):
        t0 = time.time()
        deltas = {}
        for f in facts:
            pairs = [(f["teach"], f"Not aldım: {f['teach']}"), (f["probe"], f["answer"])]
            deltas[f["id"]] = train_variant(store, variant, pairs)
        train_s = time.time() - t0
        report[variant] = {"train_seconds": round(train_s, 1)}
        report[variant].update(eval_variant(store, variant, deltas, facts))
        print(variant, json.dumps(report[variant]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"facts": args.facts, "variants": report}, ensure_ascii=False, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()
