"""Model loading and a minimal chat wrapper (sandbox)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def chat(tokenizer, model, messages, max_new_tokens: int = 64) -> str:
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(encoded, "keys"):  # transformers 5: BatchEncoding
        ids = encoded["input_ids"]
    else:
        ids = encoded
    ids = ids.to(model.device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def make_lora(model, r: int = 16):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)
