"""Official EasyEdit GRACE / WISE baselines on CounterFact N=200.

Pinned protocol: docs/LIVING_MODEL_CONFIRMATORY_PREREG.md section 53.
Model: Qwen/Qwen2.5-1.5B-Instruct. Data: external_counterfact_n200.json, aligned
to raw/counterfact.json. Sequential editing of all facts. Reports EasyEdit's own
post/rephrase/locality metrics AND our-protocol (greedy + contains) evaluation.

    python official_baseline.py --method grace --limit 2
    python official_baseline.py --method wise

Run from a scratch cwd (EasyEdit writes ./logs/results.json relative to cwd).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / ".deps" / "EasyEdit"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from easyeditor import BaseEditor, GraceHyperParams, WISEHyperParams  # noqa: E402

SYSTEM = "Answer briefly."
CANARIES = ["What is the capital of France?", "What is 7 times 8?"]
HP_REL = {
    "grace": "hparams/GRACE/qwen2.5-1.5b.yaml",
    "wise": "hparams/WISE/qwen2.5-1.5b.yaml",
}
HP_CLS = {"grace": GraceHyperParams, "wise": WISEHyperParams}


def log(msg: str, logf: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(logf, "a") as f:
        f.write(line + "\n")


def ci(values, n=2000, seed=0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n, arr.size), replace=True).mean(axis=1)
    return [round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)]


def hit(resp: str, token: str) -> bool:
    return token.lower() in resp.lower()


def adapt_counterfact(item: dict):
    """Exact adaptation used by confirm/external.py (for raw alignment)."""
    rw = item.get("requested_rewrite") or {}
    prompt = rw.get("prompt") or ""
    subject = rw.get("subject") or ""
    target = (rw.get("target_new") or {}).get("str") or ""
    paras = item.get("paraphrase_prompts") or []
    if not (prompt and subject and target and paras):
        return None
    probe = prompt.replace("{}", subject).strip()
    para = paras[0].replace("{}", subject).strip()
    if para == probe or subject.lower() not in para.lower():
        return None
    return {
        "subject": subject,
        "probe": probe,
        "answer": target,
        "target_true": (rw.get("target_true") or {}).get("str"),
    }


def build_raw_alignment(raw_path: Path, facts):
    """Map each fact to its raw/counterfact.json case_id (subject+probe match)."""
    items = json.loads(raw_path.read_text())
    lookup = {}
    for i, it in enumerate(items):
        a = adapt_counterfact(it)
        if a:
            lookup.setdefault((a["subject"], a["probe"]), (i, a.get("target_true")))
    raw_index, raw_true, missing = {}, {}, []
    for f in facts:
        got = lookup.get((f["subject"], f["probe"]))
        if got is None:
            raw_index[f["id"]] = None
            raw_true[f["id"]] = None
            missing.append(f["id"])
        else:
            raw_index[f["id"]] = got[0]
            raw_true[f["id"]] = got[1]
    return raw_index, raw_true, missing


@torch.no_grad()
def gen(model, tok, prompt: str, device, max_new_tokens: int = 64) -> str:
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def gen_raw(model, tok, prompt: str, device, max_new_tokens: int = 8) -> str:
    """Greedy continuation WITHOUT chat template (the format EasyEdit GRACE edits)."""
    enc = tok(prompt, return_tensors="pt")
    ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        use_cache=False,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def next_logits(model, tok, q: str, device) -> torch.Tensor:
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
    out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out
    return logits[0, -1].float().cpu()


def canary_kld(base_logits, edited_logits):
    ks = []
    for b, a in zip(base_logits, edited_logits):
        pb, pa = torch.softmax(b, -1), torch.softmax(a, -1)
        ks.append(float((pb * (pb / pa).log()).sum() + (pa * (pa / pb).log()).sum()))
    return sum(ks) / len(ks), ks


def phase_mean(metrics, phase: str, key: str):
    vals = []
    for m in metrics:
        v = m.get(phase, {}).get(key)
        if v is None:
            continue
        if isinstance(v, list):
            vals.append(float(np.mean(v)))
        else:
            vals.append(float(v))
    return round(float(np.mean(vals)), 4) if vals else None


def easyedit_summary(metrics):
    out = {}
    for phase in ("pre", "post"):
        d = {}
        for key in ("rewrite_acc", "rephrase_acc", "rewrite_F1"):
            v = phase_mean(metrics, phase, key)
            if v is not None:
                d[key] = v
        loc = []
        for m in metrics:
            sec = m.get(phase, {}).get("locality", {})
            if isinstance(sec, dict) and "neighborhood_acc" in sec:
                loc.append(float(np.mean(sec["neighborhood_acc"])))
        if loc:
            d["locality_neighborhood_acc"] = round(float(np.mean(loc)), 4)
        out[phase] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["grace", "wise"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N facts")
    ap.add_argument("--hparams", default=None, help="override the default hparams path")
    ap.add_argument("--facts", default=str(ROOT / "results/live_learning/confirm/external_counterfact_n200.json"))
    ap.add_argument("--raw", default=str(ROOT / "results/live_learning/confirm/raw/counterfact.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    method = args.method
    facts_all = json.loads(Path(args.facts).read_text())["facts"]
    facts = facts_all[: args.limit] if args.limit else facts_all
    out_path = Path(args.out) if args.out else (
        ROOT / f"results/live_learning/confirm/official_{method}_n{len(facts)}.json")
    log_path = Path(args.log) if args.log else (
        ROOT / f"results/live_learning/confirm/official_{method}.log")

    log(f"=== official {method.upper()} baseline start: n={len(facts)} ===", log_path)
    raw_index, raw_true, missing = build_raw_alignment(Path(args.raw), facts)
    if missing:
        log(f"WARNING raw alignment missing for {len(missing)} facts: {missing[:5]}", log_path)

    hp_path = ROOT / ".deps" / "EasyEdit" / (args.hparams or HP_REL[method])
    hparams = HP_CLS[method].from_hparams(str(hp_path))
    # WISE's sequential-edit guard reads config.sequential_edit (absent from the
    # text dataclass); set it so the adapter accumulates edits instead of resetting.
    hparams.sequential_edit = True
    # Memory adaptation for the shared ~8 GB GPU: the official 7B yamls use the
    # fp32 default (BaseEditor torch_dtype=float32), which OOMs for WISE here.
    hparams.fp16 = True
    log(f"hparams={HP_REL[method]} model={hparams.model_name} device={hparams.device} "
        f"n_iter={hparams.n_iter} lr={hparams.edit_lr} inner={hparams.inner_params} "
        f"fp16={hparams.fp16}", log_path)

    editor = BaseEditor.from_hparams(hparams)
    device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
    tok_eval = AutoTokenizer.from_pretrained(hparams.model_name)
    base_model = editor.model
    log(f"model loaded; tokenizer eos={tok_eval.eos_token!r}", log_path)

    # --- base references (before any edit) ---
    base_dist, base_dist_gt = {}, {}
    for f in facts:
        r = gen(base_model, tok_eval, f["distractor"], device, 64)
        base_dist[f["id"]] = r
        toks = tok_eval.encode(r, add_special_tokens=False)[:12]
        base_dist_gt[f["id"]] = (tok_eval.decode(toks).strip() or "unknown")
    base_canary = [next_logits(base_model, tok_eval, q, device) for q in CANARIES]
    log(f"base references done ({len(facts)} distractors, {len(CANARIES)} canaries)", log_path)

    # --- EasyEdit inputs (hallucination-branch format) ---
    prompts = [f["probe"] for f in facts]
    target_new = [f["answer"] for f in facts]
    subject = [f["subject"] for f in facts]
    rephrase_prompts = [f["paraphrase"] for f in facts]
    ground_truth = [raw_true[f["id"]] for f in facts]
    loc_prompts = [f"{f['distractor']} {base_dist_gt[f['id']]}" for f in facts]
    locality_inputs = {
        "neighborhood": {
            "prompt": [f["distractor"] for f in facts],
            "ground_truth": [base_dist_gt[f["id"]] for f in facts],
        }
    }

    quirks = []
    log("starting sequential edit ...", log_path)
    t0 = time.time()
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        ground_truth=ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        subject=subject,
        loc_prompts=loc_prompts,
        sequential_edit=True,
        eval_metric="token_em",
    )
    runtime = time.time() - t0
    log(f"editing finished in {runtime:.1f}s", log_path)

    # --- our-protocol evaluation of the final edited model ---
    words = [f["answer"] for f in facts]
    rows = []
    for f in facts:
        resp = gen(edited_model, tok_eval, f["probe"], device, 64)
        resp_para = gen(edited_model, tok_eval, f["paraphrase"], device, 64)
        d_resp = gen(edited_model, tok_eval, f["distractor"], device, 64)
        # diagnostic: same greedy+contains but with the raw (no chat template) prompt,
        # the format EasyEdit's GRACE edits/keys are built on.
        resp_raw = gen_raw(edited_model, tok_eval, f["probe"], device, 8)
        rows.append({
            "fid": f["id"],
            "eff": hit(resp, f["answer"]),
            "para": hit(resp_para, f["answer"]),
            "no_leak": not any(
                hit(d_resp, w) and not hit(base_dist[f["id"]], w) for w in words),
            "eff_raw_prompt": hit(resp_raw, f["answer"]),
            "resp": resp[:70],
            "resp_para": resp_para[:70],
            "resp_raw": resp_raw[:70],
            "dist_resp": d_resp[:70],
            "base_dist_resp": base_dist[f["id"]][:70],
        })
        log(f"ours {f['id']}: eff={rows[-1]['eff']} para={rows[-1]['para']} "
            f"no_leak={rows[-1]['no_leak']} eff_raw={rows[-1]['eff_raw_prompt']}", log_path)
    eff = [r["eff"] for r in rows]
    para = [r["para"] for r in rows]
    leak = [r["no_leak"] for r in rows]
    eff_raw = [r["eff_raw_prompt"] for r in rows]
    our_summary = {
        "n": len(rows),
        "efficacy": round(float(np.mean(eff)), 4) if rows else None,
        "efficacy_ci": ci(eff),
        "paraphrase": round(float(np.mean(para)), 4) if rows else None,
        "paraphrase_ci": ci(para),
        "distractor_no_leak": round(float(np.mean(leak)), 4) if rows else None,
        "distractor_no_leak_ci": ci(leak),
        "efficacy_raw_prompt_diagnostic": round(float(np.mean(eff_raw)), 4) if rows else None,
    }

    # --- canary KLD on the final edited model ---
    kld_info = {"value": None, "per_canary": None, "status": "not attempted"}
    try:
        edited_canary = [next_logits(edited_model, tok_eval, q, device) for q in CANARIES]
        kld, ks = canary_kld(base_canary, edited_canary)
        kld_info = {"value": round(kld, 4), "per_canary": [round(x, 4) for x in ks],
                    "status": "ok"}
        log(f"canary KLD = {kld:.4f} ({[round(x,3) for x in ks]})", log_path)
    except Exception as e:  # noqa: BLE001
        kld_info = {"value": None, "per_canary": None,
                    "status": f"skipped: {type(e).__name__}: {e}"}
        quirks.append(f"canary KLD skipped: {type(e).__name__}: {e}")
        log(f"canary KLD skipped: {type(e).__name__}: {e}", log_path)

    result = {
        "method": method.upper(),
        "model": hparams.model_name,
        "n": len(facts),
        "sequential_edit": True,
        "hparams_file": HP_REL[method],
        "eval_metric": "token_em",
        "system": SYSTEM,
        "canaries": CANARIES,
        "runtime_seconds": round(runtime, 1),
        "easyedit": easyedit_summary(metrics),
        "our_protocol": {"summary": our_summary, "rows": rows},
        "canary_kld": kld_info,
        "raw_alignment": raw_index,
        "raw_alignment_missing": missing,
        "quirks": quirks,
        "easyedit_metrics": metrics,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                   default=lambda o: float(o) if isinstance(o, np.floating)
                                   else (o.tolist() if isinstance(o, np.ndarray) else str(o))))
    log(f"wrote {out_path}", log_path)
    log(f"easyedit summary: {json.dumps(result['easyedit'])}", log_path)
    log(f"our-protocol summary: {json.dumps(our_summary)}", log_path)
    log(f"=== official {method.upper()} baseline done ===", log_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise
