"""Confirmatory pilot A harness (in-house arms; staged note in the prereg).

Arms: rag (memory-only), sequential (one shared LoRA), summed (composition of the
per-fact deltas), ours (per-fact deltas + MiniLM router + ledger). Metrics: efficacy,
paraphrase, distractor no-leak, router accuracy, revoke (token gone + retain), time and
storage; bootstrap 95% CIs over facts.

    python pilot.py --n 5      # smoke
    python pilot.py --n 50     # pilot A
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from facts import NUMBERS, generate  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from harness import load_model, make_lora  # noqa: E402
from ledger import CAP, DeltaStore, R  # noqa: E402

SYSTEM = "Answer briefly."
TAU = 0.656
CANARIES = ["What is the capital of France?", "What is 7 times 8?"]


class PilotStore(DeltaStore):
    """CAP=64 is the merged-adapter capacity; routed serving needs one delta at a time."""

    def materialize(self) -> None:
        if len(self.deltas) <= CAP:
            super().materialize()
        else:
            self._zero()


def ci(values, n=2000, seed=0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n, arr.size), replace=True).mean(axis=1)
    return [
        round(float(np.percentile(boots, 2.5)), 4),
        round(float(np.percentile(boots, 97.5)), 4),
    ]


def hit(resp, token):
    return token.lower() in resp.lower()


def base_answer(store, q, note=None):
    full = store.deltas
    store.deltas = {}
    store.materialize()
    system = SYSTEM if note is None else f"{SYSTEM} Context: {note}"
    resp = store.answer(q, system=system)
    store.deltas = full
    store.materialize()
    return resp


def serve_expert(store, fid, q):
    full = store.deltas
    store.deltas = {fid: full[fid]}
    store.materialize()
    resp = store.answer(q, system=SYSTEM)
    store.deltas = full
    store.materialize()
    return resp


class Router:
    """Entity-aware: a known subject must appear in the question, then the semantic key
    picks among that subject's facts; unseen subjects abstain (the pilot A failure)."""

    def __init__(self, enc, facts, tau=TAU):
        self.tau = tau
        self.keys = enc.encode([f["probe"] for f in facts], normalize_embeddings=True)
        self.subjects = [f["subject"].lower() for f in facts]

    def route(self, q, enc):
        ql = q.lower()
        cand = [i for i, s in enumerate(self.subjects) if s and s in ql]
        if not cand:
            return None, 0.0
        if len(cand) > 1:  # keep the longest subject match (prefix safety)
            cand = [max(cand, key=lambda i: len(self.subjects[i]))]
        v = enc.encode([q], normalize_embeddings=True)[0]
        sims = {i: float(self.keys[i] @ v) for i in cand}
        j = max(sims, key=sims.get)
        return (j if sims[j] >= self.tau else None), sims[j]


def summarize(rows):
    eff = [r["eff"] for r in rows]
    para = [r["para"] for r in rows]
    leak = [r["no_leak"] for r in rows]
    route = [r["route_ok"] for r in rows]
    return {
        "n": len(rows),
        "efficacy": round(float(np.mean(eff)), 4),
        "efficacy_ci": ci(eff),
        "paraphrase": round(float(np.mean(para)), 4),
        "paraphrase_ci": ci(para),
        "distractor_no_leak": round(float(np.mean(leak)), 4),
        "distractor_no_leak_ci": ci(leak),
        "route_accuracy": round(float(np.mean(route)), 4),
    }


def run_rag(store, facts, enc, router, words):
    rows = []
    for i, f in enumerate(facts):
        j, sim = router.route(f["probe"], enc)
        resp = (
            base_answer(store, f["probe"], note=facts[j]["teach"])
            if j is not None
            else base_answer(store, f["probe"])
        )
        pj, _ = router.route(f["paraphrase"], enc)
        resp_para = (
            base_answer(store, f["paraphrase"], note=facts[pj]["teach"])
            if pj is not None
            else base_answer(store, f["paraphrase"])
        )
        d_resp = base_answer(store, f["distractor"])
        rows.append(
            {
                "fid": f["id"],
                "eff": hit(resp, f["answer"]),
                "para": hit(resp_para, f["answer"]),
                "no_leak": not any(hit(d_resp, w) for w in words),
                "route_ok": j == i,
                "resp": resp[:70],
            }
        )
    return summarize(rows)


def run_ours(store, facts, enc, router, words):
    adds = []
    t0 = time.time()
    for f in facts:
        t = time.time()
        res = store.propose_and_commit(
            f["id"], fact_pairs(f), key=f["probe"], kld_limit=2.0
        )
        adds.append(
            {
                "fid": f["id"],
                "committed": res["committed"],
                "kld": round(res["kld"], 3),
                "seconds": round(time.time() - t, 2),
            }
        )
    train_seconds = time.time() - t0
    rows = []
    serve_times = []
    for i, f in enumerate(facts):
        j, _ = router.route(f["probe"], enc)
        t = time.time()
        resp = (
            serve_expert(store, f["id"], f["probe"])
            if j is not None
            else base_answer(store, f["probe"])
        )
        serve_times.append(time.time() - t)
        pj, _ = router.route(f["paraphrase"], enc)
        resp_para = (
            serve_expert(store, facts[pj]["id"], f["paraphrase"])
            if pj is not None
            else base_answer(store, f["paraphrase"])
        )
        dj, _ = router.route(f["distractor"], enc)
        d_resp = (
            base_answer(store, f["distractor"])
            if dj is None
            else serve_expert(store, facts[dj]["id"], f["distractor"])
        )
        rows.append(
            {
                "fid": f["id"],
                "eff": hit(resp, f["answer"]),
                "para": hit(resp_para, f["answer"]),
                "no_leak": not any(hit(d_resp, w) for w in words),
                "route_ok": j == i,
                "para_route_ok": pj == i,
                "dist_abstained": dj is None,
                "resp": resp[:70],
            }
        )
    m = summarize(rows)
    storage = sum(
        t.numel() * t.element_size()
        for d in store.deltas.values()
        for n in d
        for t in (d[n]["A"], d[n]["B"])
    )
    m.update(
        {
            "adds": adds,
            "train_seconds": round(train_seconds, 1),
            "mean_add_seconds": round(float(np.mean([a["seconds"] for a in adds])), 2),
            "mean_serve_seconds": round(float(np.mean(serve_times)), 2),
            "storage_mb": round(storage / 1e6, 1),
            "router_abstention_on_distractors": round(
                float(np.mean([r["dist_abstained"] for r in rows])), 4
            ),
            "paraphrase_route_accuracy": round(
                float(np.mean([r["para_route_ok"] for r in rows])), 4
            ),
        }
    )
    return m


def run_summed(store, facts, words):
    store.materialize()
    rows = []
    for f in facts:
        resp = store.answer(f["probe"], system=SYSTEM)
        resp_para = store.answer(f["paraphrase"], system=SYSTEM)
        d_resp = store.answer(f["distractor"], system=SYSTEM)
        rows.append(
            {
                "fid": f["id"],
                "eff": hit(resp, f["answer"]),
                "para": hit(resp_para, f["answer"]),
                "no_leak": not any(hit(d_resp, w) for w in words),
                "route_ok": False,
                "resp": resp[:70],
            }
        )
    return summarize(rows)


def train_sequential(store, facts, steps_per_fact=8):
    base = load_model(store.model_name)[1]
    m = make_lora(base, r=R)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=store.lr)
    with torch.no_grad():
        m.eval()
        store._base_logits = [store._logits(m, q) for q in store.kl_prompts]
    m.train()
    for f in facts:
        for _ in range(steps_per_fact):
            for user, assistant in fact_pairs(f):
                store._pair_step(m, opt, user, assistant)
    store._base_logits = None
    delta = {}
    for n, p in dict(m.named_parameters()).items():
        if "lora_A" in n:
            delta[n] = {"A": p.detach().float().cpu().clone()}
    for n, p in dict(m.named_parameters()).items():
        if "lora_B" in n:
            delta[n.replace("lora_B", "lora_A")]["B"] = p.detach().float().cpu().clone()
    del m, base
    torch.cuda.empty_cache()
    return delta


def run_sequential_arm(store, facts, words, steps_per_fact=8):
    saved = dict(store.deltas)
    store.deltas = {}
    store.materialize()
    delta = train_sequential(store, facts, steps_per_fact)
    store.deltas = {"seq": delta}
    store.materialize()
    rows = []
    for f in facts:
        resp = store.answer(f["probe"], system=SYSTEM)
        resp_para = store.answer(f["paraphrase"], system=SYSTEM)
        d_resp = store.answer(f["distractor"], system=SYSTEM)
        rows.append(
            {
                "fid": f["id"],
                "eff": hit(resp, f["answer"]),
                "para": hit(resp_para, f["answer"]),
                "no_leak": not any(hit(d_resp, w) for w in words),
                "route_ok": False,
                "resp": resp[:70],
            }
        )
    store.deltas = saved
    store.materialize()
    return summarize(rows)


def run_revoke(store, facts, enc, router, pre, sample=10):
    sample = min(sample, max(1, len(facts) // 2))
    sample_facts, keep = facts[:sample], facts[sample:]
    times = []
    for f in sample_facts:
        t = time.time()
        store.revoke(f["id"])
        times.append(time.time() - t)
    router_keep = Router(enc, keep)
    gone = []
    for f in sample_facts:
        j, _ = router_keep.route(f["probe"], enc)
        resp = (
            serve_expert(store, keep[j]["id"], f["probe"])
            if j is not None
            else base_answer(store, f["probe"])
        )
        gone.append(not hit(resp, f["answer"]))
    returned = []
    for f in sample_facts:
        j, _ = router_keep.route(f["probe"], enc)
        resp = (
            serve_expert(store, keep[j]["id"], f["probe"])
            if j is not None
            else base_answer(store, f["probe"])
        )
        returned.append(hit(resp, f["answer"]) == hit(pre[f["id"]], f["answer"]))
    retain = []
    for f in keep:
        j, _ = router_keep.route(f["probe"], enc)
        resp = (
            serve_expert(store, f["id"], f["probe"])
            if j is not None
            else base_answer(store, f["probe"])
        )
        retain.append(hit(resp, f["answer"]))
    return {
        "sample": len(sample_facts),
        "token_gone_rate": round(float(np.mean(gone)), 4),
        "token_gone_ci": ci(gone),
        "return_match_rate": round(float(np.mean(returned)), 4),
        "return_match_ci": ci(returned),
        "retain_rate": round(float(np.mean(retain)), 4),
        "retain_ci": ci(retain),
        "mean_revoke_seconds": round(float(np.mean(times)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--arms", default="rag,ours,summed,sequential")
    ap.add_argument("--facts", default=None, help="external facts JSON (else nonce generator)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.facts:
        data = json.loads(Path(args.facts).read_text())
        facts = data["facts"] if isinstance(data, dict) else data
    else:
        facts = generate(args.n, args.n)
    words = [f["answer"] for f in facts if f["answer"] not in NUMBERS]

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    router = Router(enc, facts)
    store = PilotStore(args.model, steps=16)
    store.kl_prompts = list(CANARIES)

    arms = args.arms.split(",")
    out = {
        "n": args.n,
        "arms": arms,
        "tau": TAU,
        "system": SYSTEM,
        "canaries": CANARIES,
        "results": {},
    }
    t0 = time.time()
    sample_n = min(10, max(1, len(facts) // 2))
    pre = {f["id"]: base_answer(store, f["probe"]) for f in facts[:sample_n]}
    if "rag" in arms:
        out["results"]["rag"] = run_rag(store, facts, enc, router, words)
    if "ours" in arms:
        out["results"]["ours"] = run_ours(store, facts, enc, router, words)
    if "summed" in arms:
        out["results"]["summed"] = run_summed(store, facts, words)
    if "sequential" in arms:
        out["results"]["sequential"] = run_sequential_arm(store, facts, words)
    if "ours" in arms:
        out["results"]["revoke"] = run_revoke(store, facts, enc, router, pre)
    out["wall_seconds"] = round(time.time() - t0, 1)

    path = Path(args.out or f"results/live_learning/confirm/pilot_n{args.n}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"n={args.n} wall={out['wall_seconds']}s -> {path}")
    for arm, m in out["results"].items():
        if arm == "revoke":
            print(f"revoke: gone {m['token_gone_rate']} {m['token_gone_ci']} | "
                  f"retain {m['retain_rate']} {m['retain_ci']}")
        else:
            print(
                f"{arm}: eff {m['efficacy']} {m['efficacy_ci']} | "
                f"para {m['paraphrase']} {m['paraphrase_ci']} | "
                f"no_leak {m['distractor_no_leak']} | route {m['route_accuracy']}"
            )
    if "ours" in out["results"]:
        o = out["results"]["ours"]
        print(f"ours extras: add {o['mean_add_seconds']}s serve {o['mean_serve_seconds']}s "
              f"storage {o['storage_mb']}MB abstention {o['router_abstention_on_distractors']} "
              f"para_route {o['paraphrase_route_accuracy']}")


if __name__ == "__main__":
    main()
