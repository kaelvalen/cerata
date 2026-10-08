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
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from facts import NUMBERS, generate  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from harness import load_model, make_lora  # noqa: E402
from ledger import CAP, DeltaStore, R  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

SYSTEM = "Answer briefly."
TAU = 0.656
CANARIES = ["What is the capital of France?", "What is 7 times 8?"]
BASE_DIST: dict = {}
_LIBC = ctypes.CDLL("libc.so.6")


def trim() -> None:
    """Return glibc arena memory to the OS (the training loop's RSS balloon)."""
    _LIBC.malloc_trim(0)


class PilotStore(DeltaStore):
    """CAP=64 is the merged-adapter capacity; routed serving needs one delta at a time.
    The store model is parked on CPU while a candidate trains (the fresh base needs the
    GPU; peak memory halves)."""

    def materialize(self) -> None:
        if len(self.deltas) <= CAP:
            super().materialize()
        else:
            self._zero()

    def _train_delta(self, pairs):
        self.model.to("cpu")
        torch.cuda.empty_cache()
        try:
            return super()._train_delta(pairs)
        finally:
            self.model.to("cuda:0")


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


def base_answer(store, q, note=None, system=None):
    full = store.deltas
    store.deltas = {}
    store.materialize()
    if system is None:
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


def _norm(s: str) -> str:
    return s.lower().replace("-", " ").replace("_", " ").strip()


def _osa(a: str, b: str) -> int:
    """Optimal string alignment (adjacent transposition counts as 1)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


class Router:
    """Entity-aware v3: a normalized substring match (longest wins, prefix safety) or
    a fuzzy token match (edit distance <=1, <=2 for tokens of length >=6) selects the
    candidate facts; the semantic key picks among them; no candidate -> abstain."""

    def __init__(self, enc, facts, tau=TAU):
        self.tau = tau
        self.keys = enc.encode([f["probe"] for f in facts], normalize_embeddings=True)
        self.subjects = [_norm(f["subject"]) for f in facts]
        self.tokens = [s.split() for s in self.subjects]

    def _entity_candidates(self, q: str) -> list:
        qn = _norm(q)
        qtok = qn.split()
        exact = [i for i, sn in enumerate(self.subjects) if sn and sn in qn]
        if exact:
            return [max(exact, key=lambda i: len(self.subjects[i]))]
        cand = []
        for i, toks in enumerate(self.tokens):
            for st in toks:
                if len(st) < 4:
                    continue
                thresh = 2 if len(st) >= 6 else 1
                if any(st == qt or _osa(st, qt) <= thresh for qt in qtok):
                    cand.append(i)
                    break
        return cand

    def route(self, q, enc):
        cand = self._entity_candidates(q)
        if not cand:
            return None, 0.0
        v = enc.encode([q], normalize_embeddings=True)[0]
        sims = {i: float(self.keys[i] @ v) for i in cand}
        j = max(sims, key=sims.get)
        return (j if sims[j] >= self.tau else None), sims[j]


class RouterV2:
    """Pilot's substring gate (reference): exact lowercased subject substring."""

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


def rag_system(mode, facts, j):
    f = facts[j]
    if mode == "plain":
        return f"{SYSTEM} Context: {f['teach']}"
    if mode == "instruct":
        return (
            f"{SYSTEM} Use the context even if it contradicts what you know. "
            f"Context: {f['teach']}"
        )
    if mode == "qa":
        return f"{SYSTEM} Context:\nQ: {f['probe']}\nA: {f['answer']}"
    demo = facts[(j + 1) % len(facts)]
    return (
        f"{SYSTEM} Follow the context over your prior knowledge.\n"
        f"Example:\nQ: {demo['probe']}\nA: {demo['answer']}\n"
        f"Context: {f['teach']}"
    )


def run_rag(store, facts, enc, router, words, mode="plain"):
    rows = []
    for i, f in enumerate(facts):
        j, sim = router.route(f["probe"], enc)
        resp = (
            base_answer(store, f["probe"], system=rag_system(mode, facts, j))
            if j is not None
            else base_answer(store, f["probe"])
        )
        pj, _ = router.route(f["paraphrase"], enc)
        resp_para = (
            base_answer(store, f["paraphrase"], system=rag_system(mode, facts, pj))
            if pj is not None
            else base_answer(store, f["paraphrase"])
        )
        d_resp = base_answer(store, f["distractor"])
        rows.append(
            {
                "fid": f["id"],
                "eff": hit(resp, f["answer"]),
                "para": hit(resp_para, f["answer"]),
                "no_leak": not any(
                    hit(d_resp, w) and not hit(BASE_DIST[f["id"]], w) for w in words
                ),
                "route_ok": j == i,
                "resp": resp[:70],
            }
        )
    return summarize(rows)


def train_part(store, part, adds, offset=0):
    t0 = time.time()
    for k, f in enumerate(part):
        i = offset + k
        if f["id"] in store.deltas:
            continue
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
        trim()
        if (i + 1) % 50 == 0:
            print(f"ours: trained {i + 1} ({time.time() - t0:.0f}s elapsed)", flush=True)
    return time.time() - t0


def eval_ours(store, facts, enc, router, words, train_seconds, adds):
    print(f"ours: evaluating ({train_seconds:.0f}s training)", flush=True)
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
                "no_leak": not any(
                    hit(d_resp, w) and not hit(BASE_DIST[f["id"]], w) for w in words
                ),
                "route_ok": j == i,
                "para_route_ok": pj == i,
                "dist_abstained": dj is None,
                "resp": resp[:70],
            }
        )
        trim()
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


def run_ours(store, facts, enc, router, words, capped=True, train=True, adds=None):
    adds = [] if adds is None else adds
    if train and capped:
        train_seconds = train_part(store, facts, adds)
    elif train:
        t0 = time.time()
        for f in facts:
            secs = store.add(f["id"], fact_pairs(f), key=f["probe"])
            adds.append(
                {
                    "fid": f["id"],
                    "committed": True,
                    "kld": None,
                    "seconds": round(secs, 2),
                }
            )
        train_seconds = time.time() - t0
    else:
        train_seconds = 0.0
    return eval_ours(store, facts, enc, router, words, train_seconds, adds)


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
                "no_leak": not any(
                    hit(d_resp, w) and not hit(BASE_DIST[f["id"]], w) for w in words
                ),
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
                "no_leak": not any(
                    hit(d_resp, w) and not hit(BASE_DIST[f["id"]], w) for w in words
                ),
                "route_ok": False,
                "resp": resp[:70],
            }
        )
    store.deltas = saved
    store.materialize()
    return summarize(rows)


class ForceToken(LogitsProcessor):
    def __init__(self, token_id: int, bias: float = 10.0):
        self.token_id = token_id
        self.bias = bias

    def __call__(self, input_ids, scores):
        scores[:, self.token_id] += self.bias
        return scores


def grace_answer(store, q, token_id):
    ids = store.tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(store.model.device)
    out = store.model.generate(
        input_ids=ids,
        max_new_tokens=24,
        do_sample=False,
        pad_token_id=store.tok.eos_token_id,
        logits_processor=LogitsProcessorList([ForceToken(token_id)]),
    )
    return store.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_grace(store, facts, enc, router, words):
    """GRACE-style: codebook + deferral radius, value forced in logit space."""
    saved = dict(store.deltas)
    store.deltas = {}
    store.materialize()
    token_ids = [store.tok.encode(f["answer"], add_special_tokens=False)[0] for f in facts]
    rows = []
    for i, f in enumerate(facts):
        j, _ = router.route(f["probe"], enc)
        resp = (
            grace_answer(store, f["probe"], token_ids[j])
            if j is not None
            else base_answer(store, f["probe"])
        )
        pj, _ = router.route(f["paraphrase"], enc)
        resp_para = (
            grace_answer(store, f["paraphrase"], token_ids[pj])
            if pj is not None
            else base_answer(store, f["paraphrase"])
        )
        dj, _ = router.route(f["distractor"], enc)
        d_resp = (
            grace_answer(store, f["distractor"], token_ids[dj])
            if dj is not None
            else base_answer(store, f["distractor"])
        )
        rows.append(
            {
                "fid": f["id"],
                "eff": hit(resp, f["answer"]),
                "para": hit(resp_para, f["answer"]),
                "no_leak": not any(
                    hit(d_resp, w) and not hit(BASE_DIST[f["id"]], w) for w in words
                ),
                "route_ok": j == i,
                "resp": resp[:70],
            }
        )
    store.deltas = saved
    store.materialize()
    return summarize(rows)


def run_wise(store, facts, enc, router, words):
    """WISE-style: side memory only for facts the base does not already answer."""
    base = {f["id"]: base_answer(store, f["probe"]) for f in facts}
    conflicts = [f for f in facts if not hit(base[f["id"]], f["answer"])]
    side = Router(enc, conflicts)
    rows = []
    for f in facts:
        j, _ = side.route(f["probe"], enc)
        served_side = j is not None and conflicts[j]["id"] == f["id"]
        resp = (
            base_answer(store, f["probe"], note=conflicts[j]["teach"])
            if served_side
            else base_answer(store, f["probe"])
        )
        pj, _ = side.route(f["paraphrase"], enc)
        served_para = pj is not None and conflicts[pj]["id"] == f["id"]
        resp_para = (
            base_answer(store, f["paraphrase"], note=conflicts[pj]["teach"])
            if served_para
            else base_answer(store, f["paraphrase"])
        )
        dj, _ = side.route(f["distractor"], enc)
        d_resp = (
            base_answer(store, f["distractor"], note=conflicts[dj]["teach"])
            if dj is not None
            else base_answer(store, f["distractor"])
        )
        is_conflict = f["id"] in {c["id"] for c in conflicts}
        rows.append(
            {
                "fid": f["id"],
                "eff": hit(resp, f["answer"]),
                "para": hit(resp_para, f["answer"]),
                "no_leak": not any(
                    hit(d_resp, w) and not hit(BASE_DIST[f["id"]], w) for w in words
                ),
                "route_ok": served_side if is_conflict else not served_side,
                "resp": resp[:70],
            }
        )
    return summarize(rows)


def run_revoke(store, facts, enc, router, pre, sample=100):
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
    ap.add_argument("--ckpt", default=None, help="checkpoint path for chunked runs")
    ap.add_argument("--train-range", default=None, help="train facts[lo:hi] and exit")
    ap.add_argument("--eval-only", action="store_true", help="load ckpt and evaluate only")
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
    ckpt_path = (
        Path(args.ckpt)
        if args.ckpt
        else Path(f"results/live_learning/confirm/pilot_n{args.n}.ckpt")
    )
    adds: list = []
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        store.deltas = st["deltas"]
        store.keys = st["keys"]
        adds = st.get("adds", [])
        print(f"resumed {len(store.deltas)}/{len(facts)}", flush=True)
    if args.train_range:
        lo, hi = [int(x) for x in args.train_range.split(":")]
        t = train_part(store, facts[lo:hi], adds, offset=lo)
        torch.save({"deltas": store.deltas, "keys": store.keys, "adds": adds}, ckpt_path)
        print(f"ckpt saved: {len(store.deltas)}/{len(facts)} ({t:.0f}s)", flush=True)
        return
    if args.eval_only and len(store.deltas) != len(facts):
        raise SystemExit(f"ckpt incomplete: {len(store.deltas)}/{len(facts)}")
    print("base distractor references", flush=True)
    BASE_DIST.update({f["id"]: base_answer(store, f["distractor"]) for f in facts})
    # Pre-revoke references must cover the whole revoke sample (run_revoke's rule:
    # min(100, n//2)); the arm is only reachable when "ours" runs.
    revoke_n = min(100, max(1, len(facts) // 2))
    pre = (
        {f["id"]: base_answer(store, f["probe"]) for f in facts[:revoke_n]}
        if "ours" in arms
        else {}
    )
    if "rag" in arms:
        print("rag: evaluating", flush=True)
        out["results"]["rag"] = run_rag(store, facts, enc, router, words)
    if "rag_instruct" in arms:
        print("rag_instruct: evaluating", flush=True)
        out["results"]["rag_instruct"] = run_rag(
            store, facts, enc, router, words, mode="instruct"
        )
    if "rag_qa" in arms:
        print("rag_qa: evaluating", flush=True)
        out["results"]["rag_qa"] = run_rag(store, facts, enc, router, words, mode="qa")
    if "rag_fewshot" in arms:
        print("rag_fewshot: evaluating", flush=True)
        out["results"]["rag_fewshot"] = run_rag(
            store, facts, enc, router, words, mode="fewshot"
        )
    if "ours" in arms:
        out["results"]["ours"] = run_ours(
            store, facts, enc, router, words, train=not args.eval_only, adds=adds
        )
    if "melo_like" in arms:
        out["results"]["melo_like"] = run_ours(
            store, facts, enc, router, words, capped=False
        )
    if "summed" in arms:
        out["results"]["summed"] = run_summed(store, facts, words)
    if "grace_like" in arms:
        out["results"]["grace_like"] = run_grace(store, facts, enc, router, words)
    if "wise_like" in arms:
        out["results"]["wise_like"] = run_wise(store, facts, enc, router, words)
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
