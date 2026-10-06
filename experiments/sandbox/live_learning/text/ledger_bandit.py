"""Controller v1 (learned): epsilon-greedy bandit over {promote, defer} (pinned).

Two arms on the same stream as controller v0 (6 facts x3 queries, 12 facts x1):
  rule   - promote on the second query, deferred facts served by the memory tier;
  bandit - epsilon-greedy over {promote, defer}, context (last outcome, min(count,2)),
           immediate reward = token hit minus the one-time promotion cost 0.3.

The served query is the probe; the memory tier is MiniLM retrieval + note injection
(served with all deltas zeroed). Key == probe by construction, so retrieval isolates
the generation tier. Readings: experts, refusals, final readout recall over all 18
facts, and the Q table.

    python ledger_bandit.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from ledger import DeltaStore  # noqa: E402
from stream import FACTS  # noqa: E402

RECURRING = 6
HITS = 3
THRESHOLD = 2
COST = 0.3
EPS = 0.2
ALPHA = 0.5
SEED = 0
KLD_LIMIT = 2.0


def stream_ids() -> list[str]:
    ids: list[str] = []
    for i, f in enumerate(FACTS):
        ids += [f.id] * (HITS if i < RECURRING else 1)
    return ids


def hit(resp: str, expect: str) -> bool:
    return expect.lower() in resp.lower()


def train_pairs(f) -> list[tuple[str, str]]:
    return [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)]


class Retriever:
    """E-tier retrieval: MiniLM over the fact keys, note = the teach text."""

    def __init__(self, enc):
        self.enc = enc
        self.ids = [f.id for f in FACTS]
        self.notes = {f.id: f.teach for f in FACTS}
        self.K = enc.encode([f.probe for f in FACTS], normalize_embeddings=True)

    def note_for(self, q: str) -> str:
        v = self.enc.encode([q], normalize_embeddings=True)[0]
        return self.notes[self.ids[int((self.K @ v).argmax())]]


def memory_serve(store: DeltaStore, note: str, q: str) -> str:
    full = store.deltas
    store.deltas = {}
    store.materialize()
    resp = store.answer(q, system=f"Kısa ve net cevap ver. İlgili bağlam: {note}")
    store.deltas = full
    store.materialize()
    return resp


def expert_serve(store: DeltaStore, fid: str, q: str) -> str:
    full = store.deltas
    store.deltas = {fid: full[fid]}
    store.materialize()
    resp = store.answer(q)
    store.deltas = full
    store.materialize()
    return resp


def final_readout(store: DeltaStore, ret: Retriever, promoted: set[str]) -> dict:
    final = {}
    for f in FACTS:
        if f.id in promoted:
            resp, path = expert_serve(store, f.id, f.probe), "expert"
        else:
            resp, path = memory_serve(store, ret.note_for(f.probe), f.probe), "memory"
        final[f.id] = {"path": path, "ok": hit(resp, f.answer), "resp": resp}
    return final


def run_rule(store: DeltaStore, ret: Retriever) -> dict:
    facts = {f.id: f for f in FACTS}
    counts = {fid: 0 for fid in facts}
    promoted: dict[str, float] = {}
    refused: dict[str, float] = {}
    trace = []
    for fid in stream_ids():
        f = facts[fid]
        counts[fid] += 1
        if counts[fid] == THRESHOLD and fid not in promoted and fid not in refused:
            res = store.propose_and_commit(
                fid, train_pairs(f), key=f.probe, kld_limit=KLD_LIMIT
            )
            (promoted if res["committed"] else refused)[fid] = res["kld"]
        if fid in promoted:
            resp, path = expert_serve(store, fid, f.probe), "expert"
        else:
            resp, path = memory_serve(store, ret.note_for(f.probe), f.probe), "memory"
        trace.append(
            {"fid": fid, "count": counts[fid], "path": path, "ok": hit(resp, f.answer)}
        )
    final = final_readout(store, ret, set(promoted))
    return {
        "policy": "rule",
        "promoted": sorted(promoted),
        "refused": {k: round(v, 3) for k, v in refused.items()},
        "experts": len(promoted),
        "final_readout": final,
        "final_recall": sum(v["ok"] for v in final.values()),
        "trace": trace,
    }


def run_learned(store: DeltaStore, ret: Retriever, mode: str = "eps") -> dict:
    facts = {f.id: f for f in FACTS}
    counts = {fid: 0 for fid in facts}
    promoted: dict[str, float] = {}
    refused: dict[str, float] = {}
    last: dict[str, str] = {}
    q: dict[tuple, float] = {}
    n: dict[tuple, int] = {}
    total = 0
    rng = random.Random(SEED)
    trace = []
    for fid in stream_ids():
        f = facts[fid]
        counts[fid] += 1
        if fid in promoted:
            resp = expert_serve(store, fid, f.probe)
            ok = hit(resp, f.answer)
            last[fid] = "ok" if ok else "fail"
            trace.append(
                {"fid": fid, "count": counts[fid], "path": "expert", "ok": ok}
            )
            continue
        if fid in refused:  # locked to memory after a KLD refusal
            resp = memory_serve(store, ret.note_for(f.probe), f.probe)
            ok = hit(resp, f.answer)
            last[fid] = "ok" if ok else "fail"
            trace.append(
                {
                    "fid": fid,
                    "count": counts[fid],
                    "path": "memory",
                    "ok": ok,
                    "locked_refused": True,
                }
            )
            continue
        ctx = (last.get(fid, "none"), min(counts[fid], 2))
        total += 1
        if mode == "eps":
            if rng.random() < EPS:
                action = rng.choice(["promote", "defer"])
                explore = True
            else:
                action = (
                    "promote"
                    if q.get((ctx, "promote"), 0.0) > q.get((ctx, "defer"), 0.0)
                    else "defer"
                )
                explore = False
        else:  # UCB1: unsampled action first, then the bonus term
            action, best, explore = "defer", -1.0, False
            for a in ("defer", "promote"):
                na = n.get((ctx, a), 0)
                score = (
                    float("inf")
                    if na == 0
                    else q.get((ctx, a), 0.0) + math.sqrt(math.log(total + 1) / na)
                )
                if score > best:
                    best, action, explore = score, a, na == 0
        n[(ctx, action)] = n.get((ctx, action), 0) + 1
        if action == "promote":
            res = store.propose_and_commit(
                fid, train_pairs(f), key=f.probe, kld_limit=KLD_LIMIT
            )
            if res["committed"]:
                promoted[fid] = res["kld"]
                resp = expert_serve(store, fid, f.probe)
                ok = hit(resp, f.answer)
                reward = (1.0 if ok else 0.0) - COST
                path = "expert"
            else:
                refused[fid] = res["kld"]
                resp = memory_serve(store, ret.note_for(f.probe), f.probe)
                ok = hit(resp, f.answer)
                reward = 1.0 if ok else 0.0
                path = "memory"
        else:
            resp = memory_serve(store, ret.note_for(f.probe), f.probe)
            ok = hit(resp, f.answer)
            reward = 1.0 if ok else 0.0
            path = "memory"
        q[(ctx, action)] = q.get((ctx, action), 0.0) + ALPHA * (
            reward - q.get((ctx, action), 0.0)
        )
        last[fid] = "ok" if ok else "fail"
        trace.append(
            {
                "fid": fid,
                "count": counts[fid],
                "ctx": list(ctx),
                "action": action,
                "explore": explore,
                "committed": path == "expert",
                "path": path,
                "ok": ok,
                "reward": round(reward, 3),
            }
        )
    final = final_readout(store, ret, set(promoted))
    return {
        "policy": "bandit" if mode == "eps" else mode,
        "promoted": sorted(promoted),
        "refused": {k: round(v, 3) for k, v in refused.items()},
        "experts": len(promoted),
        "commits_explore": sum(
            1 for t in trace if t.get("explore") and t.get("committed")
        ),
        "q_table": {
            f"{ctx[0]}|c{ctx[1]}|{a}": round(v, 3) for (ctx, a), v in sorted(q.items())
        },
        "n_table": {
            f"{ctx[0]}|c{ctx[1]}|{a}": v for (ctx, a), v in sorted(n.items())
        },
        "final_readout": final,
        "final_recall": sum(v["ok"] for v in final.values()),
        "trace": trace,
    }


def run_repair(store: DeltaStore, ret: Retriever) -> dict:
    """Serve-first policy: memory serves; a miss is repaired ({promote, defer}, UCB1)."""
    facts = {f.id: f for f in FACTS}
    counts = {fid: 0 for fid in facts}
    promoted: dict[str, float] = {}
    refused: dict[str, float] = {}
    q: dict[tuple, float] = {}
    n: dict[tuple, int] = {}
    total = 0
    trace = []
    for fid in stream_ids():
        f = facts[fid]
        counts[fid] += 1
        if fid in promoted:
            resp = expert_serve(store, fid, f.probe)
            trace.append(
                {
                    "fid": fid,
                    "count": counts[fid],
                    "path": "expert",
                    "decision": False,
                    "ok": hit(resp, f.answer),
                }
            )
            continue
        if fid in refused:  # locked to memory after a KLD refusal
            resp = memory_serve(store, ret.note_for(f.probe), f.probe)
            trace.append(
                {
                    "fid": fid,
                    "count": counts[fid],
                    "path": "memory",
                    "decision": False,
                    "locked_refused": True,
                    "ok": hit(resp, f.answer),
                }
            )
            continue
        resp = memory_serve(store, ret.note_for(f.probe), f.probe)
        if hit(resp, f.answer):  # hit: no decision needed
            trace.append(
                {
                    "fid": fid,
                    "count": counts[fid],
                    "path": "memory",
                    "decision": False,
                    "ok": True,
                }
            )
            continue
        # miss -> repair decision (UCB1)
        ctx = (min(counts[fid], 2),)
        total += 1
        action, best, explore = "defer", -1.0, False
        for a in ("defer", "promote"):
            na = n.get((ctx, a), 0)
            score = (
                float("inf")
                if na == 0
                else q.get((ctx, a), 0.0) + math.sqrt(math.log(total + 1) / na)
            )
            if score > best:
                best, action, explore = score, a, na == 0
        n[(ctx, action)] = n.get((ctx, action), 0) + 1
        if action == "promote":
            res = store.propose_and_commit(
                fid, train_pairs(f), key=f.probe, kld_limit=KLD_LIMIT
            )
            if res["committed"]:
                promoted[fid] = res["kld"]
                ok_out = hit(expert_serve(store, fid, f.probe), f.answer)
                reward = (1.0 if ok_out else 0.0) - COST
                path = "expert"
            else:
                refused[fid] = res["kld"]
                ok_out, reward, path = False, 0.0, "memory"
        else:
            ok_out, reward, path = False, 0.0, "memory"
        q[(ctx, action)] = q.get((ctx, action), 0.0) + ALPHA * (
            reward - q.get((ctx, action), 0.0)
        )
        trace.append(
            {
                "fid": fid,
                "count": counts[fid],
                "ctx": list(ctx),
                "action": action,
                "explore": explore,
                "committed": path == "expert",
                "path": path,
                "ok": ok_out,
                "reward": round(reward, 3),
                "repair": True,
            }
        )
    final = final_readout(store, ret, set(promoted))
    return {
        "policy": "repair",
        "promoted": sorted(promoted),
        "refused": {k: round(v, 3) for k, v in refused.items()},
        "experts": len(promoted),
        "commits_explore": sum(
            1 for t in trace if t.get("explore") and t.get("committed")
        ),
        "q_table": {
            f"c{ctx[0]}|{a}": round(v, 3) for (ctx, a), v in sorted(q.items())
        },
        "n_table": {f"c{ctx[0]}|{a}": v for (ctx, a), v in sorted(n.items())},
        "final_readout": final,
        "final_recall": sum(v["ok"] for v in final.values()),
        "trace": trace,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_bandit_15b.json")
    ap.add_argument("--arms", default="rule,bandit", help="rule, bandit (eps), ucb")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ret = Retriever(enc)
    out: dict = {
        "model": args.model,
        "eps": EPS,
        "alpha": ALPHA,
        "cost": COST,
        "arms": {},
    }
    for arm in args.arms.split(","):
        store = DeltaStore(args.model, steps=16)
        if arm == "rule":
            res = run_rule(store, ret)
        elif arm == "repair":
            res = run_repair(store, ret)
        else:
            res = run_learned(store, ret, mode="eps" if arm == "bandit" else "ucb")
        out["arms"][arm] = res
        del store
        torch.cuda.empty_cache()
        print(
            f"{arm}: experts {res['experts']} | recall {res['final_recall']}/18 | "
            f"promoted {res['promoted']} | refused {sorted(res['refused'])}",
            flush=True,
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    last_arm = [a for a in out["arms"]][-1]
    b = out["arms"][last_arm]
    print(
        f"{last_arm} first-sample commits {b['commits_explore']} | "
        f"Q {json.dumps(b['q_table'])}"
    )


if __name__ == "__main__":
    main()
