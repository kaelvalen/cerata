"""Cross-session E+Delta hybrid on the event stream (pinned in the prereg).

Session 1 of stream.build_events drives the hybrid: the memory tier (notes + MiniLM
retrieval, tau 0.5) serves every probe; the adopted v1.2 repair policy promotes a
fact-tagged probe after an observed miss (UCB1, same constants); update events retrain
a promoted fact (revoke + propose_and_commit); unlearn drops the note and revokes any
expert. Session 2 replays probes only, controller online.

    python ledger_session.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger import DeltaStore  # noqa: E402
from ledger_bandit import (  # noqa: E402
    COST,
    KLD_LIMIT,
    expert_serve,
    hit,
)
from stream import FACTS, UPDATES, build_events  # noqa: E402

ALPHA = 0.5


class Notes:
    """The memory tier: per-fact note + key, retrieval with tau."""

    def __init__(self, enc, tau: float = 0.5):
        self.enc = enc
        self.tau = tau
        self.note = {f.id: f.teach for f in FACTS}
        self.key = {f.id: f.probe for f in FACTS}

    def drop(self, fid: str) -> None:
        self.note.pop(fid, None)
        self.key.pop(fid, None)

    def set_note(self, fid: str, text: str) -> None:
        self.note[fid] = text

    def retrieve(self, q: str):
        if not self.key:
            return None, 0.0
        fids = list(self.key)
        K = self.enc.encode([self.key[f] for f in fids], normalize_embeddings=True)
        v = self.enc.encode([q], normalize_embeddings=True)[0]
        sims = K @ v
        j = int(sims.argmax())
        return fids[j], float(sims[j])

    def serve(self, store: DeltaStore, q: str):
        fid, sim = self.retrieve(q)
        full = store.deltas
        store.deltas = {}
        store.materialize()
        if fid is not None and sim >= self.tau:
            system = f"Kısa ve net cevap ver. İlgili bağlam: {self.note[fid]}"
        else:
            system = "Kısa ve net cevap ver."
        resp = store.answer(q, system=system)
        store.deltas = full
        store.materialize()
        return resp, fid, sim


def outcome(e, resp: str):
    if e.expect:
        return hit(resp, e.expect)
    if e.forbid:
        return e.forbid.lower() not in resp.lower()
    return None


def run(s1, s2, store: DeltaStore, notes: Notes, args) -> dict:
    counts: dict[str, int] = {f.id: 0 for f in FACTS}
    answers: dict[str, str] = {f.id: f.answer for f in FACTS}
    promoted: dict[str, float] = {}
    refused: dict[str, float] = {}
    q: dict[tuple, float] = {}
    n: dict[tuple, int] = {}
    state = {"total": 0, "cost": 0.0}
    log: list[dict] = []
    events_log: list[dict] = []

    def probe_event(e) -> None:
        fid = e.fact.id if e.fact else None
        if fid:
            counts[fid] += 1
        if fid and fid in promoted:
            resp = expert_serve(store, fid, e.text)
            log.append(
                {
                    "note": e.note,
                    "fid": fid,
                    "path": "expert",
                    "ok": hit(resp, answers[fid]),
                    "expect": answers[fid],
                    "retrieved": fid,
                    "sim": 1.0,
                    "resp": resp,
                }
            )
            return
        resp, ret_fid, sim = notes.serve(store, e.text)
        ok = outcome(e, resp) if fid is None else hit(resp, answers[fid])
        path = "memory"
        if fid and fid not in refused and ok is False:
            ctx = (min(counts[fid], 2),)
            state["total"] += 1
            total = state["total"]
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
            reward = 0.0
            if action == "promote":
                f = e.fact
                note = notes.note[fid]
                res = store.propose_and_commit(
                    fid,
                    [(note, f"Not aldım: {note}"), (f.probe, answers[fid])],
                    key=f.probe,
                    kld_limit=KLD_LIMIT,
                )
                if res["committed"]:
                    promoted[fid] = res["kld"]
                    resp = expert_serve(store, fid, e.text)
                    ok = outcome(e, resp)
                    reward = (1.0 if ok else 0.0) - COST
                    state["cost"] += COST
                    path = "expert"
                else:
                    refused[fid] = res["kld"]
            q[(ctx, action)] = q.get((ctx, action), 0.0) + ALPHA * (
                reward - q.get((ctx, action), 0.0)
            )
            events_log.append(
                {"event": "repair", "fid": fid, "ctx": list(ctx), "action": action,
                 "explore": explore, "reward": round(reward, 3),
                 "committed": path == "expert"}
            )
        log.append(
            {
                "note": e.note,
                "fid": fid,
                "path": path,
                "ok": ok,
                "expect": answers[fid] if fid else (e.expect or e.forbid),
                "retrieved": ret_fid,
                "sim": round(sim, 3),
                "resp": resp,
            }
        )

    def update_event(e) -> None:
        fid = e.fact.id
        _, text, probe, expect = next(u for u in UPDATES if u[0] == fid)
        notes.set_note(fid, text)
        answers[fid] = expect
        info = {"event": "update", "fid": fid, "was_promoted": fid in promoted,
                "retrained": False, "kld": None}
        if fid in promoted:
            store.revoke(fid)
            del promoted[fid]
            res = store.propose_and_commit(
                fid, [(text, f"Not aldım: {text}"), (probe, expect)],
                key=probe, kld_limit=KLD_LIMIT,
            )
            info["kld"] = round(res["kld"], 3)
            if res["committed"]:
                promoted[fid] = res["kld"]
                state["cost"] += COST
                info["retrained"] = True
            else:
                refused[fid] = res["kld"]
        events_log.append(info)

    def unlearn_event(e) -> None:
        fid = e.fact.id
        notes.drop(fid)
        answers.pop(fid, None)
        revoked = False
        if fid in promoted:
            store.revoke(fid)
            del promoted[fid]
            revoked = True
        refused.pop(fid, None)
        events_log.append({"event": "unlearn", "fid": fid, "revoked": revoked})

    for e in s1:
        if e.kind == "probe":
            probe_event(e)
        elif e.kind == "update":
            update_event(e)
        elif e.kind == "unlearn":
            unlearn_event(e)
    s1_log = list(log)

    for e in s2:
        if e.kind == "probe":
            probe_event(e)
    s2_log = log[len(s1_log):]

    def agg(entries) -> dict:
        by: dict[str, list] = {}
        for r in entries:
            if r["ok"] is None:
                continue
            by.setdefault(r["note"], [0, 0])
            by[r["note"]][0] += 1
            by[r["note"]][1] += int(bool(r["ok"]))
        return {k: f"{v[1]}/{v[0]}" for k, v in sorted(by.items())}

    imm = {r["fid"]: r["ok"] for r in s1_log if r["note"] == "s1-imm" and r["fid"]}
    s2_by_fid = {r["fid"]: r["ok"] for r in s2_log if r["note"] == "s2" and r["fid"]}
    retention = {fid: [imm.get(fid), s2_by_fid.get(fid)] for fid in sorted(s2_by_fid)}

    return {
        "model": args.model,
        "tau": args.tau,
        "promoted": sorted(promoted),
        "refused": {k: round(v, 3) for k, v in refused.items()},
        "experts": len(promoted),
        "events": events_log,
        "s1": agg(s1_log),
        "s2": agg(s2_log),
        "retention": retention,
        "cost_paid": round(state["cost"], 3),
        "q_table": {f"c{c[0]}|{a}": round(v, 3) for (c, a), v in sorted(q.items())},
        "n_table": {f"c{c[0]}|{a}": v for (c, a), v in sorted(n.items())},
        "s1_log": s1_log,
        "s2_log": s2_log,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/live_learning/ledger_session_15b.json")
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    s1, s2 = build_events(seed=0)
    notes = Notes(enc, args.tau)
    store = DeltaStore(args.model, steps=16)
    res = run(s1, s2, store, notes, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    upd = next((r for r in res["events"] if r["event"] == "update"), None)
    unl = next((r for r in res["events"] if r["event"] == "unlearn"), None)
    print(
        f"experts {res['experts']} {res['promoted']} | "
        f"s1 {res['s1']} | s2 {res['s2']} | update {upd} | unlearn {unl} | "
        f"cost {res['cost_paid']}"
    )


if __name__ == "__main__":
    main()
