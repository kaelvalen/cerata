"""Run one live-learning mechanism over the general stream (sandbox).

    export LD_LIBRARY_PATH=/nix/store/38v10xhwhypb747h3z4c2i0a19hkiwx2-nvidia-x11-615.71.09/lib
    .venv/bin/python experiments/sandbox/live_learning/run.py --mechanism context
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
import stream as stream_mod  # noqa: E402
from mechanisms import MECHANISMS  # noqa: E402


def score(ev, resp: str):
    if ev.expect is not None:
        return ev.expect.lower() in resp.lower()
    if ev.forbid is not None:
        return ev.forbid.lower() not in resp.lower()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", choices=tuple(MECHANISMS), default="context")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sessions", default="1,2")
    ap.add_argument("--keep-state", action="store_true")
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--lora-steps", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    tok, base = harness.load_model(args.model)
    model_tag = args.model.rstrip("/").split("/")[-1].lower()
    state_dir = (
        Path("results/live_learning/state")
        / f"{args.mechanism}_{model_tag}_seed{args.seed}"
    )
    if not args.keep_state and state_dir.exists():
        shutil.rmtree(state_dir)
    mech_kwargs = (
        {"steps": args.lora_steps, "lr": args.lora_lr}
        if args.mechanism == "lora"
        else {}
    )
    mech = MECHANISMS[args.mechanism](
        tok, base, state_dir, args.model, args.seed, **mech_kwargs
    )

    s1, s2 = stream_mod.build_events(args.seed)
    if args.limit:
        s1, s2 = s1[: args.limit], s2[: args.limit]
    sessions = [int(s) for s in args.sessions.split(",")]
    records = []
    for session, events in ((1, s1), (2, s2)):
        if session not in sessions:
            continue
        for ev in events:
            t = time.time()
            if ev.kind in ("teach", "update"):
                mech.teach(
                    ev.fact,
                    ev.text,
                    answer=ev.expect if ev.kind == "update" else None,
                )
                records.append(
                    {
                        "kind": ev.kind,
                        "session": session,
                        "note": ev.note,
                        "fact": getattr(ev.fact, "id", None),
                        "seconds": round(time.time() - t, 3),
                    }
                )
            elif ev.kind == "unlearn":
                mech.unlearn(ev.fact)
                records.append(
                    {
                        "kind": "unlearn",
                        "session": session,
                        "note": ev.note,
                        "fact": getattr(ev.fact, "id", None),
                        "seconds": round(time.time() - t, 3),
                    }
                )
            elif ev.kind == "filler":
                mech.filler(ev.text)
            elif ev.kind == "probe":
                resp = mech.answer(ev.text)
                records.append(
                    {
                        "kind": "probe",
                        "session": session,
                        "note": ev.note,
                        "q": ev.text,
                        "resp": resp,
                        "ok": score(ev, resp),
                        "expect": ev.expect,
                        "forbid": ev.forbid,
                        "seconds": round(time.time() - t, 3),
                    }
                )
        mech.persist()

    def bucket(pred):
        hits = [
            r["ok"]
            for r in records
            if r["kind"] == "probe" and pred(r) and r["ok"] is not None
        ]
        return {
            "n": len(hits),
            "acc": round(sum(hits) / len(hits), 3) if hits else None,
        }

    teach_times = [
        r["seconds"] for r in records if r["kind"] in ("teach", "update", "unlearn")
    ]
    answer_times = [r["seconds"] for r in records if r["kind"] == "probe"]
    result = {
        "mechanism": args.mechanism,
        "model": args.model,
        "seed": args.seed,
        "seconds_total": round(time.time() - t0, 1),
        "summary": {
            "immediate": bucket(lambda r: r["note"] == "s1-imm"),
            "delayed": bucket(lambda r: r["note"] == "s1-del"),
            "update": bucket(lambda r: r["note"] == "s1-upd"),
            "interference": bucket(lambda r: r["note"] == "s1-intf"),
            "unlearn": bucket(lambda r: r["note"] == "s1-unlearn"),
            "neighbour": bucket(lambda r: r["note"] == "s1-unlearn-neigh"),
            "session2": bucket(lambda r: r["note"] == "s2"),
            "capability": bucket(lambda r: "cap" in r["note"]),
            "teach_seconds_mean": (
                round(sum(teach_times) / len(teach_times), 3) if teach_times else None
            ),
            "answer_seconds_mean": (
                round(sum(answer_times) / len(answer_times), 3)
                if answer_times
                else None
            ),
        },
        "records": records,
    }
    out = Path(
        args.out
        or f"results/live_learning/{args.mechanism}_{model_tag}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    s = result["summary"]
    print(
        f"[{args.mechanism}] imm {s['immediate']['acc']} del {s['delayed']['acc']} "
        f"upd {s['update']['acc']} intf {s['interference']['acc']} "
        f"unlearn {s['unlearn']['acc']} neigh {s['neighbour']['acc']} "
        f"s2 {s['session2']['acc']} cap {s['capability']['acc']} | "
        f"teach {s['teach_seconds_mean']}s ans {s['answer_seconds_mean']}s "
        f"| total {result['seconds_total']}s"
    )


if __name__ == "__main__":
    main()
