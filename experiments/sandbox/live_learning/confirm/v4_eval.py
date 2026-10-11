"""SS67/68: re-evaluate a saved store with router v4 (eval-only).

Per fact and query (edit prompt, paraphrase) three arms are served through router v4:
the routed adapter, instructed retrieval with the routed fact's statement, and both
(adapter active and the statement in the system prompt). Locality: synthetic-name
distractors; on CounterFact the neighborhood prompts (E2) and other relations of the
stored subject (E3, the SS65 prompts); on zsRE the dataset's `loc` questions. Removal of
the first 100 facts (gone / return / retention), and on zsRE additionally removal by
content (every entry with the same subject and answer). An abstention is answered by the
base model; base answers are cached per query string because an abstained query is
served by the unmodified base (SS65: identical in 1122/1122 cases).

    .venv/bin/python experiments/sandbox/live_learning/confirm/v4_eval.py \\
      --ckpt results/live_learning/confirm/cf_n1000.ckpt \\
      --facts results/live_learning/confirm/external_counterfact_n1000.json \\
      --out results/live_learning/confirm/cf_n1000_v4_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import NUMBERS  # noqa: E402
from locality_eval import (  # noqa: E402
    match_raw,
    other_relation_prompts,
    paired_diff_ci,
)
from pilot import (  # noqa: E402
    CANARIES,
    SYSTEM,
    PilotStore,
    base_answer,
    hit,
    rag_system,
    trim,
)
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _metric,
    _repo_path,
    _validate_checkpoint,
    load_verified_checkpoint,
    verify_manifest_inputs,
)
from router_v4 import RouterV4  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")


class Server:
    """Serving with a base-answer cache; `system=None` means the plain system prompt."""

    def __init__(self, store):
        self.store = store
        self._base = {}

    def base(self, q, system=None):
        key = (q, system)
        if key not in self._base:
            self._base[key] = base_answer(self.store, q, system=system)
        return self._base[key]

    def adapter(self, fid, q, system=None):
        store = self.store
        full = store.deltas
        store.deltas = {fid: full[fid]}
        store.materialize()
        resp = store.answer(q, system=system or SYSTEM)
        store.deltas = full
        store.materialize()
        return resp


def serve_arms(server, router, facts, q, enc):
    """(route index, similarity, {arm: response}) for one query."""
    j, sim = router.route(q, enc)
    if j is None:
        base = server.base(q)
        return j, sim, {"adapter": base, "retrieval": base, "both": base}
    note = rag_system("instruct", facts, j)
    fid = facts[j]["id"]
    return j, sim, {
        "adapter": server.adapter(fid, q),
        "retrieval": server.base(q, system=note),
        "both": server.adapter(fid, q, system=note),
    }


def locality(server, router, facts, queries, enc, answers):
    rows = []
    for fact, q in zip(facts, queries):
        base = server.base(q)
        j, sim = router.route(q, enc)
        resp = base if j is None else server.adapter(facts[j]["id"], q)
        rows.append(
            {
                "fact_id": fact["id"],
                "query": q,
                "route_fact_id": None if j is None else facts[j]["id"],
                "route_score": round(sim, 6),
                "routed": j is not None,
                "same_as_base": resp.strip() == base.strip(),
                "leak": any(hit(resp, w) and not hit(base, w) for w in answers),
                "response": resp,
                "base_response": base,
            }
        )
        trim()
    return {
        "metrics": {
            "routed": _metric([r["routed"] for r in rows]),
            "same_as_base": _metric([r["same_as_base"] for r in rows]),
            "leak": _metric([r["leak"] for r in rows]),
        },
        "examples": rows,
    }


def removal(server, enc, facts, removed, tau, make_router=None):
    keep_ids = {f["id"] for f in facts} - {f["id"] for f in removed}
    keep = [f for f in facts if f["id"] in keep_ids]
    full = server.store.deltas
    server.store.deltas = {fid: full[fid] for fid in keep_ids}
    router = make_router(keep) if make_router else RouterV4(enc, keep, tau)
    rows = []
    for f in removed:
        pre = server.base(f["probe"])
        j, _ = router.route(f["probe"], enc)
        resp = pre if j is None else server.adapter(keep[j]["id"], f["probe"])
        rows.append(
            {
                "fact_id": f["id"],
                "routed_to": None if j is None else keep[j]["id"],
                "gone": not hit(resp, f["answer"]),
                "return": hit(resp, f["answer"]) == hit(pre, f["answer"]),
                "base_hit": hit(pre, f["answer"]),
                "response": resp,
            }
        )
    server.store.deltas = full
    return rows, keep, router


def evaluate(store, facts, enc, tau, dataset, raw, progress=100, make_router=None,
             run_locality=True):
    server = Server(store)
    router = make_router(facts) if make_router else RouterV4(enc, facts, tau)
    answers = [f["answer"] for f in facts if f["answer"] not in NUMBERS]

    rows = []
    for i, f in enumerate(facts):
        row = {"fact_id": f["id"], "answer": f["answer"]}
        for kind in ("probe", "paraphrase"):
            j, sim, arms = serve_arms(server, router, facts, f[kind], enc)
            row[kind] = {
                "route_fact_id": None if j is None else facts[j]["id"],
                "route_correct": j == i,
                "route_score": round(sim, 6),
                **{f"{arm}_hit": hit(resp, f["answer"]) for arm, resp in arms.items()},
                "responses": arms,
            }
        base = server.base(f["distractor"])
        j, _ = router.route(f["distractor"], enc)
        resp = base if j is None else server.adapter(facts[j]["id"], f["distractor"])
        row["distractor"] = {
            "routed": j is not None,
            "leak": any(hit(resp, w) and not hit(base, w) for w in answers),
        }
        rows.append(row)
        trim()
        if progress and (i + 1) % progress == 0:
            print(f"serving: {i + 1}/{len(facts)}", flush=True)

    def col(kind, key):
        return [r[kind][key] for r in rows]

    serving = {
        kind: {
            "route_correct": _metric(col(kind, "route_correct")),
            "routed": _metric([r[kind]["route_fact_id"] is not None for r in rows]),
            **{arm: _metric(col(kind, f"{arm}_hit")) for arm in ("adapter", "retrieval", "both")},
            "adapter_minus_retrieval": paired_diff_ci(
                col(kind, "adapter_hit"), col(kind, "retrieval_hit")
            ),
            "adapter_minus_retrieval_route_correct": paired_diff_ci(
                [r[kind]["adapter_hit"] for r in rows if r[kind]["route_correct"]],
                [r[kind]["retrieval_hit"] for r in rows if r[kind]["route_correct"]],
            ),
        }
        for kind in ("probe", "paraphrase")
    }
    serving["distractor"] = {
        "routed": _metric([r["distractor"]["routed"] for r in rows]),
        "no_leak": _metric([not r["distractor"]["leak"] for r in rows]),
    }

    # SS68 tiered store: a note if retrieval answers the edit prompt, else an adapter
    tier = ["note" if r["probe"]["retrieval_hit"] else "adapter" for r in rows]
    tiered = {
        "adapter_fraction": round(tier.count("adapter") / len(tier), 4),
        "probe": _metric(
            [r["probe"][f"{'retrieval' if t == 'note' else 'adapter'}_hit"] for r, t in zip(rows, tier)]
        ),
        "paraphrase": _metric(
            [r["paraphrase"][f"{'retrieval' if t == 'note' else 'adapter'}_hit"] for r, t in zip(rows, tier)]
        ),
    }
    for r, t in zip(rows, tier):
        r["tier"] = t

    loc = {}
    if not run_locality:
        pass
    elif dataset == "counterfact":
        records = match_raw(facts, raw)
        loc["E2_neighborhood"] = locality(
            server, router, facts, [r["neighborhood_prompts"][0] for r in records], enc, answers
        )
        print("E2 done", flush=True)
        loc["E3_other_relation"] = locality(
            server, router, facts, [p["query"] for p in other_relation_prompts(records, raw)],
            enc, answers,
        )
        print("E3 done", flush=True)
    else:
        by_src = {item["src"].strip(): item for item in raw}
        loc["loc_questions"] = locality(
            server, router, facts, [by_src[f["probe"]]["loc"] for f in facts], enc, answers
        )
        print("loc done", flush=True)

    removed = facts[:100]
    t0 = time.time()
    rem_rows, keep, keep_router = removal(server, enc, facts, removed, tau, make_router)
    retention = []
    full = store.deltas
    store.deltas = {f["id"]: full[f["id"]] for f in keep}
    for f in keep:
        j, _ = keep_router.route(f["probe"], enc)
        resp = server.base(f["probe"]) if j is None else server.adapter(keep[j]["id"], f["probe"])
        retention.append(hit(resp, f["answer"]))
    store.deltas = full
    removal_report = {
        "gone": _metric([r["gone"] for r in rem_rows]),
        "return": _metric([r["return"] for r in rem_rows]),
        "base_hit": _metric([r["base_hit"] for r in rem_rows]),
        "retention": _metric(retention),
        "seconds": round(time.time() - t0, 1),
        "examples": rem_rows,
    }
    if dataset == "zsre":
        content = {(f["subject"], f["answer"]) for f in removed}
        dupes = [f for f in facts[100:] if (f["subject"], f["answer"]) in content]
        rows_c, _, _ = removal(server, enc, facts, removed + dupes, tau, make_router)
        rows_c = rows_c[: len(removed)]
        removal_report["by_content"] = {
            "extra_entries_removed": len(dupes),
            "gone": _metric([r["gone"] for r in rows_c]),
            "return": _metric([r["return"] for r in rows_c]),
        }

    return {
        "serving": serving,
        "tiered": tiered,
        "locality": {k: v["metrics"] for k, v in loc.items()},
        "removal": {k: v for k, v in removal_report.items() if k != "examples"},
        "examples": {
            "serving": rows,
            "locality": {k: v["examples"] for k, v in loc.items()},
            "removal": removal_report["examples"],
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default=str(CONFIRM / "router_v4_calibration.json"))
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists():
        ap.error(f"refusing to overwrite existing output: {out_path}")
    tau = json.loads(Path(args.calibration).read_text())["calibration"]["tau"]

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest, _repo_path(args.ckpt, REPO_ROOT), _repo_path(args.facts, REPO_ROOT),
        model_id, repo_root=REPO_ROOT,
    )
    facts, dataset = verified["facts"], verified["artifact"]["dataset"]
    raw = json.loads((CONFIRM / f"raw/{dataset}.json").read_text())

    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    enc = SentenceTransformer(manifest["models"]["router_encoder"]["id"])
    store = PilotStore(model_id, steps=16)
    store.kl_prompts = list(CANARIES)
    checkpoint, ckpt_sha = load_verified_checkpoint(
        verified["checkpoint_path"], verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    _validate_checkpoint(checkpoint, facts)
    store.deltas = checkpoint["deltas"]
    store.keys = checkpoint["keys"]
    store.materialize()
    print(f"loaded {len(store.deltas)} adapters; router v4 tau={tau}", flush=True)

    started = time.time()
    result = evaluate(store, facts, enc, tau, dataset, raw, progress=args.progress_every)
    result.update(
        {
            "schema": "confirm.v4_eval.v1",
            "prereg": "SS67/SS68",
            "dataset": dataset,
            "tau_v4": tau,
            "training_performed": False,
            "checkpoint_sha256": ckpt_sha,
            "facts_sha256": verified["facts_sha256"],
            "wall_seconds": round(time.time() - started, 1),
        }
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: result[k] for k in ("serving", "tiered", "locality", "removal")}, indent=1))
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
