"""SS71: locality of the serving arms (eval-only).

(a) CounterFact n=1000, existing adapters, router v4: E3 (other relation of the stored
subject) and E2 (neighborhood) prompts served by four policies: adapter, note
(instructed retrieval), both, and the SS68 tiered store. (b) The 200 SS70 local-anchor
adapters served together with their note, next to the existing adapters with their note
on the same facts.

    .venv/bin/python experiments/sandbox/live_learning/confirm/serving_locality.py
"""

from __future__ import annotations

import argparse
import hashlib
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
from pilot import CANARIES, PilotStore, hit, rag_system, trim  # noqa: E402
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
from v4_eval import Server  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")
POLICIES = ("adapter", "note", "both", "tiered")


def serve(server, facts, j, q, tiers):
    """{policy: response} for a query routed to index j (None = abstain)."""
    base = server.base(q)
    if j is None:
        return {p: base for p in POLICIES}
    fid = facts[j]["id"]
    note = rag_system("instruct", facts, j)
    out = {
        "adapter": server.adapter(fid, q),
        "note": server.base(q, system=note),
        "both": server.adapter(fid, q, system=note),
    }
    out["tiered"] = out["note"] if tiers[fid] == "note" else out["adapter"]
    return out


def locality(server, router, facts, queries, enc, answers, tiers):
    rows = []
    for fact, q in zip(facts, queries):
        base = server.base(q)
        j, _ = router.route(q, enc)
        resp = serve(server, facts, j, q, tiers)
        row = {"fact_id": fact["id"], "query": q, "routed": j is not None,
               "route_fact_id": None if j is None else facts[j]["id"], "base": base}
        for p in POLICIES:
            row[p] = {
                "same": resp[p].strip() == base.strip(),
                "leak": any(hit(resp[p], w) and not hit(base, w) for w in answers),
                "response": resp[p],
            }
        rows.append(row)
        trim()
    metrics = {"routed": _metric([r["routed"] for r in rows])}
    for p in POLICIES:
        metrics[p] = {k: _metric([r[p][k] for r in rows]) for k in ("same", "leak")}
    return metrics, rows


def with_note(server, deltas, facts, enc, router, e3, answers):
    """Probe, paraphrase and E3 leak with the routed adapter and its note (SS70 facts)."""
    full = server.store.deltas
    server.store.deltas = deltas
    rows = []
    for i, f in enumerate(facts):
        row = {"fact_id": f["id"]}
        for kind, q in (("probe", f["probe"]), ("paraphrase", f["paraphrase"]), ("E3", e3[i])):
            j, _ = router.route(q, enc)
            base = server.base(q)
            if j is None:
                resp = base
            else:
                resp = server.adapter(facts[j]["id"], q, system=rag_system("instruct", facts, j))
            if kind == "E3":
                row["E3_leak"] = any(hit(resp, w) and not hit(base, w) for w in answers)
            else:
                row[kind] = hit(resp, f["answer"])
        rows.append(row)
        trim()
    server.store.deltas = full
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "serving_locality.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")
    tau = json.loads((CONFIRM / "router_v4_calibration.json").read_text())["calibration"]["tau"]
    v4 = json.loads((CONFIRM / "cf_n1000_v4_eval.json").read_text())
    tiers = {r["fact_id"]: r["tier"] for r in v4["examples"]["serving"]}

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest, _repo_path(CONFIRM / "cf_n1000.ckpt", REPO_ROOT),
        _repo_path(CONFIRM / "external_counterfact_n1000.json", REPO_ROOT), model_id,
        repo_root=REPO_ROOT,
    )
    facts = verified["facts"]
    raw = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    records = match_raw(facts, raw)
    e3 = [p["query"] for p in other_relation_prompts(records, raw)]
    e2 = [r["neighborhood_prompts"][0] for r in records]
    answers = [f["answer"] for f in facts if f["answer"] not in NUMBERS]

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
    server = Server(store)
    router = RouterV4(enc, facts, tau)
    started = time.time()

    report = {"prereg": "SS71", "tau_v4": tau, "checkpoint_sha256": ckpt_sha,
              "adapter_fraction": round(sum(t == "adapter" for t in tiers.values()) / len(tiers), 4)}
    examples = {}
    for name, queries in (("E3_other_relation", e3), ("E2_neighborhood", e2)):
        report[name], examples[name] = locality(server, router, facts, queries, enc, answers, tiers)
        print(name, json.dumps({p: report[name][p]["leak"]["rate"] for p in POLICIES}), flush=True)

    local_path = CONFIRM / "cf_n200_local.ckpt"
    local_sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
    local = torch.load(local_path, map_location="cpu", weights_only=False)["deltas"]
    sub = facts[: len(local)]
    sub_router = RouterV4(enc, sub, tau)
    sub_answers = [f["answer"] for f in sub if f["answer"] not in NUMBERS]
    old = {f["id"]: store.deltas[f["id"]] for f in sub}
    b = {}
    for name, deltas in (("existing_both", old), ("local_both", local)):
        rows = with_note(server, deltas, sub, enc, sub_router, e3[: len(sub)], sub_answers)
        b[name] = {k: _metric([r[k] for r in rows]) for k in ("probe", "paraphrase", "E3_leak")}
        examples[name] = rows
        print(name, json.dumps({k: v["rate"] for k, v in b[name].items()}), flush=True)
    for k in ("probe", "paraphrase", "E3_leak"):
        b[f"local_minus_existing_{k}"] = paired_diff_ci(
            [r[k] for r in examples["local_both"]], [r[k] for r in examples["existing_both"]]
        )
    b["local_checkpoint_sha256"] = local_sha
    report["local_anchor_with_note"] = b
    report["wall_seconds"] = round(time.time() - started, 1)
    report["examples"] = examples
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
