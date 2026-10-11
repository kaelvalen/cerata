"""SS70: locality-aware adapter training (KL on other relations of the fact's subject).

Trains new adapters for the first 200 CounterFact test facts with the confirmatory recipe
plus three subject-local KL prompts (templates of other relations filled with the fact's
subject; the fact's own relation and the SS65 E3 test relation are excluded), then
compares them with the existing adapters of the same facts under router v4 and v3.

    .venv/bin/python experiments/sandbox/live_learning/confirm/local_anchor.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from canary_eval import HELDOUT, d_on  # noqa: E402
from facts import NUMBERS  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from pilot import CANARIES, PilotStore, Router, hit, trim  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _metric,
    _repo_path,
    load_verified_checkpoint,
    verify_manifest_inputs,
)
from router_v4 import RouterV4  # noqa: E402
from v4_eval import Server  # noqa: E402

SEED = 70
N_LOCAL = 3
CONFIRM = Path("results/live_learning/confirm")


def local_prompts(records, raw, e3_relations, seed=SEED):
    by_relation = {}
    for record in raw:
        rw = record["requested_rewrite"]
        by_relation.setdefault(rw["relation_id"], []).append(rw["prompt"])
    rng = random.Random(seed)
    out = []
    for record, test_relation in zip(records, e3_relations):
        rw = record["requested_rewrite"]
        allowed = sorted(
            r for r in by_relation if r not in (rw["relation_id"], test_relation)
        )
        relations = rng.sample(allowed, N_LOCAL)
        out.append(
            [rng.choice(by_relation[r]).replace("{}", rw["subject"]) for r in relations]
        )
    return out


def evaluate(server, deltas, facts, enc, router, e3, e2, answers):
    full = server.store.deltas
    server.store.deltas = deltas
    rows = []
    for i, f in enumerate(facts):
        row = {"fact_id": f["id"]}
        for kind, q in (("probe", f["probe"]), ("paraphrase", f["paraphrase"])):
            j, _ = router.route(q, enc)
            if j is not None and facts[j]["id"] not in deltas:
                j = None  # a rejected candidate is not in the store
            resp = server.base(q) if j is None else server.adapter(facts[j]["id"], q)
            row[kind] = hit(resp, f["answer"])
        for kind, q in (("E3", e3[i]), ("E2", e2[i])):
            base = server.base(q)
            j, _ = router.route(q, enc)
            if j is not None and facts[j]["id"] not in deltas:
                j = None
            resp = base if j is None else server.adapter(facts[j]["id"], q)
            row[f"{kind}_routed"] = j is not None
            row[f"{kind}_same"] = resp.strip() == base.strip()
            row[f"{kind}_leak"] = any(
                hit(resp, w) and not hit(base, w) for w in answers
            )
            row[f"{kind}_response"] = resp
        rows.append(row)
        trim()
    server.store.deltas = full
    keys = [
        "probe",
        "paraphrase",
        "E3_routed",
        "E3_same",
        "E3_leak",
        "E2_routed",
        "E2_same",
        "E2_leak",
    ]
    return {k: _metric([r[k] for r in rows]) for k in keys}, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=str(CONFIRM / "local_anchor_n200.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")
    tau = json.loads((CONFIRM / "router_v4_calibration.json").read_text())[
        "calibration"
    ]["tau"]

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest,
        _repo_path(CONFIRM / "cf_n1000.ckpt", REPO_ROOT),
        _repo_path(CONFIRM / "external_counterfact_n1000.json", REPO_ROOT),
        model_id,
        repo_root=REPO_ROOT,
    )
    all_facts = verified["facts"]
    facts = all_facts[: args.n]
    raw = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    records_all = match_raw(all_facts, raw)
    e3_all = other_relation_prompts(records_all, raw)
    records, e3 = records_all[: args.n], [p["query"] for p in e3_all[: args.n]]
    e2 = [r["neighborhood_prompts"][0] for r in records]
    local = local_prompts(records, raw, [p["relation_id"] for p in e3_all[: args.n]])

    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    enc = SentenceTransformer(manifest["models"]["router_encoder"]["id"])
    store = PilotStore(model_id, steps=16)
    checkpoint, _ = load_verified_checkpoint(
        verified["checkpoint_path"],
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    old = {f["id"]: checkpoint["deltas"][f["id"]] for f in facts}
    del checkpoint

    new, train_log = {}, []
    for f, prompts in zip(facts, local):
        store.kl_prompts = list(CANARIES) + prompts
        t = time.time()
        delta = store._train_delta(fact_pairs(f))
        seconds = time.time() - t
        store.kl_prompts = list(CANARIES)
        d = d_on(store, delta, CANARIES)
        if d <= 2.0:
            new[f["id"]] = delta
        train_log.append(
            {
                "fact_id": f["id"],
                "D": round(d, 4),
                "committed": d <= 2.0,
                "seconds": round(seconds, 2),
                "local_prompts": prompts,
            }
        )
        trim()
        if len(train_log) % 50 == 0:
            print(f"trained {len(train_log)}/{len(facts)}", flush=True)
    store.kl_prompts = list(CANARIES)
    torch.save(
        {"deltas": new, "keys": {f["id"]: f["probe"] for f in facts}},
        CONFIRM / f"cf_n{args.n}_local.ckpt",
    )

    server = Server(store)
    answers = [f["answer"] for f in facts if f["answer"] not in NUMBERS]
    report = {
        "prereg": "SS70",
        "tau_v4": tau,
        "n": len(facts),
        "committed": sum(r["committed"] for r in train_log),
        "train_seconds_mean": round(
            float(np.mean([r["seconds"] for r in train_log])), 2
        ),
    }
    examples = {}
    for name, deltas in (("existing", old), ("local", new)):
        for rname, router in (
            ("v4", RouterV4(enc, facts, tau)),
            ("v3", Router(enc, facts)),
        ):
            metrics, rows = evaluate(
                server, deltas, facts, enc, router, e3, e2, answers
            )
            report[f"{name}_{rname}"] = metrics
            examples[f"{name}_{rname}"] = rows
            print(
                name,
                rname,
                json.dumps({k: v["rate"] for k, v in metrics.items()}),
                flush=True,
            )
        sample = list(deltas)[:50]
        report[f"{name}_D_in_sample_mean"] = round(
            float(np.mean([d_on(store, deltas[k], CANARIES) for k in sample])), 4
        )
        report[f"{name}_D_heldout_mean"] = round(
            float(np.mean([d_on(store, deltas[k], HELDOUT) for k in sample])), 4
        )
    report["train_log"] = train_log
    report["examples"] = examples
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("train_log", "examples")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
