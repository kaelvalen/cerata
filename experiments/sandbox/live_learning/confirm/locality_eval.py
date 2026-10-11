"""SS65: paraphrase-conditional retrieval comparison and harder locality (eval-only).

Loads a saved confirmatory store through the same SHA-256 manifest verification as
routed_eval.py and runs three readings on CounterFact, all served through the router
(the base model on abstention):

  E1  retrieval (instructed prompt, routed fact's statement) vs the routed adapter on
      the held-out paraphrases, overall and on the route-correct subset;
  E2  the first CounterFact neighborhood prompt of each fact (other subject, same
      relation): abstention, output identical to the base, leak of the stored target;
  E3  a prompt template of another relation filled with the fact's own subject
      (drawn with random.Random(65)): routed rate, identical to the base, leak.

No training or checkpoint mutation. Run from the repository root:

    .venv/bin/python experiments/sandbox/live_learning/confirm/locality_eval.py \\
      --ckpt results/live_learning/confirm/cf_n1000.ckpt \\
      --facts results/live_learning/confirm/external_counterfact_n1000.json \\
      --raw results/live_learning/confirm/raw/counterfact.json \\
      --out results/live_learning/confirm/cf_n1000_locality_eval.json
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

from pilot import (  # noqa: E402
    CANARIES,
    PilotStore,
    Router,
    base_answer,
    hit,
    rag_system,
    routed_answer,
    trim,
)
from routed_eval import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    MANIFEST_PATH,
    REPO_ROOT,
    _metric,
    _repo_path,
    _route_fields,
    _validate_checkpoint,
    load_verified_checkpoint,
    verify_manifest_inputs,
)

TEMPLATE_SEED = 65


def match_raw(facts, raw):
    """Map each adapted fact to its CounterFact record by (subject, probe, target)."""
    index = {}
    for record in raw:
        rw = record["requested_rewrite"]
        key = (
            rw["subject"],
            rw["prompt"].replace("{}", rw["subject"]).strip(),
            rw["target_new"]["str"],
        )
        index.setdefault(key, []).append(record)
    matched = []
    for fact in facts:
        records = index.get((fact["subject"], fact["probe"], fact["answer"]), [])
        if len(records) != 1:
            raise ValueError(f"fact {fact['id']} matches {len(records)} raw records")
        matched.append(records[0])
    return matched


def other_relation_prompts(records, raw, seed=TEMPLATE_SEED):
    """For each record, a template of a different relation filled with its subject."""
    by_relation = {}
    for record in raw:
        rw = record["requested_rewrite"]
        by_relation.setdefault(rw["relation_id"], []).append(rw["prompt"])
    relations = sorted(by_relation)
    rng = random.Random(seed)
    prompts = []
    for record in records:
        rw = record["requested_rewrite"]
        others = [r for r in relations if r != rw["relation_id"]]
        relation = rng.choice(others)
        template = rng.choice(by_relation[relation])
        prompts.append(
            {"relation_id": relation, "query": template.replace("{}", rw["subject"])}
        )
    return prompts


def paired_diff_ci(a, b, n=BOOTSTRAP_SAMPLES, seed=0):
    """Mean of a - b with a 95% percentile bootstrap interval over paired items."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return {"n": 0, "diff": None, "ci95": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n, a.size))
    boots = (a[idx] - b[idx]).mean(axis=1)
    return {
        "n": int(a.size),
        "diff": round(float((a - b).mean()), 4),
        "ci95": [
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4),
        ],
    }


def _locality_rows(store, router, facts, queries, enc, progress, label):
    rows = []
    for i, (fact, query) in enumerate(zip(facts, queries)):
        base = base_answer(store, query)
        index, score, resp = routed_answer(store, router, facts, query, enc)
        rows.append(
            {
                "fact_id": fact["id"],
                "query": query,
                **_route_fields(index, score, facts),
                "identical_to_base": resp.strip() == base.strip(),
                "leak": hit(resp, fact["answer"]) and not hit(base, fact["answer"]),
                "base_response": base,
                "response": resp,
            }
        )
        trim()
        if progress and (i + 1) % progress == 0:
            print(f"{label}: {i + 1}/{len(facts)}", flush=True)
    return rows


def _locality_metrics(rows):
    return {
        "abstention_rate": _metric([r["abstained"] for r in rows]),
        "identical_to_base": _metric([r["identical_to_base"] for r in rows]),
        "leak_rate": _metric([r["leak"] for r in rows]),
        "identical_given_routed": _metric(
            [r["identical_to_base"] for r in rows if not r["abstained"]]
        ),
        "leak_given_routed": _metric([r["leak"] for r in rows if not r["abstained"]]),
    }


def evaluate(store, facts, records, raw, enc, progress=100):
    router = Router(enc, facts)

    para = []
    for i, fact in enumerate(facts):
        index, score, ours = routed_answer(
            store, router, facts, fact["paraphrase"], enc
        )
        system = None if index is None else rag_system("instruct", facts, index)
        rag = base_answer(store, fact["paraphrase"], system=system)
        para.append(
            {
                "fact_id": fact["id"],
                "query": fact["paraphrase"],
                **_route_fields(index, score, facts),
                "route_correct": index == i,
                "ours_hit": hit(ours, fact["answer"]),
                "rag_hit": hit(rag, fact["answer"]),
                "ours_response": ours,
                "rag_response": rag,
            }
        )
        trim()
        if progress and (i + 1) % progress == 0:
            print(f"E1 paraphrases: {i + 1}/{len(facts)}", flush=True)
    correct = [r for r in para if r["route_correct"]]
    e1 = {
        "all": {
            "ours": _metric([r["ours_hit"] for r in para]),
            "rag_instruct": _metric([r["rag_hit"] for r in para]),
            "diff_ours_minus_rag": paired_diff_ci(
                [r["ours_hit"] for r in para], [r["rag_hit"] for r in para]
            ),
        },
        "route_correct": {
            "ours": _metric([r["ours_hit"] for r in correct]),
            "rag_instruct": _metric([r["rag_hit"] for r in correct]),
            "diff_ours_minus_rag": paired_diff_ci(
                [r["ours_hit"] for r in correct], [r["rag_hit"] for r in correct]
            ),
        },
    }

    neighborhood = [r["neighborhood_prompts"][0] for r in records]
    e2_rows = _locality_rows(store, router, facts, neighborhood, enc, progress, "E2")

    other = other_relation_prompts(records, raw)
    e3_rows = _locality_rows(
        store, router, facts, [p["query"] for p in other], enc, progress, "E3"
    )
    for row, prompt in zip(e3_rows, other):
        row["relation_id"] = prompt["relation_id"]

    return {
        "E1_paraphrase_vs_retrieval": {"metrics": e1, "examples": para},
        "E2_neighborhood": {"metrics": _locality_metrics(e2_rows), "examples": e2_rows},
        "E3_same_subject_other_relation": {
            "metrics": _locality_metrics(e3_rows),
            "examples": e3_rows,
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--raw", required=True, help="ROME counterfact.json")
    ap.add_argument("--out", required=True, help="new output path")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args(argv)

    ckpt_path = _repo_path(args.ckpt, REPO_ROOT)
    facts_path = _repo_path(args.facts, REPO_ROOT)
    out_path = Path(args.out)
    if out_path.exists():
        ap.error(f"refusing to overwrite existing output: {out_path}")

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    try:
        verified = verify_manifest_inputs(
            manifest, ckpt_path, facts_path, model_id, repo_root=REPO_ROOT
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        ap.error(f"pinned input verification failed: {exc}")
    if verified["artifact"]["dataset"] != "counterfact":
        ap.error("SS65 is defined on CounterFact only")
    facts = verified["facts"]
    raw = json.loads(_repo_path(args.raw, REPO_ROOT).read_text())
    records = match_raw(facts, raw)

    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    enc = SentenceTransformer(manifest["models"]["router_encoder"]["id"])
    store = PilotStore(model_id, steps=16)
    store.kl_prompts = list(CANARIES)
    checkpoint, ckpt_sha = load_verified_checkpoint(
        ckpt_path,
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    _validate_checkpoint(checkpoint, facts)
    store.deltas = checkpoint["deltas"]
    store.keys = checkpoint["keys"]
    store.materialize()
    print(f"loaded {len(store.deltas)} adapters; training disabled", flush=True)

    started = time.time()
    result = evaluate(store, facts, records, raw, enc, progress=args.progress_every)
    result.update(
        {
            "schema": "confirm.locality_eval.v1",
            "prereg": "SS65",
            "training_performed": False,
            "checkpoint": {"path": str(args.ckpt), "sha256": ckpt_sha},
            "facts": {"path": str(args.facts), "sha256": verified["facts_sha256"]},
            "template_seed": TEMPLATE_SEED,
            "wall_seconds": round(time.time() - started, 1),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    summary = {
        "E1": result["E1_paraphrase_vs_retrieval"]["metrics"],
        "E2": result["E2_neighborhood"]["metrics"],
        "E3": result["E3_same_subject_other_relation"]["metrics"],
    }
    print(json.dumps(summary, indent=1), flush=True)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
