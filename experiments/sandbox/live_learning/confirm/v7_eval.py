"""SS75 (b): end-to-end evaluation of the selected router v7 configuration (eval-only).

Same readings as SS67/68 (serving arms, tiered store, removal) and SS71 (locality under
the four serving policies), with the router chosen on the dev split by
router_v7_select.py at its dev threshold.

    .venv/bin/python experiments/sandbox/live_learning/confirm/v7_eval.py --dataset counterfact
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import NUMBERS  # noqa: E402
from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from pilot import CANARIES, PilotStore  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _repo_path,
    _validate_checkpoint,
    load_verified_checkpoint,
    verify_manifest_inputs,
)
from router_v7 import RouterV7, make_scorer  # noqa: E402
from serving_locality import POLICIES  # noqa: E402
from serving_locality import locality as policy_locality  # noqa: E402
from v4_eval import Server, evaluate  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")
FILES = {
    "counterfact": ("cf_n1000.ckpt", "external_counterfact_n1000.json", "cf"),
    "zsre": ("zsre_n1000.ckpt", "external_zsre_n1000.json", "zsre"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(FILES), required=True)
    ap.add_argument("--selection", default=str(CONFIRM / "router_v7_select.json"))
    ap.add_argument("--suffix", default="v7")
    ap.add_argument("--prereg", default="SS75")
    args = ap.parse_args(argv)
    ckpt, facts_file, tag = FILES[args.dataset]
    out_path = CONFIRM / f"{tag}_n1000_{args.suffix}_eval.json"
    if out_path.exists():
        ap.error(f"refusing to overwrite {out_path}")
    sel = json.loads(Path(args.selection).read_text())
    chosen, theta = sel["chosen"], sel["theta"]
    name, aug = chosen.removesuffix("+keys"), chosen.endswith("+keys")
    gen = {
        k: v["keys"]
        for k, v in json.loads((CONFIRM / "generated_keys.json").read_text())[
            "keys"
        ].items()
    }
    extra = gen if aug else None

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest,
        _repo_path(CONFIRM / ckpt, REPO_ROOT),
        _repo_path(CONFIRM / facts_file, REPO_ROOT),
        model_id,
        repo_root=REPO_ROOT,
    )
    facts = verified["facts"]
    raw = json.loads((CONFIRM / f"raw/{args.dataset}.json").read_text())

    import torch

    torch.manual_seed(0)
    scorer = make_scorer(name)
    store = PilotStore(model_id, steps=16)
    store.kl_prompts = list(CANARIES)
    checkpoint, ckpt_sha = load_verified_checkpoint(
        verified["checkpoint_path"],
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    _validate_checkpoint(checkpoint, facts)
    store.deltas = checkpoint["deltas"]
    store.keys = checkpoint["keys"]
    store.materialize()
    print(
        f"loaded {len(store.deltas)} adapters; router v7 {chosen} theta={theta}",
        flush=True,
    )

    def make_router(fs):
        return RouterV7(fs, scorer, theta, extra)

    started = time.time()
    result = evaluate(
        store,
        facts,
        None,
        theta,
        args.dataset,
        raw,
        make_router=make_router,
        run_locality=False,
    )
    tiers = {r["fact_id"]: r["tier"] for r in result["examples"]["serving"]}
    server = Server(store)
    router = make_router(facts)
    answers = [f["answer"] for f in facts if f["answer"] not in NUMBERS]
    if args.dataset == "counterfact":
        records = match_raw(facts, raw)
        queries = {
            "E3_other_relation": [
                p["query"] for p in other_relation_prompts(records, raw)
            ],
            "E2_neighborhood": [r["neighborhood_prompts"][0] for r in records],
        }
    else:
        by_src = {item["src"].strip(): item for item in raw}
        queries = {"loc_questions": [by_src[f["probe"]]["loc"] for f in facts]}
    loc, loc_examples = {}, {}
    for key, qs in queries.items():
        loc[key], loc_examples[key] = policy_locality(
            server, router, facts, qs, None, answers, tiers
        )
        print(
            key,
            json.dumps({p: loc[key][p]["leak"]["rate"] for p in POLICIES}),
            flush=True,
        )
    result["locality"] = loc
    result["examples"]["locality"] = loc_examples
    result.update(
        {
            "schema": "confirm.v7_eval.v1",
            "prereg": args.prereg,
            "dataset": args.dataset,
            "router": chosen,
            "theta": theta,
            "training_performed": False,
            "checkpoint_sha256": ckpt_sha,
            "wall_seconds": round(time.time() - started, 1),
        }
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {k: result[k] for k in ("serving", "tiered", "locality", "removal")},
            indent=1,
        )
    )
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
