"""Corrected, eval-only evaluation of a saved confirmatory PilotStore checkpoint.

This is deliberately separate from the legacy aggregate pilot JSON path. The routed
metrics below serve the router-selected expert (and the base model on abstention),
record every selected route and response, and are tagged so they cannot be mistaken
for the earlier oracle-conditioned aggregates. No training or checkpoint mutation is
performed. The adjacent routed_eval_manifest.json pins full-file SHA256 values for
the existing checkpoint/fact pairs and the runtime model IDs. The saved checkpoints do
not contain a model ID, base-weight hash, or revision, so the output explicitly marks
the checkpoint-to-base model binding as unverified.

Run from the repository root (outputs are new files; an existing output is never
overwritten):

    .venv/bin/python experiments/sandbox/live_learning/confirm/routed_eval.py \\
      --ckpt results/live_learning/confirm/cf_n1000.ckpt \\
      --facts results/live_learning/confirm/external_counterfact_n1000.json \\
      --out results/live_learning/confirm/cf_n1000_routed_eval.json \\
      --model Qwen/Qwen2.5-1.5B-Instruct --revoke-sample 100

    .venv/bin/python experiments/sandbox/live_learning/confirm/routed_eval.py \\
      --ckpt results/live_learning/confirm/zsre_n1000.ckpt \\
      --facts results/live_learning/confirm/external_zsre_n1000.json \\
      --out results/live_learning/confirm/zsre_n1000_routed_eval.json \\
      --model Qwen/Qwen2.5-1.5B-Instruct --revoke-sample 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import NUMBERS  # noqa: E402
from pilot import (  # noqa: E402
    CANARIES,
    TAU,
    PilotStore,
    Router,
    base_answer,
    ci,
    hit,
    routed_answer,
    trim,
)

BOOTSTRAP_SAMPLES = 2000
REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = Path(__file__).resolve().with_name("routed_eval_manifest.json")
MANIFEST_SCHEMA = "confirm.routed_eval_manifest.v1"


def _metric(values):
    values = [bool(value) for value in values]
    if not values:
        return {"n": 0, "rate": None, "ci95": None}
    return {
        "n": len(values),
        "rate": round(float(np.mean(values)), 4),
        "ci95": ci(values, n=BOOTSTRAP_SAMPLES, seed=0),
    }


def _route_fields(index, similarity, candidates):
    return {
        "route_index": index,
        "route_fact_id": candidates[index]["id"] if index is not None else None,
        "route_score": round(similarity, 6),
        "abstained": index is None,
    }


def _parse_facts(data):
    facts = data["facts"] if isinstance(data, dict) else data
    if not isinstance(facts, list) or not facts:
        raise ValueError("facts JSON must contain a non-empty list")
    required = ("id", "subject", "answer", "probe", "paraphrase", "distractor")
    for i, fact in enumerate(facts):
        if not isinstance(fact, dict) or any(
            not isinstance(fact.get(key), str) or not fact[key] for key in required
        ):
            raise ValueError(
                f"fact at index {i} is missing a required non-empty string"
            )
    ids = [fact["id"] for fact in facts]
    if len(set(ids)) != len(ids):
        raise ValueError("fact IDs must be unique")
    return data, facts


def _sha256_file(path):
    with Path(path).open("rb") as source:
        return _sha256_stream(source)


def _sha256_stream(source):
    source.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def _assert_same_open_file(path, source):
    try:
        path_stat = Path(path).stat()
        source_stat = os.fstat(source.fileno())
    except OSError as exc:
        raise ValueError("checkpoint path changed while it was being loaded") from exc
    if (path_stat.st_dev, path_stat.st_ino) != (
        source_stat.st_dev,
        source_stat.st_ino,
    ):
        raise ValueError("checkpoint path was replaced while it was being loaded")


def load_verified_checkpoint(path, expected_sha256, loader):
    """Load through one open file and verify its contents and path identity afterward."""
    with Path(path).open("rb") as source:
        _assert_same_open_file(path, source)
        source.seek(0)
        checkpoint = loader(source)
        loaded_sha256 = _sha256_stream(source)
        _assert_same_open_file(path, source)
    if loaded_sha256 != expected_sha256:
        raise ValueError(
            "checkpoint SHA256 changed between manifest verification and load"
        )
    return checkpoint, loaded_sha256


def _repo_path(path, repo_root):
    path = Path(path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def verify_manifest_inputs(
    manifest, ckpt_path, facts_path, model_id, repo_root=REPO_ROOT
):
    """Verify the exact pinned checkpoint/fact pair before model loading or inference."""
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    models = manifest.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("base"), dict):
        raise ValueError("manifest must pin a base model identity")
    base_model = models["base"]
    expected_model_id = base_model.get("id")
    if not isinstance(expected_model_id, str) or not expected_model_id:
        raise ValueError("manifest base model ID is missing")
    if model_id != expected_model_id:
        raise ValueError(
            f"requested model ID {model_id!r} does not match pinned model "
            f"{expected_model_id!r}"
        )

    ckpt_path = _repo_path(ckpt_path, repo_root)
    facts_path = _repo_path(facts_path, repo_root)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts must be a list")
    matches = [
        artifact
        for artifact in artifacts
        if _repo_path(artifact.get("checkpoint", {}).get("path", ""), repo_root)
        == ckpt_path
        and _repo_path(artifact.get("facts", {}).get("path", ""), repo_root)
        == facts_path
    ]
    if len(matches) != 1:
        raise ValueError("checkpoint/fact paths are not a unique pinned manifest pair")
    artifact = matches[0]

    checkpoint_sha256 = _sha256_file(ckpt_path)
    expected_checkpoint_sha256 = artifact["checkpoint"].get("sha256")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA256 does not match the routed-evaluation manifest"
        )
    facts_bytes = facts_path.read_bytes()
    facts_sha256 = hashlib.sha256(facts_bytes).hexdigest()
    expected_facts_sha256 = artifact["facts"].get("sha256")
    if facts_sha256 != expected_facts_sha256:
        raise ValueError(
            "fact-file SHA256 does not match the routed-evaluation manifest"
        )

    facts_data, facts = _parse_facts(json.loads(facts_bytes))
    dataset = facts_data.get("set") if isinstance(facts_data, dict) else None
    if dataset != artifact.get("dataset"):
        raise ValueError(
            f"fact dataset {dataset!r} does not match pinned dataset "
            f"{artifact.get('dataset')!r}"
        )
    if len(facts) != artifact.get("n"):
        raise ValueError(
            f"fact count {len(facts)} does not match pinned N={artifact.get('n')}"
        )

    return {
        "artifact": artifact,
        "facts_data": facts_data,
        "facts": facts,
        "checkpoint_path": ckpt_path,
        "facts_path": facts_path,
        "checkpoint_sha256": checkpoint_sha256,
        "facts_sha256": facts_sha256,
        "models": models,
    }


def _verify_checkpoint_identity(checkpoint, base_model):
    """Compare any persisted checkpoint model metadata with the pinned runtime ID."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    sources = []
    pending = [checkpoint]
    seen = set()
    containers = (
        "metadata",
        "meta",
        "model_metadata",
        "base_metadata",
        "model_config",
        "config",
        "model",
        "base_model",
    )
    while pending:
        source = pending.pop()
        if id(source) in seen:
            continue
        seen.add(id(source))
        sources.append(source)
        pending.extend(
            nested
            for key in containers
            if isinstance((nested := source.get(key)), dict)
        )

    expected_model_id = base_model["id"]
    expected_revision = base_model.get("revision")
    expected_base_hash = base_model.get("checkpoint_base_hash")
    expected_weight_sha256 = base_model.get("base_weight_sha256")
    observed = {
        "model_id_fields": {},
        "revision_fields": {},
        "base_hash_fields": {},
        "base_weight_sha256_fields": {},
    }
    for source in sources:
        model_values = {
            key: source[key]
            for key in (
                "model_id",
                "model_name",
                "base_model_id",
                "base_model_name",
            )
            if key in source
        }
        for model_key in ("model", "base_model"):
            model_value = source.get(model_key)
            if isinstance(model_value, str):
                model_values[model_key] = model_value
            elif isinstance(model_value, dict):
                model_values.update(
                    {
                        f"{model_key}.{key}": model_value[key]
                        for key in (
                            "id",
                            "model_id",
                            "model_name",
                            "base_model",
                            "_name_or_path",
                        )
                        if key in model_value
                    }
                )
        for key, value in model_values.items():
            if value != expected_model_id:
                raise ValueError(
                    f"checkpoint {key} {value!r} does not match pinned model "
                    f"{expected_model_id!r}"
                )
        observed["model_id_fields"].update(model_values)

        revision_values = {
            key: source[key]
            for key in (
                "revision",
                "model_revision",
                "base_revision",
                "base_model_revision",
                "commit_hash",
                "_commit_hash",
            )
            if key in source
        }
        for key, value in revision_values.items():
            if expected_revision is None or value != expected_revision:
                raise ValueError(
                    f"checkpoint {key} cannot be verified against the manifest revision"
                )
        observed["revision_fields"].update(revision_values)

        base_hash_values = {
            key: source[key]
            for key in ("base_hash", "base_model_hash")
            if key in source
        }
        for key, value in base_hash_values.items():
            if expected_base_hash is None or value != expected_base_hash:
                raise ValueError(
                    f"checkpoint {key} is not pinned to a matching manifest value"
                )
        observed["base_hash_fields"].update(base_hash_values)

        weight_hash_values = {
            key: source[key]
            for key in (
                "base_weight_sha256",
                "base_weights_sha256",
                "base_weight_hash",
                "base_weights_hash",
            )
            if key in source
        }
        for key, value in weight_hash_values.items():
            if expected_weight_sha256 is None or value != expected_weight_sha256:
                raise ValueError(
                    f"checkpoint {key} is not pinned to a matching manifest value"
                )
        observed["base_weight_sha256_fields"].update(weight_hash_values)

    observed["model_id_verified"] = bool(observed["model_id_fields"])
    observed["revision_verified"] = bool(observed["revision_fields"])
    observed["base_hash_verified"] = bool(observed["base_hash_fields"])
    observed["base_weight_sha256_verified"] = bool(
        observed["base_weight_sha256_fields"]
    )
    return observed


def _validate_checkpoint(checkpoint, facts):
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("deltas"), dict
    ):
        raise ValueError("checkpoint must contain a deltas mapping")
    deltas = checkpoint["deltas"]
    fact_by_id = {fact["id"]: fact for fact in facts}
    fact_ids = set(fact_by_id)
    delta_ids = set(deltas)
    if delta_ids != fact_ids:
        missing = sorted(fact_ids - delta_ids)
        extra = sorted(delta_ids - fact_ids)
        raise ValueError(
            "checkpoint delta IDs do not match the fact subset "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    keys = checkpoint.get("keys")
    if not isinstance(keys, dict) or set(keys) != fact_ids:
        raise ValueError("checkpoint keys must match the fact subset IDs")
    mismatched = [fid for fid, fact in fact_by_id.items() if keys[fid] != fact["probe"]]
    if mismatched:
        raise ValueError(
            "checkpoint key prompts differ from the supplied facts for IDs "
            f"{mismatched[:5]}"
        )


def evaluate_routed(
    store, facts, enc, router=None, revoke_sample=100, progress_every=100
):
    """Run routed probe/paraphrase/distractor and revoke/retain inference.

    The route index is always interpreted against its candidate list: all facts for
    the main pass, then only the retained facts for revoke/retain. An abstention is
    served by the base model. This function accepts fake stores/routers so the
    behavioral contract can be tested without loading a language model.
    """
    if not facts:
        raise ValueError("cannot evaluate an empty fact subset")
    if revoke_sample < 1:
        raise ValueError("revoke_sample must be positive")
    if router is None:
        router = Router(enc, facts)

    words = [fact["answer"] for fact in facts if fact["answer"] not in NUMBERS]

    # Match the original revoke design: its pre-revoke reference is the base model,
    # not the merged store or a gold-selected expert.
    base_dist = {}
    for i, fact in enumerate(facts):
        base_dist[fact["id"]] = base_answer(store, fact["distractor"])
        if progress_every and (i + 1) % progress_every == 0:
            print(f"routed eval: base distractors {i + 1}/{len(facts)}", flush=True)

    sample_size = min(revoke_sample, max(1, len(facts) // 2))
    sample_facts = facts[:sample_size]
    keep = facts[sample_size:]
    pre = {fact["id"]: base_answer(store, fact["probe"]) for fact in sample_facts}

    examples = []
    for i, fact in enumerate(facts):
        probe_index, probe_score, probe_resp = routed_answer(
            store, router, facts, fact["probe"], enc
        )
        para_index, para_score, para_resp = routed_answer(
            store, router, facts, fact["paraphrase"], enc
        )
        dist_index, dist_score, dist_resp = routed_answer(
            store, router, facts, fact["distractor"], enc
        )

        probe_hit = hit(probe_resp, fact["answer"])
        para_hit = hit(para_resp, fact["answer"])
        new_answer_leak = any(
            hit(dist_resp, word) and not hit(base_dist[fact["id"]], word)
            for word in words
        )
        examples.append(
            {
                "fact_id": fact["id"],
                "answer": fact["answer"],
                "probe": {
                    "query": fact["probe"],
                    **_route_fields(probe_index, probe_score, facts),
                    "route_correct": probe_index == i,
                    "answer_hit": probe_hit,
                    "response": probe_resp,
                },
                "paraphrase": {
                    "query": fact["paraphrase"],
                    **_route_fields(para_index, para_score, facts),
                    "route_correct": para_index == i,
                    "answer_hit": para_hit,
                    "response": para_resp,
                },
                "distractor": {
                    "query": fact["distractor"],
                    **_route_fields(dist_index, dist_score, facts),
                    "known_answer_match": any(hit(dist_resp, word) for word in words),
                    "new_known_answer_leak": new_answer_leak,
                    "no_leak": not new_answer_leak,
                    "base_response": base_dist[fact["id"]],
                    "response": dist_resp,
                },
            }
        )
        trim()
        if progress_every and (i + 1) % progress_every == 0:
            print(f"routed eval: routed probes {i + 1}/{len(facts)}", flush=True)

    revoke_seconds = []
    for fact in sample_facts:
        start = time.time()
        store.revoke(fact["id"])
        revoke_seconds.append(time.time() - start)

    router_keep = Router(enc, keep)
    removed = []
    for fact in sample_facts:
        gone_index, gone_score, gone_resp = routed_answer(
            store, router_keep, keep, fact["probe"], enc
        )
        answer_hit = hit(gone_resp, fact["answer"])

        return_index, return_score, return_resp = routed_answer(
            store, router_keep, keep, fact["probe"], enc
        )
        base_hit = hit(pre[fact["id"]], fact["answer"])
        returned_hit = hit(return_resp, fact["answer"])
        removed.append(
            {
                "fact_id": fact["id"],
                "answer": fact["answer"],
                "gone_check": {
                    "query": fact["probe"],
                    **_route_fields(gone_index, gone_score, keep),
                    "answer_hit": answer_hit,
                    "gone": not answer_hit,
                    "response": gone_resp,
                },
                "return_check": {
                    "query": fact["probe"],
                    **_route_fields(return_index, return_score, keep),
                    "base_answer_hit": base_hit,
                    "post_revoke_answer_hit": returned_hit,
                    "return_match": returned_hit == base_hit,
                    "response": return_resp,
                },
                "pre_revoke_base_response": pre[fact["id"]],
            }
        )

    retained = []
    for i, fact in enumerate(keep):
        index, score, resp = routed_answer(store, router_keep, keep, fact["probe"], enc)
        retained.append(
            {
                "fact_id": fact["id"],
                "answer": fact["answer"],
                "query": fact["probe"],
                **_route_fields(index, score, keep),
                "route_correct": index == i,
                "answer_hit": hit(resp, fact["answer"]),
                "response": resp,
            }
        )

    gone_values = [row["gone_check"]["gone"] for row in removed]
    return_values = [row["return_check"]["return_match"] for row in removed]
    retain_values = [row["answer_hit"] for row in retained]
    metrics = {
        "probe_route_accuracy": _metric(
            [row["probe"]["route_correct"] for row in examples]
        ),
        "paraphrase_route_accuracy": _metric(
            [row["paraphrase"]["route_correct"] for row in examples]
        ),
        "probe_abstention_rate": _metric(
            [row["probe"]["abstained"] for row in examples]
        ),
        "paraphrase_abstention_rate": _metric(
            [row["paraphrase"]["abstained"] for row in examples]
        ),
        "distractor_abstention_rate": _metric(
            [row["distractor"]["abstained"] for row in examples]
        ),
        "probe_efficacy": _metric([row["probe"]["answer_hit"] for row in examples]),
        "paraphrase_efficacy": _metric(
            [row["paraphrase"]["answer_hit"] for row in examples]
        ),
        "distractor_no_leak": _metric(
            [row["distractor"]["no_leak"] for row in examples]
        ),
        "revoke_token_gone_rate": _metric(gone_values),
        "revoke_return_match_rate": _metric(return_values),
        "retain_rate": _metric(retain_values),
        "retain_route_accuracy": _metric([row["route_correct"] for row in retained]),
    }
    return {
        "examples": examples,
        "revoke": {
            "sample": len(sample_facts),
            "removed": removed,
            "retained": retained,
            "mean_revoke_seconds": round(float(np.mean(revoke_seconds)), 4),
        },
        "metrics": metrics,
    }


def _default_output(ckpt_path):
    return ckpt_path.with_name(f"{ckpt_path.stem}_routed_eval.json")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, help="saved PilotStore .ckpt")
    ap.add_argument("--facts", required=True, help="matching saved fact subset JSON")
    ap.add_argument("--model", default=None, help="must match the pinned manifest ID")
    ap.add_argument("--revoke-sample", type=int, default=100)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument(
        "--out", default=None, help="new output path; existing files refused"
    )
    args = ap.parse_args(argv)

    ckpt_path = _repo_path(args.ckpt, REPO_ROOT)
    facts_path = _repo_path(args.facts, REPO_ROOT)
    out_path = Path(args.out) if args.out else _default_output(ckpt_path)
    if args.revoke_sample < 1:
        ap.error("--revoke-sample must be positive")
    if args.progress_every < 0:
        ap.error("--progress-every must be non-negative")
    if not MANIFEST_PATH.is_file():
        ap.error(f"required pinned manifest not found: {MANIFEST_PATH}")
    if not ckpt_path.is_file():
        ap.error(f"checkpoint not found: {ckpt_path}")
    if not facts_path.is_file():
        ap.error(f"fact subset not found: {facts_path}")
    protected_paths = {ckpt_path.resolve(), facts_path.resolve()}
    if out_path.resolve() in protected_paths:
        ap.error("--out must be distinct from the checkpoint and fact subset")
    if out_path.exists():
        ap.error(
            f"refusing to overwrite existing output: {out_path}; choose a new --out"
        )

    try:
        manifest_bytes = MANIFEST_PATH.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        models = manifest["models"]
        base_model_id = args.model or models["base"]["id"]
        verified = verify_manifest_inputs(
            manifest, ckpt_path, facts_path, base_model_id, repo_root=REPO_ROOT
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        ap.error(f"pinned input verification failed: {exc}")

    facts = verified["facts"]
    router_model = verified["models"].get("router_encoder")
    if not isinstance(router_model, dict) or not router_model.get("id"):
        ap.error("pinned manifest is missing its router encoder identity")

    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    enc = SentenceTransformer(router_model["id"])
    store = PilotStore(base_model_id, steps=16)
    store.kl_prompts = list(CANARIES)
    try:
        checkpoint, loaded_checkpoint_sha256 = load_verified_checkpoint(
            ckpt_path,
            verified["checkpoint_sha256"],
            lambda source: torch.load(source, map_location="cpu", weights_only=False),
        )
        checkpoint_identity = _verify_checkpoint_identity(
            checkpoint, verified["models"]["base"]
        )
        _validate_checkpoint(checkpoint, facts)
    except ValueError as exc:
        ap.error(f"checkpoint verification failed: {exc}")
    store.deltas = checkpoint["deltas"]
    store.keys = checkpoint["keys"]
    store.materialize()
    print(
        f"loaded {len(store.deltas)}/{len(facts)} experts; training disabled",
        flush=True,
    )

    started = time.time()
    evaluated = evaluate_routed(
        store,
        facts,
        enc,
        router=Router(enc, facts),
        revoke_sample=args.revoke_sample,
        progress_every=args.progress_every,
    )
    artifact = verified["artifact"]
    base_identity = verified["models"]["base"]
    verified_identity = {
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256": manifest_sha256,
            "schema": manifest["schema"],
            "artifact_id": artifact["id"],
        },
        "dataset": artifact["dataset"],
        "n": artifact["n"],
        "facts": {
            "path": artifact["facts"]["path"],
            "sha256": verified["facts_sha256"],
            "sha256_scope": artifact["facts"].get("sha256_scope"),
        },
        "checkpoint": {
            "path": artifact["checkpoint"]["path"],
            "sha256": loaded_checkpoint_sha256,
            "sha256_scope": artifact["checkpoint"].get("sha256_scope"),
        },
        "models": {
            "base": {
                "runtime_id": base_identity["id"],
                "id_binding": base_identity.get("id_binding"),
                "checkpoint_model_id_verified": checkpoint_identity[
                    "model_id_verified"
                ],
                "revision": base_identity.get("revision"),
                "revision_verified": checkpoint_identity["revision_verified"],
                "base_weight_sha256": base_identity.get("base_weight_sha256"),
                "base_weight_sha256_verified": checkpoint_identity[
                    "base_weight_sha256_verified"
                ],
                "limitation": base_identity.get("limitation"),
                "checkpoint_metadata_observed": checkpoint_identity,
            },
            "router_encoder": {
                "runtime_id": router_model["id"],
                "revision": router_model.get("revision"),
                "revision_verified": router_model.get("revision_verified", False),
                "loader": router_model.get("loader"),
            },
        },
    }
    output = {
        "schema": "confirm.routed_eval.v1",
        "evaluation_kind": "corrected_routed_eval_only",
        "metric_conditioning": "router-selected-expert-or-base-on-abstention",
        "warning": (
            "Do not treat legacy pilot_external_* aggregate efficacy values as routed: "
            "their probe path served the gold fact ID whenever routing did not abstain."
        ),
        "training_performed": False,
        "verified_identity": verified_identity,
        "dataset": artifact["dataset"],
        "checkpoint": artifact["checkpoint"]["path"],
        "facts": artifact["facts"]["path"],
        "runtime_model_id": base_model_id,
        "n": len(facts),
        "router": {"class": "Router", "tau": TAU, "encoder": router_model["id"]},
        "bootstrap": {
            "method": "percentile",
            "confidence": 0.95,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": 0,
        },
        **evaluated,
        "wall_seconds": round(time.time() - started, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("x", encoding="utf-8") as output_file:
            json.dump(output, output_file, ensure_ascii=False, indent=1)
            output_file.write("\n")
    except FileExistsError:
        ap.error(
            f"refusing to overwrite existing output: {out_path}; choose a new --out"
        )
    print(f"corrected routed eval saved: {out_path}", flush=True)
    print(json.dumps(output["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
