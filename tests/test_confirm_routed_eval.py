"""Deterministic route-serving regressions; no model downloads or GPU required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "sandbox" / "live_learning" / "confirm"
sys.path.insert(0, str(CONFIRM))

import pilot  # noqa: E402
import routed_eval  # noqa: E402


class FakeTensor:
    def numel(self):
        return 1

    def element_size(self):
        return 4


class FakeStore:
    def __init__(self, facts):
        self.answers = {fact["id"]: fact["answer"] for fact in facts}
        self.deltas = {
            fid: {"weight": {"A": FakeTensor(), "B": FakeTensor()}}
            for fid in self.answers
        }
        self.calls = []

    def materialize(self):
        pass

    def answer(self, query, system=None):
        active = tuple(self.deltas)
        self.calls.append((query, active))
        return self.answers[active[0]] if active else "BASE"

    def revoke(self, fid):
        del self.deltas[fid]


class TableRouter:
    def __init__(self, routes=None):
        self.routes = routes or {}

    def route(self, query, enc):
        return self.routes.get(query, (None, 0.0))


def fact(fid, answer):
    return {
        "id": fid,
        "subject": fid,
        "answer": answer,
        "probe": f"probe {fid}",
        "paraphrase": f"paraphrase {fid}",
        "distractor": f"distractor {fid}",
    }


def pinned_fixture(tmp_path):
    facts_path = tmp_path / "facts.json"
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    facts_doc = {
        "set": "counterfact",
        "facts": [fact("f0", "alpha"), fact("f1", "beta")],
    }
    facts_path.write_text(json.dumps(facts_doc, ensure_ascii=False, indent=1))
    checkpoint_path.write_bytes(b"small checkpoint fixture")
    manifest = {
        "schema": routed_eval.MANIFEST_SCHEMA,
        "models": {
            "base": {"id": "test/base-model", "revision": None},
            "router_encoder": {"id": "test/router-encoder", "revision": None},
        },
        "artifacts": [
            {
                "id": "counterfact_fixture_n2",
                "dataset": "counterfact",
                "n": 2,
                "checkpoint": {
                    "path": "checkpoint.ckpt",
                    "sha256": routed_eval._sha256_file(checkpoint_path),
                },
                "facts": {
                    "path": "facts.json",
                    "sha256": routed_eval._sha256_file(facts_path),
                },
            }
        ],
    }
    return manifest, checkpoint_path, facts_path, facts_doc


@pytest.fixture(autouse=True)
def no_allocator_or_model_work(monkeypatch):
    monkeypatch.setattr(pilot, "trim", lambda: None)
    monkeypatch.setattr(routed_eval, "trim", lambda: None)


def test_probe_efficacy_serves_selected_route_not_gold_and_scores_accuracy(
    monkeypatch,
):
    facts = [fact("f0", "alpha"), fact("f1", "beta")]
    routes = {
        "probe f0": (1, 0.9),  # non-gold expert for f0
        "probe f1": (1, 0.9),  # correct/gold expert for f1
        "paraphrase f0": (0, 0.9),
        "paraphrase f1": (1, 0.9),
        "distractor f0": (None, 0.0),
        "distractor f1": (None, 0.0),
    }
    store = FakeStore(facts)
    monkeypatch.setitem(pilot.BASE_DIST, "f0", "BASE")
    monkeypatch.setitem(pilot.BASE_DIST, "f1", "BASE")

    metrics = pilot.eval_ours(
        store,
        facts,
        enc=None,
        router=TableRouter(routes),
        words=["alpha", "beta"],
        train_seconds=0,
        adds=[{"seconds": 0.0}],
    )

    probe_calls = {
        query: active for query, active in store.calls if query.startswith("probe ")
    }
    assert probe_calls == {"probe f0": ("f1",), "probe f1": ("f1",)}
    assert metrics["route_accuracy"] == 0.5
    assert metrics["efficacy"] == 0.5


def test_retain_serves_router_selected_keep_expert_for_gold_and_non_gold_routes(
    monkeypatch,
):
    facts = [
        fact("gone0", "answer0"),
        fact("gone1", "answer1"),
        fact("keep0", "answer2"),
        fact("keep1", "answer3"),
    ]
    store = FakeStore(facts)
    selected_routes = {
        "probe gone0": (0, 0.9),
        "probe gone1": (0, 0.9),
        "probe keep0": (0, 0.9),  # correct/gold index in keep
        "probe keep1": (0, 0.9),  # non-gold index in keep
    }

    class KeepRouter(TableRouter):
        def __init__(self, enc, keep):
            super().__init__(selected_routes)

    monkeypatch.setattr(pilot, "Router", KeepRouter)
    result = pilot.run_revoke(
        store,
        facts,
        enc=None,
        router=TableRouter(),
        pre={"gone0": "BASE", "gone1": "BASE"},
        sample=2,
    )

    keep_calls = {
        query: active
        for query, active in store.calls
        if query in {"probe keep0", "probe keep1"}
    }
    assert keep_calls == {"probe keep0": ("keep0",), "probe keep1": ("keep0",)}
    assert result["retain_rate"] == 0.5
    assert result["token_gone_rate"] == 1.0
    assert result["return_match_rate"] == 1.0


def test_abstention_serves_base_for_probe_and_retained_fact(monkeypatch):
    one_fact = [fact("only", "answer")]
    one_store = FakeStore(one_fact)
    monkeypatch.setitem(pilot.BASE_DIST, "only", "BASE")
    pilot.eval_ours(
        one_store,
        one_fact,
        enc=None,
        router=TableRouter(),
        words=["answer"],
        train_seconds=0,
        adds=[{"seconds": 0.0}],
    )
    assert {query: active for query, active in one_store.calls} == {
        "probe only": (),
        "paraphrase only": (),
        "distractor only": (),
    }

    facts = [
        fact("gone0", "answer0"),
        fact("gone1", "answer1"),
        fact("keep", "answer2"),
    ]
    store = FakeStore(facts)

    class AbstainingKeepRouter(TableRouter):
        def __init__(self, enc, keep):
            super().__init__()

    monkeypatch.setattr(pilot, "Router", AbstainingKeepRouter)
    result = pilot.run_revoke(
        store,
        facts,
        enc=None,
        router=TableRouter(),
        pre={"gone0": "BASE"},
        sample=1,
    )
    retain_calls = [active for query, active in store.calls if query == "probe keep"]
    assert retain_calls == [()]
    assert result["retain_rate"] == 0.0
    assert result["return_match_rate"] == 1.0


def test_detailed_eval_persists_routes_hits_and_bootstrap_intervals(monkeypatch):
    facts = [
        fact("gone0", "answer0"),
        fact("gone1", "answer1"),
        fact("keep0", "answer2"),
        fact("keep1", "answer3"),
    ]
    routes = {}
    for i, item in enumerate(facts):
        routes[item["probe"]] = (1, 0.9) if i == 0 else (i, 0.9)
        routes[item["paraphrase"]] = (i, 0.9)
        routes[item["distractor"]] = (None, 0.0)
    store = FakeStore(facts)

    class KeepRouter(TableRouter):
        def __init__(self, enc, keep):
            super().__init__(
                {
                    "probe gone0": (0, 0.9),
                    "probe gone1": (0, 0.9),
                    "probe keep0": (0, 0.9),
                    "probe keep1": (0, 0.9),
                }
            )

    monkeypatch.setattr(routed_eval, "Router", KeepRouter)
    result = routed_eval.evaluate_routed(
        store,
        facts,
        enc=None,
        router=TableRouter(routes),
        revoke_sample=2,
        progress_every=0,
    )

    assert result["examples"][0]["probe"]["route_fact_id"] == "gone1"
    assert result["examples"][0]["probe"]["answer_hit"] is False
    assert result["examples"][0]["probe"]["route_correct"] is False
    assert result["metrics"]["probe_route_accuracy"]["rate"] == 0.75
    assert result["metrics"]["probe_route_accuracy"]["n"] == 4
    assert len(result["metrics"]["probe_route_accuracy"]["ci95"]) == 2
    for name in (
        "paraphrase_route_accuracy",
        "distractor_abstention_rate",
        "probe_efficacy",
        "retain_rate",
    ):
        assert len(result["metrics"][name]["ci95"]) == 2
    assert result["examples"][0]["distractor"]["abstained"] is True
    assert result["revoke"]["retained"][1]["route_fact_id"] == "keep0"
    assert result["revoke"]["retained"][1]["answer_hit"] is False


def test_route_index_contract_rejects_out_of_bounds_results():
    candidate = fact("candidate", "answer")
    store = FakeStore([candidate])
    with pytest.raises(IndexError, match="outside candidate range"):
        pilot.routed_answer(
            store,
            TableRouter({"query": (1, 0.9)}),
            [candidate],
            "query",
            enc=None,
        )


@pytest.mark.parametrize(
    ("route_result", "exception"),
    [
        (None, TypeError),
        ((0, 0.9, "extra"), TypeError),
        ((0.5, 0.9), TypeError),
        ((None, float("nan")), ValueError),
    ],
)
def test_route_index_contract_rejects_malformed_results(route_result, exception):
    candidate = fact("candidate", "answer")
    store = FakeStore([candidate])
    with pytest.raises(exception):
        pilot.routed_answer(
            store,
            TableRouter({"query": route_result}),
            [candidate],
            "query",
            enc=None,
        )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [("answer", "changed answer"), ("paraphrase", "changed paraphrase")],
)
def test_manifest_rejects_fact_mutations_even_when_ids_and_probes_match(
    tmp_path, field, changed_value
):
    manifest, checkpoint_path, facts_path, facts_doc = pinned_fixture(tmp_path)
    original_ids_and_probes = [
        (item["id"], item["probe"]) for item in facts_doc["facts"]
    ]
    facts_doc["facts"][0][field] = changed_value
    facts_path.write_text(json.dumps(facts_doc, ensure_ascii=False, indent=1))
    assert [(item["id"], item["probe"]) for item in facts_doc["facts"]] == (
        original_ids_and_probes
    )

    with pytest.raises(ValueError, match="fact-file SHA256"):
        routed_eval.verify_manifest_inputs(
            manifest,
            checkpoint_path,
            facts_path,
            "test/base-model",
            repo_root=tmp_path,
        )


def test_manifest_rejects_model_and_checkpoint_identity_mismatches(tmp_path):
    manifest, checkpoint_path, facts_path, _ = pinned_fixture(tmp_path)
    with pytest.raises(ValueError, match="requested model ID"):
        routed_eval.verify_manifest_inputs(
            manifest,
            checkpoint_path,
            facts_path,
            "other/base-model",
            repo_root=tmp_path,
        )

    with pytest.raises(ValueError, match="checkpoint model_name"):
        routed_eval._verify_checkpoint_identity(
            {"model_name": "other/base-model"}, manifest["models"]["base"]
        )

    with pytest.raises(ValueError, match="revision"):
        routed_eval._verify_checkpoint_identity(
            {"metadata": {"revision": "un-pinned-revision"}},
            manifest["models"]["base"],
        )
    with pytest.raises(ValueError, match="base_hash"):
        routed_eval._verify_checkpoint_identity(
            {"base_hash": "un-pinned-base-hash"}, manifest["models"]["base"]
        )


def test_manifest_rejects_checkpoint_byte_mutation(tmp_path):
    manifest, checkpoint_path, facts_path, _ = pinned_fixture(tmp_path)
    checkpoint_path.write_bytes(b"different checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        routed_eval.verify_manifest_inputs(
            manifest,
            checkpoint_path,
            facts_path,
            "test/base-model",
            repo_root=tmp_path,
        )


@pytest.mark.parametrize("change", ["in_place", "replace"])
def test_checkpoint_change_between_hash_and_load_aborts(tmp_path, change):
    manifest, checkpoint_path, facts_path, _ = pinned_fixture(tmp_path)
    verified = routed_eval.verify_manifest_inputs(
        manifest,
        checkpoint_path,
        facts_path,
        "test/base-model",
        repo_root=tmp_path,
    )
    loaded_bytes = []

    def mutate_then_load(source):
        if change == "in_place":
            checkpoint_path.write_bytes(b"changed checkpoint bytes")
        else:
            replacement = tmp_path / "replacement.ckpt"
            replacement.write_bytes(b"replacement checkpoint bytes")
            replacement.replace(checkpoint_path)
        source.seek(0)
        loaded_bytes.append(source.read())
        return loaded_bytes[-1]

    with pytest.raises(
        ValueError,
        match="checkpoint (SHA256 changed between manifest verification and load|"
        "path was replaced while it was being loaded)",
    ):
        routed_eval.load_verified_checkpoint(
            checkpoint_path, verified["checkpoint_sha256"], mutate_then_load
        )
    assert loaded_bytes


def test_verified_checkpoint_load_returns_the_loaded_stream_digest(tmp_path):
    manifest, checkpoint_path, facts_path, _ = pinned_fixture(tmp_path)
    verified = routed_eval.verify_manifest_inputs(
        manifest,
        checkpoint_path,
        facts_path,
        "test/base-model",
        repo_root=tmp_path,
    )
    original_bytes = checkpoint_path.read_bytes()

    loaded, loaded_sha256 = routed_eval.load_verified_checkpoint(
        checkpoint_path,
        verified["checkpoint_sha256"],
        lambda source: source.read(),
    )

    assert loaded == original_bytes
    assert loaded_sha256 == verified["checkpoint_sha256"]
