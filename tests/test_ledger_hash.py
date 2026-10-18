"""State hash of the transactional delta store covers every part of an entry."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "experiments/sandbox/live_learning/text"),
)

from ledger import DeltaStore  # noqa: E402


def _store():
    store = DeltaStore.__new__(DeltaStore)
    store.base_hash = "base"
    store.router_config = "enc=a|tau=0.75"
    store.deltas, store.keys, store.notes = {}, {}, {}
    return store


def _entry(store, fid, key="The mother tongue of X is", note="It is French."):
    store.deltas[fid] = {
        "m.lora_A": {"A": torch.ones(2, 3), "B": torch.zeros(3, 2)},
    }
    store.keys[fid], store.notes[fid] = key, note


def test_hash_changes_with_note_key_and_router_config():
    store = _store()
    _entry(store, "f1")
    h = store.state_hash()
    store.notes["f1"] = "It is Italian."
    assert store.state_hash() != h
    store.notes["f1"] = "It is French."
    store.keys["f1"] = "X was born in"
    assert store.state_hash() != h
    store.keys["f1"] = "The mother tongue of X is"
    assert store.state_hash() == h
    store.router_config = "enc=b|tau=0.75"
    assert store.state_hash() != h


def test_hash_is_a_function_of_the_active_set():
    a, b = _store(), _store()
    _entry(a, "f1")
    _entry(a, "f2", key="X plays", note="Jazz.")
    _entry(b, "f2", key="X plays", note="Jazz.")
    _entry(b, "f1")
    assert a.state_hash() == b.state_hash()
    empty = _store().state_hash()
    for fid in ("f1", "f2"):
        a.deltas.pop(fid), a.keys.pop(fid), a.notes.pop(fid)
    assert a.state_hash() == empty


def test_canary_divergence_is_finite_when_a_probability_underflows():
    from ledger import sym_kl

    base, edited = torch.tensor([0.0, 0.0]), torch.tensor([0.0, -120.0])
    kld = sym_kl(base, edited)
    assert kld == pytest.approx(60.0, abs=0.01)
    # the old ratio form gave NaN here, which passed `kld > limit` unrejected
    assert sym_kl(base, base) == 0.0
