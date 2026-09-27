"""PTM-CIL study infrastructure: the RanPAC feature map in the medium path, the
bank-agnostic P2 decomposition, the benchmark protocol and the runner's smoke."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cerata.api import Batch, Cerata
from cerata.core.hashing import digest
from cerata.core.random_features import RandomProjection
from cerata.data.ptm_benchmarks import (
    BENCHMARKS,
    class_order,
    split_tasks,
    stream_test_order,
    task_increments,
)
from cerata.edit import LinearStats, one_hot, select_ridge
from cerata.eval.decomposition import ExpertDump, decompose, p2_decomposition

ROOT = Path(__file__).resolve().parents[1]
D, C, M = 12, 6, 96


def _batches(k=3, n=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(n, D, generator=g), torch.randint(0, C, (n,), generator=g))
        for _ in range(k)
    ]


# -- the feature map ----------------------------------------------------------------


def test_random_projection_is_a_seeded_fixed_buffer():
    a, b, c = RandomProjection(D, M, seed=1), RandomProjection(D, M, seed=1), None
    c = RandomProjection(D, M, seed=2)
    z = torch.randn(4, D, dtype=torch.float64)
    assert torch.equal(a(z), b(z)) and a.digest == b.digest != c.digest
    assert a.parameter_count() == 0 and bool((a(z) >= 0).all())
    with pytest.raises(ValueError):
        a(torch.randn(2, D + 1))


def test_stats_with_feature_map_continual_equals_one_shot_and_forgets():
    fm = RandomProjection(D, M, seed=0)
    bs = _batches()
    cont = LinearStats(D, C, feature_map=fm)
    for i, (z, y) in enumerate(bs):
        cont.add(cont.contribution(f"e{i}", z, one_hot(y, C)))
    once = LinearStats(D, C, feature_map=fm)
    z_all, y_all = torch.cat([b[0] for b in bs]), torch.cat([b[1] for b in bs])
    once.add(once.contribution("all", z_all, one_hot(y_all, C)))
    assert float((cont.solve() - once.solve()).abs().max()) <= 1e-10
    assert cont.order_report()["pass"] and cont.recompute_report()["pass"]
    # An edit stores its raw keys (n x dim), not phi(K) or dA.
    c = cont._contrib["e1"]
    assert c.Z is not None and c.K is None and c.dA is None and c.Z.shape == (25, D)
    cont.remove("e1")
    never = LinearStats(D, C, feature_map=fm)
    for i in (0, 2):
        never.add(never.contribution(f"e{i}", *[bs[i][0], one_hot(bs[i][1], C)]))
    assert float((cont.solve() - never.solve()).abs().max()) <= 1e-10
    cont.remove("e0")
    cont.remove("e2")
    assert torch.equal(cont.A, cont.A0) and not bool(cont.B.any())


def test_stats_without_feature_map_keep_their_digest_and_layout():
    s = LinearStats(D, C)
    z, y = _batches(1)[0]
    s.add(s.contribution("e", z, one_hot(y, C)))
    c = s._contrib["e"]
    assert s.state_digest() == digest(s.A, s.B) and c.Z is None
    assert c.dA is not None  # n = 25 > d' = 13: the (dA, dB) form, as before
    t = LinearStats(D, C, feature_map=RandomProjection(D, M))
    t.add(t.contribution("e", z, one_hot(y, C)))
    assert t.d1 == M + 1 and t.state_digest() != digest(t.A, t.B)


def test_select_ridge_is_deterministic_and_on_the_grid():
    z, y = torch.cat([b[0] for b in _batches()]), torch.cat([b[1] for b in _batches()])
    fm = RandomProjection(D, M)
    a = select_ridge(z, y, C, feature_map=fm, seed=3)
    b = select_ridge(z, y, C, feature_map=fm, seed=3)
    assert a["ridge"] == b["ridge"] and a["ridge"] in a["grid"]
    assert len(a["val_mse"]) == len(a["grid"]) == 17


def test_palmoe_with_random_features_passes_every_guard():
    bs = _batches(3, n=30)
    m = Cerata(dim=D, num_classes=C, random_features=M, canary=torch.randn(20, D))
    recs = [m.write(Batch(z, y, task=i)) for i, (z, y) in enumerate(bs)]
    for r in recs:
        assert r.order_report["pass"] and r.reversibility_report["pass"]
        assert r.purity_report["pass"]
    assert m.stats.d1 == M + 1 and m.forget(recs[1].id)["pass"]
    assert m.predict(bs[0][0]).labels.shape == (30,)


# -- the decomposition ----------------------------------------------------------------


def test_p2_bound_reexports_the_package_function():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import p2_bound
    finally:
        sys.path.pop(0)
    assert p2_bound.decomposition is p2_decomposition


def _dump(N=300, T=4, per=3, seed=0, native=True):
    g = torch.Generator().manual_seed(seed)
    Cn = T * per
    toc = torch.arange(Cn) // per
    y = torch.randint(0, Cn, (N,), generator=g)
    pred = torch.randint(0, Cn, (N, T), generator=g)
    return ExpertDump(
        y=y,
        task_of_class=toc,
        expert_pred=pred,
        native_pred=pred[:, 0].clone() if native else None,
        meta={"method": "test"},
    )


def test_decomposition_identity_holds_for_any_bank():
    d = _dump()
    logits = torch.randn(d.y.numel(), d.task_of_class.numel())
    out = decompose(d, logits)
    dec = out["decomposition"]
    assert dec["identity_abs_error"] <= 1e-12 and dec["counts"]["r_not_tau"] == 0
    assert abs(out["system_acc"] - out["readout_acc"] - dec["P2_pooled"]) <= 1e-12


def test_a_perfect_owner_bank_rescues_everything_it_owns():
    d = _dump()
    true_task = d.task_of_class[d.y]
    d.expert_pred[torch.arange(d.y.numel()), true_task] = d.y  # right under its owner
    logits = torch.randn(d.y.numel(), d.task_of_class.numel())
    dec = decompose(d, logits)["decomposition"]
    assert dec["rho"] == 1.0 and dec["beta"] == 0.0
    assert abs(dec["P2_pooled"] - dec["P2_max"]) <= 1e-12


def test_expert_dump_round_trips_and_validates(tmp_path):
    d = _dump()
    d.save(tmp_path / "x.npz")
    e = ExpertDump.load(tmp_path / "x.npz")
    assert torch.equal(e.expert_pred, d.expert_pred) and e.meta == {"method": "test"}
    bad = _dump()
    bad.expert_pred = bad.expert_pred[:, :2]
    with pytest.raises(ValueError):
        bad.validate()


# -- the benchmark protocol ---------------------------------------------------------------


def test_class_order_mirrors_pilot():
    np.random.seed(1993)
    expected = np.random.permutation(100).tolist()
    assert class_order(100, True) == expected
    assert class_order(50, BENCHMARKS["vtab"].shuffle) == list(range(50))


def test_task_increments_mirror_pilot():
    assert task_increments(100, 10, 10) == [10] * 10
    assert task_increments(200, 20, 20) == [20] * 10
    assert task_increments(100, 50, 7) == [50] + [7] * 7 + [1]
    assert task_increments(50, 10, 10) == [10] * 5


def test_split_tasks_remaps_labels_and_the_test_permutation_matches():
    g = torch.Generator().manual_seed(0)
    Cn, Dn = 12, 5
    ytr, yte = torch.arange(Cn).repeat(4), torch.randint(0, Cn, (40,), generator=g)
    ztr, zte = torch.randn(ytr.numel(), Dn), torch.randn(yte.numel(), Dn)
    order = class_order(Cn, True)
    tasks = split_tasks((ztr, ytr), (zte, yte), order, 4, 4)
    assert [t["classes"] for t in tasks] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    for t in tasks:
        _, y = t["splits"]["train"]
        assert set(y.tolist()) == set(t["classes"])
    perm = stream_test_order(yte, order, 4, 4)
    z_cat = torch.cat([t["splits"]["test"][0] for t in tasks])
    assert torch.equal(z_cat, zte[perm])
    # the remapped label of a sample is its original class's position in the order
    assert order.index(int(yte[perm[0]])) == int(tasks[0]["splits"]["test"][1][0])


# -- the runner ------------------------------------------------------------------------


def test_runner_synthetic_cell_passes_its_vetoes():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import ptm_cil
    finally:
        sys.path.pop(0)
    args = argparse.Namespace(
        init_cls=5,
        increment=5,
        M=64,
        no_bank=False,
        bank_epochs=1,
        external_dir=None,
        api=True,
    )
    cell = ptm_cil.run_cell(
        ptm_cil.synthetic_cache(), "synthetic", "none", 0, args, "cpu"
    )
    v = ptm_cil.vetoes([cell])
    assert v["identity_pass"] and v["r_not_tau_pass"]
    assert v["api_pass"] and v["guards_pass"]
    assert set(cell["readouts"]) == {"ncm", "ridge", "rp"} and "pal_l3" in cell["banks"]


def test_pinned_splits_give_ten_tasks_and_five_for_vtab():
    for name, spec in BENCHMARKS.items():
        n = len(task_increments(spec.num_classes, spec.init_cls, spec.increment))
        assert n == (5 if name == "vtab" else 10), name


def test_task_increments_reject_non_positive_steps():
    for init_cls, inc in ((0, 10), (10, 0), (10, -5)):
        with pytest.raises(ValueError):
            task_increments(100, init_cls, inc)


def test_storage_counts_the_fixed_projection():
    fm = RandomProjection(D, M)
    s = LinearStats(D, C, feature_map=fm)
    rep = s.storage_bytes()
    assert rep["feature_map"] == D * M * 4 and rep["total"] >= rep["feature_map"]
    assert LinearStats(D, C).storage_bytes()["feature_map"] == 0


def test_sanity_veto_needs_a_reference_and_a_close_match():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import ptm_cil
    finally:
        sys.path.pop(0)
    cells = [
        {
            "benchmark": "cifar100",
            "backbone": "in21k_1k",
            "readouts": {"ncm": {"final": a}},
        }
        for a in (0.80, 0.82)
    ]
    assert not ptm_cil.sanity_veto(cells, None)["cifar100__in21k_1k"]["pass"]
    ok = ptm_cil.sanity_veto(cells, {"cifar100__in21k_1k": 0.815})
    far = ptm_cil.sanity_veto(cells, {"cifar100__in21k_1k": 0.85})
    assert ok["cifar100__in21k_1k"]["pass"] and not far["cifar100__in21k_1k"]["pass"]


def test_feasibility_veto_writes_a_veto_failed_record(tmp_path, monkeypatch):
    import json

    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import ptm_cil
    finally:
        sys.path.pop(0)
    cache = ptm_cil.synthetic_cache()
    (tmp_path / "cache").mkdir()
    torch.save(cache, tmp_path / "cache" / "cifar100__syn.pt")
    out = tmp_path / "study.json"
    monkeypatch.setattr(ptm_cil, "FEASIBILITY_S", 0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ptm_cil.py",
            "--benchmarks=cifar100",
            "--backbones=syn",
            f"--cache_dir={tmp_path / 'cache'}",
            "--seeds=0,1",
            "--init_cls=5",
            "--increment=5",
            "--M=32",
            "--no_bank",
            f"--out={out}",
        ],
    )
    with pytest.raises(SystemExit):
        ptm_cil.main()
    rec = json.loads(out.read_text())
    assert rec["status"] == "veto_failed" and not rec["vetoes"]["feasibility_pass"]
    assert rec["vetoes"]["feasibility"]["cells_done"] == 1


def test_prepare_ptm_data_reunpacks_a_replaced_archive(tmp_path):
    import shutil
    import subprocess

    def make(tag: str):
        src = tmp_path / f"src_{tag}"
        for split in ("train", "test"):
            for c in range(BENCHMARKS["vtab"].num_classes):
                d = src / "vtab" / split / f"c{c}"
                d.mkdir(parents=True)
                (d / f"{tag}.png").write_bytes(tag.encode())
        (tmp_path / "root" / "downloads").mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(tmp_path / "root" / "downloads" / "vtab"), "zip", src)

    def run():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "prepare_ptm_data.py"),
                f"--root={tmp_path / 'root'}",
                "--only=vtab",
                "--skip_published_md5",
            ],
            check=True,
            capture_output=True,
        )

    make("a")
    run()
    target = tmp_path / "root" / "vtab-cil" / "vtab" / "train" / "c0"
    assert (target / "a.png").exists()
    make("b")  # same file name, new content
    run()
    assert (target / "b.png").exists() and not (target / "a.png").exists()
    import json

    man = json.loads((tmp_path / "root" / "MANIFEST.json").read_text())
    import hashlib

    zip_path = tmp_path / "root" / "downloads" / "vtab.zip"
    assert (
        man["vtab"]["archive_sha256"]
        == hashlib.sha256(zip_path.read_bytes()).hexdigest()
    )
    assert man["vtab"]["archive_md5"] == hashlib.md5(zip_path.read_bytes()).hexdigest()
    assert man["vtab"]["md5_matches_published"] is None  # no sum published for it here


def test_prepare_ptm_data_rejects_an_archive_with_a_wrong_published_md5(tmp_path):
    import shutil
    import subprocess

    src = tmp_path / "src"
    for split in ("train", "test"):
        for c in range(BENCHMARKS["vtab"].num_classes):
            d = src / "vtab" / split / f"c{c}"
            d.mkdir(parents=True)
            (d / "x.png").write_bytes(b"x")
    (tmp_path / "root" / "downloads").mkdir(parents=True)
    shutil.make_archive(str(tmp_path / "root" / "downloads" / "vtab"), "zip", src)
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "prepare_ptm_data.py"),
            f"--root={tmp_path / 'root'}",
            "--only=vtab",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0 and "md5 MISMATCH" in res.stdout
    assert not (tmp_path / "root" / "vtab-cil").exists()  # nothing was unpacked
