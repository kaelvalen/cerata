"""PTM-CIL amendment 4: the lambda choice must not depend on the feature scale."""

import math

import torch

from cerata.core.random_features import RandomProjection
from cerata.edit.stats import LinearStats, one_hot, select_ridge


def _data(n=600, d=32, C=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(C, d, generator=g) * 2
    y = torch.randint(0, C, (n,), generator=g)
    return centers[y] + torch.randn(n, d, generator=g), y


def _fitted(z, y, info, feature_map=None, dim=32, C=5):
    s = LinearStats(
        dim,
        C,
        ridge=info["ridge"],
        bias_ridge=info["bias_ridge"],
        feature_map=feature_map,
    )
    s.add(s.contribution("t", z, one_hot(y, C)))
    return s


def test_power_of_two_rescaling_returns_the_same_model():
    # Callers re-solve in the raw scale (LinearStats), so predictions agree to
    # fp rounding, not bitwise; the choice itself is exactly reproducible.
    z, y = _data()
    a = select_ridge(z, y, 5, seed=0)
    b = select_ridge(z * 1024.0, y, 5, seed=0)
    assert a["c"] == b["c"]
    assert b["ridge"] == a["ridge"] * 1024.0**2
    assert b["bias_ridge"] == a["bias_ridge"]
    pa = _fitted(z, y, a).predict(z)
    pb = _fitted(z * 1024.0, y, b).predict(z * 1024.0)
    assert torch.equal(pa.argmax(-1), pb.argmax(-1))
    assert torch.allclose(pa, pb, rtol=1e-9, atol=1e-12)


def test_times_100_keeps_the_choice_and_the_decisions():
    z, y = _data()
    for feature_map in (None, RandomProjection(32, 400, seed=1)):
        a = select_ridge(z, y, 5, seed=0, feature_map=feature_map)
        b = select_ridge(z * 100.0, y, 5, seed=0, feature_map=feature_map)
        assert a["c"] == b["c"]
        assert math.isclose(a["bias_ridge"], b["bias_ridge"], rel_tol=1e-9)
        pa = _fitted(z, y, a, feature_map=feature_map).predict(z)
        pb = _fitted(z * 100.0, y, b, feature_map=feature_map).predict(z * 100.0)
        assert torch.equal(pa.argmax(-1), pb.argmax(-1))


def test_flat_curve_takes_the_largest_c_and_flags_an_edge():
    z, _ = _data()
    Y = torch.zeros(z.size(0), 5, dtype=torch.float64)  # every c has the same MSE
    info = select_ridge(z, Y=Y, seed=0)
    assert info["extended"] == "up" and not info["converged"]
    assert info["c"] == max(info["grid"]) == 10.0**4


def test_regression_target_matrix_is_accepted():
    z, _ = _data()
    g = torch.Generator().manual_seed(1)
    Y = torch.rand(z.size(0), 1, generator=g)
    info = select_ridge(z, Y=Y, seed=0)
    assert info["ridge"] > 0 and info["bias_ridge"] > 0 and info["c"] in info["grid"]
    assert len(info["val_mse"]) == len(info["grid"])
