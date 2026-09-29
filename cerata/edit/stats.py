"""MEDIUM path: closed-form linear edits from additive float64 sufficient statistics.

For a linear map fitted by ridge regression, `W = B^T (lambda I + A)^-1` with
`A = sum K^T K` and `B = sum K^T V`. Both sums are additive over edits, so

- learning a batch is adding its contribution to ONE accumulator (closed form),
- the result does not depend on the order batches arrived in (up to fp rounding),
- forgetting a batch is subtracting its contribution (a downdate: exact unlearning in
  exact arithmetic, fp-rounding-close in float64).

Storage. The accumulators are one `[d', d']` and one `[d', m]` float64 matrix,
independent of the number of edits. To be able to subtract an edit later, each edit
keeps the *smaller* of its two representations: the raw factors `(K, V)` when
`n <= d'` (a single fact, a small batch: `n x d'`), else its `(dA, dB)` (`d' x d'`).
Recomputing `dA = K^T K` from stored factors is deterministic, so the subtraction
removes exactly the bits that were added.

What the guards measure (they are measurements, not consequences of a design
choice): `order_report` re-sums the live contributions in a seeded random
permutation and reports max|dW| against the accumulator; `recompute_report` re-sums
them in arrival order and reports max|dW| after downdates. Both are compared with a
tolerance. Bitwise equality is only claimed where it is real: when the last edit is
forgotten the accumulators are reset to `(lambda I, 0)` exactly.

E-TID2 (G3) measured max|dW| = 1.26e-3 between continual and one-shot ridge in
float32; hence float64.

Feature maps. `feature_map` (e.g. `cerata.core.random_features.RandomProjection`)
fits the ridge on `phi(K)` instead of `K`. An edit then stores its raw keys `Z`
(`n x dim`, far smaller than `phi(Z)` or `dA` when `phi` expands to 10^4 dimensions)
and `delta()` recomputes `phi(Z)` deterministically, so a removal subtracts the same
values the addition added. With `feature_map=None` nothing here changes: the stored
results that go through this class reproduce bitwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cerata.core.hashing import digest


@dataclass
class Contribution:
    edit_id: str
    content_hash: str
    n: int
    K: torch.Tensor | None = None  # [n, d'] augmented keys, float64 (when n <= d')
    V: torch.Tensor | None = None  # [n, m] float64
    dA: torch.Tensor | None = None  # [d', d'] (when n > d')
    dB: torch.Tensor | None = None  # [d', m]
    Z: torch.Tensor | None = None  # [n, dim] raw keys, float64 (with a feature map)

    def delta(self, augment=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dA is not None:
            return self.dA, self.dB
        if self.Z is not None:
            K = augment(self.Z)
            return K.t() @ K, K.t() @ self.V
        return self.K.t() @ self.K, self.K.t() @ self.V

    def nbytes(self) -> int:
        ts = [t for t in (self.K, self.V, self.dA, self.dB, self.Z) if t is not None]
        return sum(t.numel() * t.element_size() for t in ts)


class LinearStats:
    """Additive statistics of a ridge-fitted linear map `K (d) -> V (m)`.

    `bias=True` appends a ones column (intercept solved jointly), matching
    `cerata.arch.readouts.RidgeReadout` term for term: `A_0 = lambda I` on the
    augmented dimension. `bias_ridge` overrides the penalty on that one coordinate
    (default: `ridge`); the scale-free selector (`select_ridge`) returns the pair, so
    a fit on rescaled features maps to the same model.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        ridge: float = 1.0,
        bias: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        feature_map=None,
        bias_ridge: float | None = None,
    ):
        self.dim, self.out_dim = int(dim), int(out_dim)
        self.ridge, self.bias = float(ridge), bool(bias)
        self.bias_ridge = float(self.ridge if bias_ridge is None else bias_ridge)
        self.device, self.dtype = torch.device(device), dtype
        self.feature_map = feature_map
        if feature_map is not None and int(feature_map.in_dim) != self.dim:
            raise ValueError("feature_map.in_dim must equal dim")
        feat = self.dim if feature_map is None else int(feature_map.out_dim)
        self.d1 = feat + int(self.bias)
        pen = torch.full((self.d1,), self.ridge, device=self.device, dtype=dtype)
        if self.bias:
            pen[-1] = self.bias_ridge
        self.A0 = torch.diag(pen)
        self.A = self.A0.clone()
        self.B = torch.zeros(self.d1, self.out_dim, device=self.device, dtype=dtype)
        self._contrib: dict[str, Contribution] = {}
        self._arrival: list[str] = []
        self._W: torch.Tensor | None = None

    # -- contributions ---------------------------------------------------

    def _augment(self, k: torch.Tensor) -> torch.Tensor:
        k = k.to(self.device, self.dtype)
        if self.feature_map is not None:
            k = self.feature_map(k)
        if not self.bias:
            return k
        ones = torch.ones(k.size(0), 1, device=k.device, dtype=k.dtype)
        return torch.cat([k, ones], dim=1)

    @torch.no_grad()
    def contribution(
        self, edit_id: str, K: torch.Tensor, V: torch.Tensor
    ) -> Contribution:
        h = digest(K.detach(), V.detach())
        if self.feature_map is not None:
            Z = K.detach().reshape(-1, self.dim).to(self.device, self.dtype)
            V64 = V.detach().to(self.device, self.dtype).reshape(Z.size(0), -1)
            return Contribution(edit_id, h, Z.size(0), V=V64, Z=Z)
        Ka = self._augment(K.detach().reshape(-1, self.dim))
        V64 = V.detach().to(self.device, self.dtype).reshape(Ka.size(0), self.out_dim)
        if Ka.size(0) <= self.d1:
            return Contribution(edit_id, h, Ka.size(0), K=Ka, V=V64)
        return Contribution(edit_id, h, Ka.size(0), dA=Ka.t() @ Ka, dB=Ka.t() @ V64)

    @torch.no_grad()
    def add(self, c: Contribution) -> None:
        if c.edit_id in self._contrib:
            raise KeyError(f"duplicate edit id {c.edit_id!r}")
        dA, dB = c.delta(self._augment)
        self.A += dA
        self.B += dB
        self._contrib[c.edit_id] = c
        self._arrival.append(c.edit_id)
        self._W = None

    @torch.no_grad()
    def remove(self, edit_id: str) -> Contribution:
        """Downdate: subtract the edit's contribution from the accumulators."""
        c = self._contrib.pop(edit_id)
        self._arrival.remove(edit_id)
        if not self._contrib:  # the only exact statement available: back to the prior
            self.A = self.A0.clone()
            self.B.zero_()
        else:
            dA, dB = c.delta(self._augment)
            self.A -= dA
            self.B -= dB
        self._W = None
        return c

    def __contains__(self, edit_id: str) -> bool:
        return edit_id in self._contrib

    @property
    def edit_ids(self) -> list[str]:
        return list(self._arrival)

    @property
    def n_samples(self) -> int:
        return sum(c.n for c in self._contrib.values())

    def storage_bytes(self) -> dict:
        acc = (self.A.numel() + self.B.numel()) * self.A.element_size()
        per = sum(c.nbytes() for c in self._contrib.values())
        fm = 0 if self.feature_map is None else int(self.feature_map.nbytes())
        return {
            "accumulators": acc,
            "per_edit_total": per,
            "feature_map": fm,  # fixed, needed at inference; 0 trainable parameters
            "total": acc + per + fm,
            "n_edits": len(self._contrib),
        }

    # -- solutions -------------------------------------------------------

    @torch.no_grad()
    def solve(self) -> torch.Tensor:
        """`W` [m, d'] from the accumulators (cached until the next add/remove)."""
        if self._W is None:
            self._W = torch.linalg.solve(self.A, self.B).t().contiguous()
        return self._W

    @torch.no_grad()
    def resum(self, order: list[str]) -> torch.Tensor:
        """Reference solution: the live contributions re-summed in `order`."""
        A, B = self.A0.clone(), torch.zeros_like(self.B)
        for eid in order:
            dA, dB = self._contrib[eid].delta(self._augment)
            A += dA
            B += dB
        return torch.linalg.solve(A, B).t().contiguous()

    @torch.no_grad()
    def predict(self, K: torch.Tensor) -> torch.Tensor:
        return self._augment(K.reshape(-1, self.dim)) @ self.solve().t()

    def state_digest(self) -> str:
        if self.feature_map is None:
            return digest(self.A, self.B)
        return digest(self.A, self.B, self.feature_map.digest)

    def _argmax_identical(self, Wa, Wb, canary) -> bool | None:
        if canary is None:
            return None
        Ka = self._augment(canary.reshape(-1, self.dim))
        return bool(torch.equal((Ka @ Wa.t()).argmax(-1), (Ka @ Wb.t()).argmax(-1)))

    @torch.no_grad()
    def order_report(self, canary=None, tol: float = 1e-10, seed: int = 0) -> dict:
        """Accumulator vs the live contributions re-summed in a random permutation."""
        n = len(self._arrival)
        if n == 0:
            return {"n_edits": 0, "max_abs_dW_permutation": 0.0, "pass": True}
        g = torch.Generator().manual_seed(seed + n)
        perm = [self._arrival[i] for i in torch.randperm(n, generator=g).tolist()]
        Wp = self.resum(perm)
        W = self.solve()
        rep = {
            "n_edits": n,
            "permutation": perm if n <= 32 else None,
            "max_abs_dW_permutation": float((W - Wp).abs().max()),
            "tolerance": tol,
            "argmax_identical": self._argmax_identical(W, Wp, canary),
        }
        rep["pass"] = (
            rep["max_abs_dW_permutation"] <= tol
            and rep["argmax_identical"] is not False
        )
        return rep

    @torch.no_grad()
    def recompute_report(self, canary=None, tol: float = 1e-10) -> dict:
        """Accumulator (after any downdates) vs the live contributions re-summed."""
        if not self._arrival:
            exact = torch.equal(self.A, self.A0) and not bool(self.B.any())
            return {
                "n_edits": 0,
                "max_abs_dW_recompute": 0.0,
                "bitwise_prior": exact,
                "pass": exact,
            }
        Wr = self.resum(self._arrival)
        W = self.solve()
        rep = {
            "n_edits": len(self._arrival),
            "max_abs_dW_recompute": float((W - Wr).abs().max()),
            "tolerance": tol,
            "argmax_identical": self._argmax_identical(W, Wr, canary),
        }
        rep["pass"] = (
            rep["max_abs_dW_recompute"] <= tol and rep["argmax_identical"] is not False
        )
        return rep

    def parameter_count(self) -> int:
        return 0  # statistics are buffers, never optimised


# Scale-free c-grid of PTM-CIL amendment 4: lambda = c * trace(A_feat)/d on features
# normalised to unit mean squared norm. Ties go to the LARGEST c within `TIE_TOL`
# (relative) of the best held-out MSE; an edge choice extends the grid once.
SCALE_FREE_GRID = [10.0**k for k in range(-6, 2)]
EDGE_EXTEND = 3
TIE_TOL = 1e-3


def _largest_within(scores: dict, tie_tol: float) -> float:
    """The largest `c` whose held-out loss is within `tie_tol` (relative) of the best."""
    best = min(scores.values())
    thr = best + tie_tol * abs(best) + 1e-12
    return max(c for c, v in scores.items() if v <= thr)


@torch.no_grad()
def select_ridge(
    Z: torch.Tensor,
    y: torch.Tensor | None = None,
    num_classes: int | None = None,
    feature_map=None,
    grid=None,
    frac: float = 0.8,
    seed: int = 0,
    bias: bool = True,
    device: str | torch.device = "cpu",
    Y: torch.Tensor | None = None,
    tie_tol: float = TIE_TOL,
) -> dict:
    """Pick the ridge penalty once, on data available before the stream is fixed.

    PTM-CIL amendment 4 (a port of the v3-restructure amendment 3): the fixed
    `10^-8 .. 10^8` grid is retired, because it is not scale-free. On a wide `phi`
    (a 10^4-d random projection) every point of it was negligible - a flat held-out
    curve, 100 % train accuracy - and ties slid to the smallest lambda, silently
    under-regularising the frozen baselines. Penalties are now chosen relative to
    the feature scale, `lambda = c * trace(A_feat)/d`, on features normalised to
    unit mean squared norm; ties go to the LARGEST `c` within `tie_tol` of the best
    held-out MSE; an edge choice extends the grid once by three decades and is
    flagged `converged = False` if it stays an edge. Samples are split by a seeded
    permutation, not by loader order. `Y` passes an explicit target matrix
    (regression included); otherwise `one_hot(y, num_classes)` is used. Returns the
    penalties of the RAW feature scale - `ridge` for the feature block and
    `bias_ridge` for the appended intercept - so `LinearStats(ridge=...,
    bias_ridge=...)` reproduces the normalised-space fit; `c`, `scale`, `unit`, the
    evaluated `grid`/`val_mse` and the edge `converged` flag are reported too.
    """
    Z = torch.as_tensor(Z)
    if Z.dim() != 2:
        raise ValueError("Z must be [n, dim]")
    if Y is None:
        if y is None or num_classes is None:
            raise ValueError("select_ridge needs Y or (y, num_classes)")
        Y = one_hot(torch.as_tensor(y).reshape(-1), int(num_classes))
    Y = torch.as_tensor(Y, dtype=torch.float64).reshape(Z.size(0), -1).to(device)
    if feature_map is not None and int(feature_map.in_dim) != Z.size(1):
        raise ValueError("feature_map.in_dim must equal Z.size(1)")
    K = (
        feature_map(Z.to(device, torch.float64))
        if feature_map is not None
        else Z.to(device, torch.float64)
    )
    K = K.to(torch.float64)
    d = K.size(1) - int(bias)
    if bias:
        K = torch.cat(
            [K, torch.ones(K.size(0), 1, dtype=K.dtype, device=K.device)], dim=1
        )
    scale = (float((K[:, :d] ** 2).sum()) / max(1, K.size(0))) ** 0.5 or 1.0
    H = torch.cat([K[:, :d] / scale, K[:, d:]], dim=1)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(Z.size(0), generator=g)
    cut = int(Z.size(0) * frac)
    if not 0 < cut < Z.size(0):
        raise ValueError("frac must leave a fit and a held-out split")
    tr, va = perm[:cut], perm[cut:]
    Htr, Hva, Ytr, Yva = H[tr], H[va], Y[tr], Y[va]
    A, Q = Htr.t() @ Htr, Htr.t() @ Ytr
    unit = float(torch.diagonal(A)[:d].sum()) / d
    eye = torch.eye(H.size(1), dtype=torch.float64, device=device)
    scores: dict[float, float] = {}

    def heldout_mse(c: float) -> float:
        L = torch.linalg.cholesky(A + c * unit * eye)
        W = torch.cholesky_solve(Q, L)
        return float(((Hva @ W - Yva) ** 2).mean())

    grid = list(SCALE_FREE_GRID if grid is None else grid)
    for c in grid:
        scores[c] = heldout_mse(c)
    c = _largest_within(scores, tie_tol)
    extended = None
    if c in (min(grid), max(grid)):
        step = 10.0 if c == max(grid) else 0.1
        for i in range(1, EDGE_EXTEND + 1):
            scores[c * step**i] = heldout_mse(c * step**i)
        extended = "up" if step > 1 else "down"
        c = _largest_within(scores, tie_tol)
    all_c = sorted(scores)
    return {
        "ridge": c * unit * scale * scale,
        "bias_ridge": c * unit,
        "c": c,
        "scale": scale,
        "unit": unit,
        "grid": all_c,
        "val_mse": [scores[x] for x in all_c],
        "extended": extended,
        "converged": c not in (all_c[0], all_c[-1]),
        "seed": seed,
    }


def one_hot(y: torch.Tensor, num_classes: int, dtype=torch.float64) -> torch.Tensor:
    out = torch.zeros(y.numel(), num_classes, dtype=dtype, device=y.device)
    out[torch.arange(y.numel(), device=y.device), y.reshape(-1).long()] = 1.0
    return out
