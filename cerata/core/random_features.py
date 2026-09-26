"""A fixed random feature map for the MEDIUM path (the RanPAC projection).

`phi(z) = relu(z @ W)`, `W ~ N(0, 1)` of shape `[in_dim, out_dim]`, drawn once from a
seeded CPU generator (McDonnell et al., "RanPAC", NeurIPS 2023; their M = 10000). It is
part of the frozen key pipeline, not a learned component:

- `W` is a buffer, never optimised, and content-hashed into `digest`, so a state
  that used one projection cannot be confused with a state that used another;
- `phi` is deterministic for a given device and dtype, so an edit can store its raw
  keys `z` (n x in_dim) and recompute `phi(z)` when it is forgotten: the subtraction
  removes the same values that were added (`LinearStats` does this).

RanPAC re-selects the ridge penalty on every task from that task's data. That makes
the solution depend on task order and makes a removal inexact, so here the penalty is
chosen once, before the stream (`cerata.edit.stats.select_ridge`), and pinned.
"""

from __future__ import annotations

import torch

from .hashing import digest

ACTIVATIONS = {
    "relu": torch.relu,
    "none": lambda x: x,
}


class RandomProjection:
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 10000,
        seed: int = 0,
        activation: str = "relu",
    ):
        if activation not in ACTIVATIONS:
            raise KeyError(f"unknown activation {activation!r}")
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.seed, self.activation = int(seed), activation
        g = torch.Generator().manual_seed(self.seed)
        self.W = torch.randn(self.in_dim, self.out_dim, generator=g)  # float32, CPU
        self._cast: dict[tuple, torch.Tensor] = {}
        self.digest = digest(
            "random-projection",
            self.in_dim,
            self.out_dim,
            self.seed,
            self.activation,
            self.W,
        )

    def _weight(self, device, dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if key not in self._cast:
            self._cast[key] = self.W.to(device=device, dtype=dtype)
        return self._cast[key]

    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(-1) != self.in_dim:
            raise ValueError(f"expected keys of dim {self.in_dim}, got {z.size(-1)}")
        return ACTIVATIONS[self.activation](z @ self._weight(z.device, z.dtype))

    def parameter_count(self) -> int:
        return 0  # a fixed buffer
