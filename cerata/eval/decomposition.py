"""The P2 decomposition: what an expert bank adds on top of a readout router.

Per test sample, with a readout that predicts a class and routes to the task that owns
it, and a bank that is asked for a prediction under that routed task:

    r    the readout's class is right
    tau  the routed task is right (the owner of the true class)
    s    the system (bank under the routed task) is right

    P2 = P(s) - P(r) = m * rho + P(not r, not tau) * rho' - P(r) * beta

    m     = P(not r, tau)          rescuable mass: right task, wrong class
    rho   = P(s | not r, tau)      the rescue rate on that mass
    rho'  = P(s | not r, not tau)  rescues under a wrong route (normally ~0)
    beta  = P(not s | r)           samples the readout had right and the bank breaks

`p2_decomposition` is the P2-BOUND function (`docs/P2_BOUND_PREREG.md`), moved here
unchanged from `experiments/p2_bound.py`, which re-exports it.

Any bank can be decomposed, not only this repository's: `ExpertDump` is the exchange
format for another method's bank (EASE, MOS, MoTE, ...): its prediction under every
forced expert/task, per test sample. `decompose` joins it with a readout's logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def p2_decomposition(r, tau, s) -> dict:
    """Counts -> the P2 identity terms. r, tau, s: bool tensors over all test samples."""
    N = r.numel()
    nr, ntau = ~r, ~tau
    c = {
        "N": N,
        "r": int(r.sum()),
        "s": int(s.sum()),
        "r_not_tau": int((r & ntau).sum()),  # must be 0: a right class implies its task
        "nr_tau": int((nr & tau).sum()),
        "nr_ntau": int((nr & ntau).sum()),
        "s_nr_tau": int((s & nr & tau).sum()),
        "s_nr_ntau": int((s & nr & ntau).sum()),
        "ns_r": int((~s & r).sum()),
    }
    m = c["nr_tau"] / N
    rho = c["s_nr_tau"] / c["nr_tau"] if c["nr_tau"] else float("nan")
    rho_p = c["s_nr_ntau"] / c["nr_ntau"] if c["nr_ntau"] else 0.0
    p_r = c["r"] / N
    beta = c["ns_r"] / c["r"] if c["r"] else 0.0
    p2 = (c["s"] - c["r"]) / N
    ident = m * (rho if c["nr_tau"] else 0.0) + (c["nr_ntau"] / N) * rho_p - p_r * beta
    return {
        "counts": c,
        "m": m,
        "rho": rho,
        "rho_prime": rho_p,
        "beta": beta,
        "p_r": p_r,
        "rescue": m * rho if c["nr_tau"] else 0.0,
        "break": p_r * beta,
        "P2_pooled": p2,
        "P2_max": m + (c["nr_ntau"] / N) * rho_p,
        "identity_abs_error": abs(ident - p2),
    }


def route_by_owner(class_logits: torch.Tensor, task_of_class: torch.Tensor):
    """The ridge_class rule: the routed task is the owner of the argmax class."""
    return task_of_class[class_logits.argmax(-1)]


@dataclass
class ExpertDump:
    """Another method's bank, evaluated under every forced expert.

    y              [N]     true labels (in the stream's class indexing)
    task_of_class  [C]     owning task of every class
    expert_pred    [N, T]  predicted label when the bank is forced to expert/task t
    native_pred    [N]     optional: the bank's own prediction with its own routing
    meta           free-form provenance (method, commit, config, seed)

    Stored as `.npz` with those keys (`meta` as a JSON string).
    """

    y: torch.Tensor
    task_of_class: torch.Tensor
    expert_pred: torch.Tensor
    native_pred: torch.Tensor | None = None
    meta: dict | None = None

    def validate(self) -> None:
        N, T = self.expert_pred.shape
        if self.y.shape != (N,):
            raise ValueError(f"y has shape {tuple(self.y.shape)}, expected ({N},)")
        if int(self.task_of_class.max()) + 1 != T:
            raise ValueError("expert_pred must have one column per task")
        if self.native_pred is not None and self.native_pred.shape != (N,):
            raise ValueError("native_pred must have one entry per sample")
        C = self.task_of_class.numel()
        if int(self.y.max()) >= C or int(self.expert_pred.max()) >= C:
            raise ValueError("labels exceed the number of classes in task_of_class")

    def save(self, path: str | Path) -> None:
        import json

        arrays = {
            "y": self.y.cpu().numpy(),
            "task_of_class": self.task_of_class.cpu().numpy(),
            "expert_pred": self.expert_pred.cpu().numpy(),
            "meta": np.array(json.dumps(self.meta or {})),
        }
        if self.native_pred is not None:
            arrays["native_pred"] = self.native_pred.cpu().numpy()
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ExpertDump:
        import json

        with np.load(path, allow_pickle=False) as f:
            d = cls(
                y=torch.from_numpy(f["y"]).long(),
                task_of_class=torch.from_numpy(f["task_of_class"]).long(),
                expert_pred=torch.from_numpy(f["expert_pred"]).long(),
                native_pred=(
                    torch.from_numpy(f["native_pred"]).long()
                    if "native_pred" in f.files
                    else None
                ),
                meta=json.loads(str(f["meta"])) if "meta" in f.files else {},
            )
        d.validate()
        return d


def decompose(dump: ExpertDump, readout_logits: torch.Tensor) -> dict:
    """P2 of `dump`'s bank routed by a readout, plus the accuracies around it."""
    dump.validate()
    y = dump.y
    toc = dump.task_of_class.to(readout_logits.device)
    readout_pred = readout_logits.argmax(-1).cpu()
    routed = route_by_owner(readout_logits, toc).cpu()
    true_task = dump.task_of_class[y]
    system = dump.expert_pred.gather(1, routed.unsqueeze(1)).squeeze(1)
    oracle = dump.expert_pred.gather(1, true_task.unsqueeze(1)).squeeze(1)
    r, tau, s = readout_pred == y, routed == true_task, system == y
    out = {
        "readout_acc": float(r.double().mean()),
        "routed_acc": float(tau.double().mean()),
        "system_acc": float(s.double().mean()),
        "oracle_acc": float((oracle == y).double().mean()),
        "decomposition": p2_decomposition(r, tau, s),
    }
    if dump.native_pred is not None:
        out["native_acc"] = float((dump.native_pred == y).double().mean())
    return out
