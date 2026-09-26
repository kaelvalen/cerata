"""The v3 public API: learning is an API call, not a training run.

model = Cerata(dim=768, num_classes=100, router="ridge_class", canary=canary_feats)
rec = model.write(Batch(z, y, task=0))     # MEDIUM: closed-form, float64, reversible
rec = model.write(Example(z1, 7))          # FAST: one memory row
model.forget(rec.id)                       # exact
model.consolidate("by_arrival")            # SLOW: frozen experts
model.predict(z_test)                      # routed expert ids + scores
model.state()                              # (base_hash, ordered edit log)
"""

from .facade import Cerata
from .guards import GuardConfig, GuardedEditor
from .records import (
    Batch,
    ConsolidationReport,
    EditRecord,
    Example,
    GuardViolation,
    Prediction,
    ReversibilityError,
    StateHash,
)

# The facade's name before the 2026-09-26 rename; the same class object.
PalMoE = Cerata

__all__ = [
    "Batch",
    "ConsolidationReport",
    "EditRecord",
    "Example",
    "GuardConfig",
    "GuardViolation",
    "GuardedEditor",
    "Cerata",
    "PalMoE",
    "Prediction",
    "ReversibilityError",
    "StateHash",
]
