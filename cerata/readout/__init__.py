"""Readouts: the existing S1 registry (`ncm`, `cosine`, `linear`, `logistic`, `ridge`, `mlp`).

The implementations stay in `cerata/arch/readouts.py`: `cerata.arch` registers them
at import time through `cerata.arch.registry`, and moving the file would put a
package-init cycle between `arch` and `readout`. This package is the v3 name for the
same objects (identity, not copies): `cerata.readout.RidgeReadout is
cerata.arch.RidgeReadout`.

For the v3 medium path use `cerata.edit.LinearStats` (float64, subtractable); the
float32 `RidgeReadout` is kept for bitwise reproduction of every stored result.
"""

from cerata.arch.readouts import (
    CosineReadout,
    LinearReadout,
    LogisticReadout,
    MLPReadout,
    NCMReadout,
    RidgeReadout,
    mask_unseen,
)
from cerata.arch.registry import READOUTS, build_readout

__all__ = [
    "READOUTS",
    "CosineReadout",
    "LinearReadout",
    "LogisticReadout",
    "MLPReadout",
    "NCMReadout",
    "RidgeReadout",
    "build_readout",
    "mask_unseen",
]
