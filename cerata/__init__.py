"""CERATA: closed-form, exactly reversible, auditable learning after deployment.

Named after the *tabula cerata*, the wax tablet that is written on and wiped clean
without a trace. A frozen base provides a fixed address space; learning is an API call
on three time scales (a key-value memory, a closed-form edit, a frozen expert bank),
and every `write` / `forget` returns measured locality, reversibility and
order-invariance reports:

    from cerata.api import Cerata, Batch, Example

The project was called PAL-MoE until 2026-09-26; the v1 design of that name lives in
`cerata.legacy`, and the `pal_moe` package keeps the old import paths working.
"""

from .api import (
    Batch,
    Cerata,
    ConsolidationReport,
    EditRecord,
    Example,
    GuardConfig,
    Prediction,
)

__version__ = "0.2.0"

__all__ = [
    "Batch",
    "Cerata",
    "ConsolidationReport",
    "EditRecord",
    "Example",
    "GuardConfig",
    "Prediction",
]
