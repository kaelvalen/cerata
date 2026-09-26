"""Memory: the v3 FAST path, an append-only key-value store with exact delete.

The v1 prototype memory lives in `cerata.legacy.memory`.
"""

from .kv_store import FastMemory, MemoryHit

__all__ = ["FastMemory", "MemoryHit"]
