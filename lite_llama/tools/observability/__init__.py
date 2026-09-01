"""Observability tools: measure what the engine does, without changing it.

Re-exports :class:`CollectiveStats` — the wire accounting for tensor
parallelism — plus its enums and byte-formatting helpers.

Usage:
    from lite_llama.tools.observability import CollectiveStats
"""

from .collective_stats import (
    Collective,
    CollectiveStats,
    Plane,
    Tally,
    human_bytes,
)

__all__ = [
    "Collective",
    "CollectiveStats",
    "Plane",
    "Tally",
    "human_bytes",
]
