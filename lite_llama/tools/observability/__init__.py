"""Observability tools: measuring what the engine does, without changing what it does.

Instrumentation lives here rather than beside the code it measures, so a subsystem
carries no reporting machinery of its own and the cost of not looking stays at zero.
Today that is collective traffic; the shape generalises to anything a run wants to
account for.

Usage:
    from lite_llama.tools.observability import Collective, CollectiveStats
    with CollectiveStats.collect() as stats:
        engine.step()
    print(stats.report())
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
