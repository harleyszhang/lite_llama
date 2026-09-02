"""Observability tools: measure what the engine does, without changing it.

Three instruments live here:

* :class:`CollectiveStats` — the wire accounting for tensor parallelism,
  with its enums and byte-formatting helpers.
* :class:`EngineMetrics` (``metrics.py``) — request-level counters, gauges
  and histograms rendering Prometheus text; opt-out via
  ``LITE_LLAMA_METRICS=0``.
* :class:`Tracer` (``trace.py``) — one OTLP span per request when
  ``LITE_LLAMA_OTLP_ENDPOINT`` is set, a cheap no-op otherwise.

Usage:
    from lite_llama.tools.observability import CollectiveStats, EngineMetrics, Tracer
"""

from .collective_stats import (
    Collective,
    CollectiveStats,
    Plane,
    Tally,
    human_bytes,
)
from .metrics import METRICS_ENV, EngineMetrics
from .trace import OTLP_ENDPOINT_ENV, Tracer

__all__ = [
    "METRICS_ENV",
    "OTLP_ENDPOINT_ENV",
    "Collective",
    "CollectiveStats",
    "EngineMetrics",
    "Plane",
    "Tally",
    "Tracer",
    "human_bytes",
]
