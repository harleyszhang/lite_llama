"""Runtime observability: request metrics, and optional OTLP tracing.

* :mod:`~lite_llama.observe.metrics` — the per-request numbers (queue time,
  TTFT, TPOT, token counts) rendered as Prometheus text for ``/metrics``.
* :mod:`~lite_llama.observe.trace` — one OTLP span per request when a
  collector is configured, a no-op otherwise.

Both default to cheap: metrics are a few float additions on the finish path,
and tracing without an endpoint is a ``None`` check.
"""

from .metrics import METRICS_ENV, EngineMetrics
from .trace import OTLP_ENDPOINT_ENV, Tracer

__all__ = [
    "METRICS_ENV",
    "OTLP_ENDPOINT_ENV",
    "EngineMetrics",
    "Tracer",
]
