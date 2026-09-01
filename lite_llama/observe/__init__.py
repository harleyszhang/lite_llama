"""Runtime observability: request metrics, and optional OTLP tracing.

Re-exports :class:`EngineMetrics` and :class:`Tracer`; both degrade to
no-ops unless their environment switches opt in, so observability never
costs anything by default.

Usage:
    from lite_llama.observe import EngineMetrics, Tracer
"""

from .metrics import METRICS_ENV, EngineMetrics
from .trace import OTLP_ENDPOINT_ENV, Tracer

__all__ = [
    "METRICS_ENV",
    "OTLP_ENDPOINT_ENV",
    "EngineMetrics",
    "Tracer",
]
