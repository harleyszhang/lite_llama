"""Optional OTLP tracing: one span per request, exported over OTLP/HTTP.

A request's span opens when it enters the scheduler and closes when the engine
retires it, carrying the numbers a trace backend is good at slicing by:
prompt/output lengths, finish reason and the queue/prefill/decode breakdown.
Step-level spans would multiply volume by the decode length for information the
metrics already aggregate, so they are deliberately absent.

OpenTelemetry is an optional dependency: with it installed and
``LITE_LLAMA_OTLP_ENDPOINT`` set, spans export to that collector; without
either, every call here is a no-op costing one ``None`` check. That mirrors
how the serving extras work — tracing must never make offline generation
heavier to install.

Usage:
    tracer = Tracer.from_env()
    span = tracer.start_span("request", request_id="req-0", prompt_tokens=12)
    tracer.end_span(span, finish_reason="eos", output_tokens=42)
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Environment variable carrying the OTLP/HTTP collector endpoint.
OTLP_ENDPOINT_ENV = "LITE_LLAMA_OTLP_ENDPOINT"


class _Span(Protocol):
    """The slice of an OpenTelemetry span the engine uses."""

    def set_attribute(self, key: str, value: Any) -> None: ...

    def end(self) -> None: ...


class Tracer:
    """Starts and ends request spans; a no-op unless configured.

    Args:
        otel_tracer: An OpenTelemetry tracer. ``None`` disables everything —
            :meth:`start_span` returns ``None`` and :meth:`end_span` drops it,
            so the engine needs no branches of its own.
    """

    def __init__(self, otel_tracer: Any | None = None) -> None:
        self._otel = otel_tracer

    @property
    def enabled(self) -> bool:
        return self._otel is not None

    @classmethod
    def from_env(cls, service_name: str = "lite_llama") -> Tracer:
        """Build from ``LITE_LLAMA_OTLP_ENDPOINT``; missing pieces mean a no-op.

        An unreachable collector must not take the engine down either, so the
        OpenTelemetry wiring happens lazily at import time of the SDK only.
        """
        endpoint = os.environ.get(OTLP_ENDPOINT_ENV, "").strip()
        if not endpoint:
            return cls()
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ModuleNotFoundError:
            logger.warning(
                "%s is set but the OpenTelemetry SDK is not installed; "
                "tracing is off. Install it with `pip install 'lite-llama[trace]'`",
                OTLP_ENDPOINT_ENV,
            )
            return cls()

        provider = TracerProvider(resource=Resource({"service.name": service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        otel_trace.set_tracer_provider(provider)
        logger.info("exporting OTLP traces to %s", endpoint)
        return cls(otel_trace.get_tracer(service_name))

    # -------------------------------------------------------------- spans #
    def start_span(self, name: str, **attributes: Any) -> _Span | None:
        """Open a span; returns ``None`` when tracing is off (the cheap path)."""
        if self._otel is None:
            return None
        span = self._otel.start_span(name)
        for key, value in attributes.items():
            span.set_attribute(key, value)
        return span

    def end_span(self, span: _Span | None, **attributes: Any) -> None:
        """Close a span with its final attributes; ``None`` is a no-op."""
        if span is None:
            return
        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.end()
