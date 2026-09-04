"""Optional OTLP tracing: one span per request, exported over OTLP/HTTP.

:class:`Tracer` wraps an OpenTelemetry tracer when the endpoint is
configured via :meth:`Tracer.from_env`; unconfigured, ``start_span``
returns None and every call is a cheap no-op.

Usage:
    tracer = Tracer.from_env()
    span = tracer.start_span("generate")
"""

from __future__ import annotations

from typing import Any, Protocol

from ...utils.env_compat import getenv
from ...utils.logger import get_logger

logger = get_logger(__name__)

#: Environment variable carrying the OTLP/HTTP collector endpoint.
OTLP_ENDPOINT_ENV = "RAPID_LLM_OTLP_ENDPOINT"


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
    def from_env(cls, service_name: str = "rapid_llm") -> Tracer:
        """Build from ``RAPID_LLM_OTLP_ENDPOINT``; missing pieces mean a no-op.

        An unreachable collector must not take the engine down either, so the
        OpenTelemetry wiring happens lazily at import time of the SDK only.
        """
        endpoint = getenv(OTLP_ENDPOINT_ENV, "").strip()
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
                "tracing is off. Install it with `pip install 'rapid-llm[trace]'`",
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
