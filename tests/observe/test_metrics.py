"""Unit tests for the observability primitives, no engine required.

Counters accumulate per label set, gauges report the latest value,
histogram buckets are cumulative, and ``observe_finish`` records the
full latency breakdown — or no-ops when disabled.

Usage:
    pytest tests/observe/test_metrics.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from lite_llama.observe.metrics import (
    Counter,
    EngineMetrics,
    Gauge,
    Histogram,
)
from lite_llama.observe.trace import Tracer


def test_counter_accumulates_per_label_set():
    counter = Counter("test:requests", "doc", label_names=("finish_reason",))
    counter.inc(finish_reason="eos")
    counter.inc(finish_reason="eos")
    counter.inc(3, finish_reason="length")

    text = counter.render()
    assert "# TYPE test:requests counter" in text
    assert 'test:requests{finish_reason="eos"} 2' in text
    assert 'test:requests{finish_reason="length"} 3' in text


def test_gauge_reports_the_latest_value():
    gauge = Gauge("test:load", "doc")
    gauge.set(5)
    gauge.set(2)

    assert "test:load 2" in gauge.render()


def test_histogram_buckets_are_cumulative():
    histogram = Histogram("test:latency", "doc", buckets=(0.1, 1.0))
    histogram.observe(0.05)  # lands in every bucket
    histogram.observe(0.5)  # lands in the 1.0 bucket and +Inf
    histogram.observe(9.0)  # lands in +Inf only

    text = histogram.render()
    assert 'test:latency_bucket{le="0.1"} 1' in text
    assert 'test:latency_bucket{le="1"} 2' in text
    assert 'test:latency_bucket{le="+Inf"} 3' in text
    assert "test:latency_count 3" in text
    assert f"test:latency_sum {0.05 + 0.5 + 9.0!r}" in text


def _finished_request(**overrides):
    base = {
        "prompt_len": 10,
        "output_token_ids": [1, 2, 3, 4],
        "finish_reason": "eos",
        "arrival_time": 100.0,
        "scheduled_time": 100.5,
        "first_token_time": 101.0,
        "finish_time": 104.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_observe_finish_records_the_full_latency_breakdown():
    metrics = EngineMetrics()
    metrics.observe_finish(_finished_request())

    text = metrics.render_prometheus()
    # 10 prompt tokens, 4 generated tokens, one eos finish.
    assert "lite_llama:request_prompt_tokens_count 1" in text
    assert "lite_llama:prompt_tokens_total 10" in text
    assert "lite_llama:generation_tokens_total 4" in text
    assert 'lite_llama:request_success_total{finish_reason="eos"} 1' in text
    # TTFT = first_token - arrival = 1.0s; TPOT = (104 - 101) / (4 - 1) = 1.0s.
    # (The renderer trims integral floats, so the sums render as "1".)
    assert "lite_llama:time_to_first_token_seconds_sum 1\n" in text
    assert "lite_llama:time_per_output_token_seconds_sum 1\n" in text


def test_a_one_token_completion_has_no_tpot_gap():
    metrics = EngineMetrics()
    metrics.observe_finish(_finished_request(output_token_ids=[7]))

    assert metrics.tpot._counts[-1] == 0  # no observation, not a zero observe
    assert metrics.ttft._counts[-1] == 1


def test_queue_time_is_observed_only_when_scheduled():
    metrics = EngineMetrics()
    metrics.observe_queue_time(_finished_request(scheduled_time=None))
    assert metrics.queue_time._counts[-1] == 0

    metrics.observe_queue_time(_finished_request())
    assert metrics.queue_time._counts[-1] == 1
    assert metrics.queue_time._sum == 0.5


def test_disabled_metrics_are_noops_that_render_nothing(monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_METRICS", "0")
    metrics = EngineMetrics.from_env()

    assert not metrics.enabled
    metrics.observe_load(3, 2)
    metrics.observe_finish(_finished_request())
    assert metrics.render_prometheus() == "\n"


def test_metrics_default_on_and_honour_the_env(monkeypatch):
    monkeypatch.delenv("LITE_LLAMA_METRICS", raising=False)
    assert EngineMetrics.from_env().enabled

    monkeypatch.setenv("LITE_LLAMA_METRICS", "off")
    assert not EngineMetrics.from_env().enabled


def test_tracer_without_an_endpoint_is_a_noop(monkeypatch):
    monkeypatch.delenv("LITE_LLAMA_OTLP_ENDPOINT", raising=False)
    tracer = Tracer.from_env()

    assert not tracer.enabled
    assert tracer.start_span("request", request_id="r0") is None
    tracer.end_span(None, finish_reason="eos")  # must not raise


def test_tracer_with_an_endpoint_but_no_sdk_degrades(monkeypatch):
    """An endpoint without the SDK warns and stays off — never an engine fault."""
    monkeypatch.setenv("LITE_LLAMA_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # force ImportError

    tracer = Tracer.from_env()
    assert not tracer.enabled
    assert tracer.start_span("request") is None


def test_tracer_wraps_the_sdk_tracer():
    """The wrapper forwards attributes and end(); the SDK side is otel's own."""

    class FakeSpan:
        def __init__(self):
            self.attributes = {}
            self.ended = False

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def end(self):
            self.ended = True

    class FakeOtel:
        def __init__(self):
            self.span = FakeSpan()

        def start_span(self, name):
            self.span.name = name
            return self.span

    otel = FakeOtel()
    tracer = Tracer(otel)
    assert tracer.enabled

    span = tracer.start_span("request", request_id="r0", prompt_tokens=3)
    assert span.name == "request"
    assert span.attributes == {"request_id": "r0", "prompt_tokens": 3}
    tracer.end_span(span, finish_reason="eos")
    assert span.attributes["finish_reason"] == "eos"
    assert span.ended
