"""The metrics vocabulary: TTFT / TPOT / TPS folded from step timestamps.

``BenchResult`` plus ``steps_to_result`` define every script's numbers, and
``run_requests`` is the one submit-and-drain loop every step-driven backend
shares — which is what makes their numbers comparable.

Usage:
    from benchmarks.lib import BenchResult, steps_to_result, run_requests
"""

from __future__ import annotations

import itertools
import statistics
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class BenchResult:
    """One benchmark measurement. ``gen_tokens`` is the throughput denominator."""

    ttft_ms: float
    tpot_ms: float
    total_s: float
    steps: int
    batch: int
    gen_tokens: int
    tpot_p50_ms: float = 0.0  # only backends that time every step can supply this

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.total_s if self.total_s else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "tps": self.tps}

    def row(self, label: str) -> str:
        return (
            f"{label:18s} TTFT {self.ttft_ms:7.1f} ms | "
            f"TPOT {self.tpot_ms:6.2f} ms | "
            f"TPS {self.tps:7.1f} tok/s | "
            f"{self.gen_tokens} tok in {self.total_s:.2f}s"
        )


def steps_to_result(
    step_ends: list[float],
    *,
    t_start: float,
    total_s: float,
    batch: int,
    gen_tokens: int | None = None,
) -> BenchResult:
    """Fold per-step completion timestamps into a :class:`BenchResult`.

    TTFT is the first step's end minus submission time; TPOT is the mean interval
    of the steps after it. Every step-driven backend goes through this function,
    which is what makes their numbers comparable.

    Args:
        step_ends: ``perf_counter()`` at the end of each step.
        t_start: Submission time (taken after ``torch.cuda.synchronize()``).
        total_s: Whole-run wall clock, computed by the caller after the final sync.
        batch: Concurrent request count.
        gen_tokens: Tokens actually produced; omitted means lockstep advance
            (``batch`` per step).
    """
    deltas = [b - a for a, b in itertools.pairwise(step_ends)]
    return BenchResult(
        ttft_ms=(step_ends[0] - t_start) * 1000 if step_ends else 0.0,
        tpot_ms=(statistics.mean(deltas) * 1000) if deltas else 0.0,
        tpot_p50_ms=(statistics.median(deltas) * 1000) if deltas else 0.0,
        total_s=total_s,
        steps=len(step_ends),
        batch=batch,
        gen_tokens=len(step_ends) * batch if gen_tokens is None else gen_tokens,
    )


@dataclass
class RequestRun:
    """What one :func:`run_requests` call produced.

    Both metric bases live here because benchmarks need different ones: ``step_ends``
    gives lockstep step intervals (:meth:`result`), each request's own timestamps a
    latency distribution anchored on ``started`` (the engine uses the same clock).
    """

    requests: list
    started: float
    total_s: float
    step_ends: list[float]

    @property
    def gen_tokens(self) -> int:
        """Tokens produced. Requests leave on their own EOS, so ``steps * batch`` overcounts."""
        return sum(len(r.output_token_ids) for r in self.requests)

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.requests]

    def ttfts_ms(self) -> list[float]:
        """Per-request first-token latency in ms, from submission."""
        return [
            (r.first_token_time - self.started) * 1000 for r in self.requests if r.first_token_time
        ]

    def latencies_ms(self) -> list[float]:
        """Per-request completion latency in ms, from submission."""
        return [(r.finish_time - self.started) * 1000 for r in self.requests if r.finish_time]

    def result(self, batch: int) -> BenchResult:
        """The step-interval metrics (TTFT / TPOT), the lockstep basis."""
        return steps_to_result(
            self.step_ends,
            t_start=self.started,
            total_s=self.total_s,
            batch=batch,
            gen_tokens=self.gen_tokens,
        )


def run_requests(engine, prompts: list[str], params) -> RequestRun:
    """Submit a batch to a continuous-batching engine and step until it drains.

    The one loop every offline benchmark runs; its copies differed only in which
    numbers they derived afterwards, so the derivation moved here too. The engine is
    not warmed up — callers do that with their own parameters. ``params`` is one
    ``SamplingParams`` for the batch, or a sequence aligned with ``prompts`` when each
    request needs its own.
    """
    torch.cuda.synchronize()
    started = time.perf_counter()
    per_request = isinstance(params, (list, tuple))
    requests = [
        engine.add_request(prompt, params[i] if per_request else params)
        for i, prompt in enumerate(prompts)
    ]
    step_ends: list[float] = []
    while engine.has_unfinished_requests():
        engine.step()
        step_ends.append(time.perf_counter())
    torch.cuda.synchronize()
    return RequestRun(requests, started, time.perf_counter() - started, step_ends)


def print_table(results: dict[str, BenchResult]) -> None:
    for label, r in results.items():
        print(r.row(label))
