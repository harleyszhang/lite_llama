"""Scheduler micro-benchmark: pure-Python planning cost, no GPU.

Scenarios mirror the three shapes a production step takes:
  1. decode-heavy    — a full batch past prefill, nothing new arriving
  2. chunked-mixed   — long prompts interleaving with a large decode batch
  3. admit-churn     — deep waiting queue, slots recycling every step
  4. prefix-admission — prefix-cache admission path, shared system prompt
  5. attr-access     — raw Request attribute reads (slots pay off here)

Usage: python bench_scheduler.py [label]
"""

from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, "/home/honggao/projects/lite_llama")

from lite_llama.engine.sampler import SamplingParams
from lite_llama.engine.scheduler import Request, Scheduler, SchedulerConfig


def make_request(request_id: str, prompt_len: int) -> Request:
    return Request(
        request_id=request_id,
        prompt="x" * prompt_len,
        prompt_token_ids=list(range(prompt_len)),
        params=SamplingParams(),
    )


def bench_decode_heavy() -> tuple[float, int]:
    """64 running requests, all past prefill: schedule() fixed cost per step."""
    sched = Scheduler(
        SchedulerConfig(max_seq_len=4096, max_num_seqs=64, max_num_batched_tokens=8192),
        num_slots=64,
    )
    for i in range(64):
        sched.add_request(make_request(f"r{i}", 32))
    while True:
        out = sched.schedule()
        if len(out.decode) == 64:
            break
    start = time.perf_counter()
    steps = 2000
    for _ in range(steps):
        sched.schedule()
    return time.perf_counter() - start, steps


def bench_chunked_mixed() -> tuple[float, int]:
    """8 chunked prefills of 2048 tokens (chunk 256) beside 56 decoding."""
    sched = Scheduler(
        SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            max_chunk_size=256,
        ),
        num_slots=64,
    )
    for i in range(56):
        sched.add_request(make_request(f"d{i}", 16))
    for i in range(8):
        sched.add_request(make_request(f"p{i}", 2048))
    # Requests finish when their generation cap is hit (engine-side harvest).
    for r in [*sched.waiting]:
        r.params = SamplingParams(max_gen_len=48)
        r.max_new_tokens = 48
    start = time.perf_counter()
    steps = 0
    while sched.has_unfinished_requests():
        out = sched.schedule()
        for r in out.decode:
            r.output_token_ids.append(1)
            if not r.has_room:
                sched.finish(r, "length")
        steps += 1
    return time.perf_counter() - start, steps


def bench_admit_churn() -> tuple[float, int]:
    """256 queued, 64 slots: admit waves then finish them all, 8 rounds."""
    sched = Scheduler(
        SchedulerConfig(max_seq_len=4096, max_num_seqs=64, max_num_batched_tokens=1 << 20),
        num_slots=64,
    )
    start = time.perf_counter()
    cycles = 0
    for round_index in range(8):
        for i in range(256):
            sched.add_request(make_request(f"r{round_index}-{i}", 32))
        while sched.has_unfinished_requests():
            out = sched.schedule()
            for r in out.decode:
                r.output_token_ids.append(1)
                sched.finish(r, "eos")
            for r in out.prefill:
                if r.prefill_done:
                    r.output_token_ids.append(1)
                    sched.finish(r, "eos")
            cycles += 1
    return time.perf_counter() - start, cycles


def bench_prefix_admission() -> tuple[float, int]:
    """Prefix cache on: 256 requests sharing a 1024-token prefix, one by one."""
    sched = Scheduler(
        SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=64,
            max_num_batched_tokens=1 << 20,
            enable_prefix_cache=True,
        ),
        num_slots=64,
    )
    shared = list(range(1024))
    start = time.perf_counter()
    for i in range(256):
        req = Request(
            request_id=f"r{i}",
            prompt="x" * 1028,
            prompt_token_ids=shared + [50_000 + i] * 4,
            params=SamplingParams(max_gen_len=1),
        )
        sched.add_request(req)
        out = sched.schedule()
        for r in out.prefill:
            if r.prefill_done:
                r.output_token_ids.append(1)
                sched.finish(r, "eos")
    return time.perf_counter() - start, 256


def bench_attr_access() -> tuple[float, int]:
    """Raw attribute reads on the hot path (slots pay off here)."""
    req = make_request("a", 64)
    req.num_computed_tokens = 32
    n = 200_000
    start = time.perf_counter()
    total = 0
    for _ in range(n):
        total += req.prompt_len + req.num_computed_tokens + req.seq_len
        if req.prefill_done:
            total += 1
        if req.has_room:
            total += 1
    return time.perf_counter() - start, n


SCENARIOS = [
    ("decode-heavy (2000 steps)", bench_decode_heavy),
    ("chunked-mixed (full run)", bench_chunked_mixed),
    ("admit-churn (8x256 reqs)", bench_admit_churn),
    ("prefix-admission (256 reqs)", bench_prefix_admission),
    ("attr-access (200k reads)", bench_attr_access),
]


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    print(f"== scheduler micro-benchmark: {label} ==")
    for name, fn in SCENARIOS:
        times: list[float] = []
        for _ in range(5):
            elapsed, ops = fn()
            times.append(elapsed / ops * 1e6)  # us per op
        best = min(times)
        med = statistics.median(times)
        spread = (max(times) - min(times)) / med * 100
        print(f"{name:34s} best {best:8.3f} us/op   median {med:8.3f}   spread {spread:4.1f}%")


if __name__ == "__main__":
    main()
