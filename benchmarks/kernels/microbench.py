"""Kernel microbenchmark harness: one timing discipline, one honest metric.

``bench`` warms, L2-flushes and times with device sync;
``bench_stateful`` / ``bench_host`` cover alloc-reset and host-side
work. ``verify`` gates every result on numerical correctness first.

Usage:
    from benchmarks.kernels.microbench import bench
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
from triton.testing import do_bench

#: Dense tensor-core peak (TFLOP/s, fp16/bf16 with fp32 accumulate, no
#: sparsity) per ``torch.cuda.get_device_properties().name``. Only the compute
#: corner needs a table: it is not derivable from CUDA properties. An absent
#: entry yields ``None``, and the report then prints achieved TFLOP/s with no
#: percentage rather than a fraction of a guessed peak.
_TENSOR_CORE_TFLOPS: dict[str, float] = {
    "NVIDIA A10": 125.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100 80GB PCIe": 312.0,
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA H800": 989.0,
}

#: Env vars that silently change which kernel runs, so a table without them is
#: not reproducible (see ``lite_llama.kernels.dispatcher``).
_RELEVANT_ENV = (
    "LITE_LLAMA_FORCE_BACKEND",
    "LITE_LLAMA_AUTOTUNE",
    "LITE_LLAMA_AUTOTUNE_DIR",
    "LITE_LLAMA_FROZEN_RANK",
    "LITE_LLAMA_KERNEL_TRACE",
    "CUDA_VISIBLE_DEVICES",
    "TRITON_PRINT_AUTOTUNING",
)


# --------------------------------------------------------------------------- #
# Device roofline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Peaks:
    """The two roofline corners of the current device.

    Attributes:
        name: Device name as CUDA reports it.
        gbps: Peak HBM/GDDR bandwidth in GB/s.
        tflops: Dense tensor-core peak, or ``None`` when this device is not in
            :data:`_TENSOR_CORE_TFLOPS`.
    """

    name: str
    gbps: float
    tflops: float | None

    @property
    def ridge(self) -> float | None:
        """Machine balance in FLOP/byte: above it a kernel can be compute-bound."""
        if self.tflops is None:
            return None
        return self.tflops * 1e12 / (self.gbps * 1e9)


def device_peaks(device: int = 0) -> Peaks:
    """Read the device's roofline corners.

    The bandwidth is computed rather than looked up: memory clock times bus
    width times two transfers per clock reproduces the vendor figure on both
    GDDR6 (A10: 6251 MHz x 384 bit -> 600 GB/s) and HBM (A100: 1215 MHz x
    5120 bit -> 1555 GB/s), so a new device needs no table entry to get an
    honest memory-side percentage.
    """
    p = torch.cuda.get_device_properties(device)
    gbps = p.memory_clock_rate * 1e3 * (p.memory_bus_width / 8) * 2 / 1e9
    return Peaks(name=p.name, gbps=gbps, tflops=_TENSOR_CORE_TFLOPS.get(p.name))


def l2_bytes(device: int = 0) -> int:
    """L2 capacity, the threshold a working set must clear to touch HBM at all."""
    return torch.cuda.get_device_properties(device).L2_cache_size


# --------------------------------------------------------------------------- #
# What the operation costs, in theory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Work:
    """The work the *operation* implies, not the work a kernel happens to do.

    ``moved`` is the theoretical minimum traffic: every input byte read exactly
    once, every output byte written exactly once. Counting a kernel's real loads
    instead would reward a cache-thrashing implementation with a higher GB/s
    than a good one — the metric would improve as the kernel got worse. The same
    rule makes two implementations of one op directly comparable: they share a
    numerator, so only the measured time differs.

    Attributes:
        flops: Multiply-accumulate counted as 2 FLOP; 0 for pure data movement.
        moved: Minimum bytes crossing the memory interface.
    """

    flops: int = 0
    moved: int = 0

    @property
    def intensity(self) -> float | None:
        """FLOP per byte; compare against :attr:`Peaks.ridge` to name the bound."""
        return self.flops / self.moved if self.moved else None


@dataclass(frozen=True)
class Row:
    """One measured (implementation, case) pair.

    Attributes:
        impl: Which implementation ran — use the ``KernelSpec.name`` spelling
            (``"native/attention_decode_triton"``) when the row came from
            dispatch, so a table entry maps onto a registry entry.
        case: Shape/state label; must distinguish everything the number depends
            on, fragmentation state included.
        us: Median latency in microseconds.
        work: Theoretical cost of the operation for this case.
    """

    impl: str
    case: str
    us: float
    work: Work

    @property
    def tflops(self) -> float | None:
        return self.work.flops / (self.us * 1e6) if self.work.flops else None

    @property
    def gbps(self) -> float | None:
        return self.work.moved / (self.us * 1e3) if self.work.moved else None


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def bench(fn: Callable[[], object], *, warmup_ms: int = 25, rep_ms: int = 100) -> float:
    """Cold-L2 median latency of an idempotent ``fn``, in microseconds.

    ``do_bench`` flushes L2 before each replay and keeps the flush outside the
    timed events, so the result is the "data arrives from HBM" case — the one a
    decode step actually sees, since a few MB of L2 hold a negligible slice of a
    multi-GB KV pool. A warm-L2 number is the optimistic bound and must be
    labelled as such rather than reported as the kernel's speed.

    The untimed first call is deliberate: it forces the Triton JIT (and any
    autotune search behind :func:`lite_llama.kernels.dispatcher.autotune.get_best_config`)
    to finish, so compilation does not land inside the measurement.
    """
    fn()
    torch.cuda.synchronize()
    return float(do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")) * 1e3


def bench_stateful(
    fn: Callable[[], object],
    reset: Callable[[], object],
    *,
    warmup: int = 5,
    repeat: int = 30,
    device: int = 0,
) -> float:
    """Median GPU latency in microseconds when each call mutates state.

    ``reset`` runs between timed intervals, never inside one, so what is
    measured is still a single call of ``fn`` — the pattern for anything whose
    result depends on how much it has already been called: block allocation,
    ref-count release, cache eviction. Iteration counts are fixed rather than
    fitted to a time budget because the number of state transitions is bounded
    (a pool with N free rows can only be drained so many times), which
    ``do_bench``'s "repeat until 100 ms elapse" cannot respect.

    Every interval is enqueued before anything is read back, which is what keeps
    this comparable to :func:`bench`. Synchronising per iteration instead would
    add the launch-latency floor described in the module docstring to every row.
    The cost is a constraint on ``reset``: it must stay on the device. A reset
    containing a ``.item()`` or a ``.cpu()`` synchronises implicitly and hands
    the floor back — use :func:`bench_host` for such operations.

    Args:
        fn: The single operation under study.
        reset: Restores the precondition ``fn`` consumes, without
            synchronising. Must be *complete*: a partial reset leaves iteration
            30 measuring a different state than iteration 1.
        warmup: Untimed (reset, call) pairs, to force compilation.
        repeat: Timed calls; the median is returned.
        device: CUDA device whose L2 gets flushed between calls.
    """
    flush = _l2_flusher(device)
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize()

    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeat)
    ]
    for start, end in pairs:
        reset()
        flush()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) * 1e3 for start, end in pairs)


def bench_host(
    fn: Callable[[], object],
    reset: Callable[[], object] = lambda: None,
    *,
    warmup: int = 5,
    repeat: int = 30,
) -> float:
    """Median host wall time in microseconds, including any stall ``fn`` forces.

    For operations whose cost is not GPU work but the host waiting on it — a
    block allocator reading ``nonzero(...).item()``, a ref-count release, any
    Python bookkeeping on the decode critical path. CUDA events are the wrong
    instrument there: they measure the device timeline, and a function that
    stalls the launch queue for 250 us while issuing 3 us of kernels looks free.

    The interval covers ``fn`` plus a trailing synchronise, so an asynchronous
    tail is charged to the caller rather than to whoever launches next. That
    makes the harness floor a real term in the result — a few microseconds of
    empty synchronise — so a benchmark comparing near-floor numbers should
    print :func:`bench_host` of a no-op alongside them.

    Args:
        fn: The operation under study.
        reset: Restores the precondition, outside the timed interval.
        warmup: Untimed (reset, call) pairs.
        repeat: Timed calls; the median is returned.
    """
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1e6)
    return statistics.median(times)


def _l2_flusher(device: int = 0) -> Callable[[], None]:
    """A callable that evicts L2 by overwriting twice its capacity."""
    buf = torch.empty(2 * l2_bytes(device), dtype=torch.int8, device=f"cuda:{device}")

    def flush() -> None:
        buf.zero_()

    return flush


# --------------------------------------------------------------------------- #
# Correctness, before any timing
# --------------------------------------------------------------------------- #
def verify(
    name: str,
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> float:
    """Assert ``out`` matches ``ref`` and return the max absolute difference.

    A fast wrong kernel is not a data point, so this runs before the timing
    loop and raises on mismatch. The returned figure is the same quantity
    :class:`lite_llama.kernels.dispatcher.spec.GoldenRecord` records as
    ``max_abs_diff``: a benchmark run is where that evidence comes from, and an
    implementation without it stays out of default dispatch.
    """
    torch.testing.assert_close(out.float(), ref.float(), rtol=rtol, atol=atol)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"  ok   {name:<44} max_abs_diff={diff:.3e}  (rtol={rtol}, atol={atol})")
    return diff


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def metadata(device: int = 0) -> str:
    """Reproducibility header: without it a table is an anecdote."""
    peaks = device_peaks(device)
    lines = [
        f"device    {peaks.name}  ({peaks.gbps:.0f} GB/s peak, "
        f"{'?' if peaks.tflops is None else f'{peaks.tflops:.0f}'} TFLOP/s dense tc, "
        f"{l2_bytes(device) / 2**20:.0f} MiB L2)",
        f"software  torch {torch.__version__}, triton {triton.__version__}, "
        f"python {platform.python_version()}",
        f"commit    {_git_describe()}",
        f"command   {' '.join(sys.argv)}",
    ]
    env = {k: os.environ[k] for k in _RELEVANT_ENV if k in os.environ}
    lines.append(f"env       {env or '(none set)'}")
    return "\n".join(lines)


def _git_describe() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def report(rows: list[Row], *, device: int = 0) -> None:
    """Print one table plus a service-of-light check on every row.

    Throughput columns come first because they are what survives a change of
    shape; latency stays for absolute cost. A percentage above 100 is printed as
    a violation rather than a win: it means the work formula, the units, or the
    assumption that the kernel did the whole operation is wrong.
    """
    peaks = device_peaks(device)
    header = f"{'impl':<34} {'case':<26} {'us':>9} {'TFLOP/s':>9} {'%tc':>6} {'GB/s':>9} {'%bw':>6}"
    print(header)
    print("-" * len(header))

    violations: list[str] = []
    for r in rows:
        tf = "" if r.tflops is None else f"{r.tflops:9.2f}"
        gb = "" if r.gbps is None else f"{r.gbps:9.1f}"
        pct_tc, pct_bw = "", ""
        if r.tflops is not None and peaks.tflops is not None:
            frac = 100 * r.tflops / peaks.tflops
            pct_tc = f"{frac:6.1f}"
            if frac > 100:
                violations.append(f"{r.impl}/{r.case}: {frac:.0f}% of tensor-core peak")
        if r.gbps is not None:
            frac = 100 * r.gbps / peaks.gbps
            pct_bw = f"{frac:6.1f}"
            if frac > 100:
                violations.append(f"{r.impl}/{r.case}: {frac:.0f}% of memory peak")
        print(f"{r.impl:<34} {r.case:<26} {r.us:9.1f} {tf:>9} {pct_tc:>6} {gb:>9} {pct_bw:>6}")

    if violations:
        print("\nSOL violated — the number is wrong, not the hardware:")
        for v in violations:
            print(f"  ! {v}")
        print(
            "  Check, in order: unit factors; the FLOP/byte formula; work the\n"
            "  kernel skipped (masked tails, early exit); a working set that fits\n"
            "  in L2 so the traffic never reached HBM; a baseline that is not\n"
            "  doing the same operation."
        )
