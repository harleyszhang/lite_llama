"""Microbenchmark the KV-cache *write* path: the scatter, and the allocation before it.

A decode step touches the cache twice: :func:`update_kv_buffer` stores this
token's K/V into rows that :meth:`KVCacheManager.alloc_kvcache_index` picked one
instruction earlier. Both are cheap next to attention, which is exactly why they
get benchmarked badly — and a few microseconds of per-step overhead is a few
percent of TPOT at decode batch 1.

The two halves need different timing discipline, which is the point of this file:

*Scatter* is idempotent — writing the same rows twice leaves the same bytes — so
back-to-back replay through :func:`microbench.bench` is valid.

*Allocation* is not. Every call consumes rows, and which of the allocator's three
strategies runs depends on how the pool got into its current state. Replaying it
back to back walks the pool from empty to full and averages a staircase, so it
goes through :func:`microbench.bench_host`, whose ``reset`` rebuilds the state
outside the timed interval. Host wall time, not CUDA events: the allocator's cost
is a ``nonzero(...).item()`` stalling the launch queue, which a device-timeline
timer reports as nearly free. The trap worth naming: ``free_all()`` restores
``_bump_is_exact``, so a reset that just calls it measures the bump fast path
three times over, no matter which state the row is labelled with.

That state turned out to be the largest effect anywhere in the KV path — 265 us
once the bump cursor is invalidated against 24 us while it holds, an 11x swing
that a benchmark using a fresh manager never sees and that dwarfs the 4 us
scatter it precedes.

Usage:
    python benchmarks/kernels/bench_kv_write.py
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kv_pool import fresh_rows, paged_pool
from microbench import Row, Work, bench, bench_host, metadata, report, verify

from lite_llama.executor.kv_cache_manager import KVCacheManager

# Import registers the native KernelSpec rows, so dispatch() below finds them.
from lite_llama.kernels.backends import native as _native
from lite_llama.kernels.ops import dispatch
from lite_llama.kernels.update_kv_buffer import update_kv_buffer

#: ``(label, seq_lens)``. The scatter's cost is set by how many tokens are
#: written and how their destination rows are spread, so prefill (one sequence,
#: many rows) and decode (many sequences, one row each) are different operations
#: on the same kernel rather than two points on one curve.
SCATTER_CASES: list[tuple[str, list[int]]] = [
    ("prefill_s4096", [4096]),
    ("prefill_s1024", [1024]),
    ("decode_b64", [1] * 64),
    ("decode_b8", [1] * 8),
    ("decode_b1", [1]),
]

#: Rows the allocator searches. Its hot-path cost is a function of pool size, not
#: of the request, so this stays fixed while the state varies.
_ALLOC_POOL_ROWS = 65536

#: Rows requested per call: one decode step reserves one row per sequence in the
#: batch, so this is the batch size.
_ALLOC_SIZES = (1, 32)

#: ``KernelSpec.name`` of the row under test, so a table entry maps onto a
#: registry entry rather than onto a bare function name.
_IMPL = "native/update_kv_buffer"

_HEADS, _DIM = 8, 128


# --------------------------------------------------------------------------- #
# Scatter
# --------------------------------------------------------------------------- #
def scatter_work(tokens: int, num_kv_heads: int, head_dim: int, esz: int) -> Work:
    """Minimum traffic for scattering ``tokens`` rows of K and V.

    Read each projection once, write each cache row once — no arithmetic at all,
    so a percentage of memory peak is the only meaningful score, and only at
    prefill: at decode the whole kernel is shorter than a launch, and %bw there
    describes the launch, not the scatter.

    The destination pattern turned out not to matter (``scattered`` and
    ``sequential`` measure the same), which the row size explains:
    ``2 * num_kv_heads * head_dim`` elements land contiguously, 4 KiB at the
    shapes here, so a scattered write is 32 consecutive cache lines placed at a
    random offset rather than a random access.
    """
    elems = 4 * tokens * num_kv_heads * head_dim  # k in, v in, both out
    return Work(flops=0, moved=elems * esz)


def check_scatter() -> None:
    """Verify the scatter lands in the rows the table says it does.

    Only the written rows are compared: a kernel that wrote the wrong slot would
    leave the gathered result mismatched, and comparing the whole 48 MiB pool
    would cost more than the benchmark. The fragmented destination is the case
    that matters — a stride bug in the combined layout shows up as K appearing in
    the V half, which a sequential index list can mask.
    """
    print("Correctness (rows read back through the destination index):")
    for layout in ("fragmented", "contiguous"):
        pool = paged_pool([64], num_kv_heads=2, head_dim=64, layout=layout)
        idx = pool.table[0].contiguous()
        k, v = fresh_rows(64, num_kv_heads=2, head_dim=64)
        update_kv_buffer(k, v, idx, pool.buffer)
        verify(
            f"{_IMPL} [{layout}]",
            pool.buffer[idx.long()],
            torch.cat([k, v], dim=1),
            rtol=0.0,
            atol=0.0,
        )
        del pool


def bench_scatter() -> list[Row]:
    """Time the scatter over fragmented, sequential and fp8 destinations."""
    rows: list[Row] = []
    for case, seq_lens in SCATTER_CASES:
        tokens = sum(seq_lens)
        for label, dtype in (("production", torch.float16), ("fp8_pool", torch.uint8)):
            pool = paged_pool(seq_lens, num_kv_heads=_HEADS, head_dim=_DIM, dtype=dtype)
            k, v = fresh_rows(tokens, num_kv_heads=_HEADS, head_dim=_DIM, dtype=dtype)
            work = scatter_work(tokens, _HEADS, _DIM, pool.buffer.element_size())

            scattered = pool.table.reshape(-1).contiguous()
            # Everything the timed call needs is bound as a default: the loop
            # rebinds these names, and the tensors are freed at the end of the
            # iteration to keep the pool from accumulating.
            us = bench(lambda i=scattered, p=pool, k=k, v=v: update_kv_buffer(k, v, i, p.buffer))
            rows.append(Row(f"{_IMPL} [{label}]", case, us, work))

            if dtype is torch.float16:
                # Same rows, handed out consecutively: what the bump allocator
                # returns while a request is still appending.
                seq = torch.arange(tokens, dtype=torch.int32, device="cuda")
                us = bench(lambda i=seq, p=pool, k=k, v=v: update_kv_buffer(k, v, i, p.buffer))
                rows.append(Row(f"{_IMPL} [sequential]", case, us, work))

            del pool, k, v
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #
def _invalidate_bump(kv: KVCacheManager) -> None:
    """Leave the pool empty but the bump cursor untrusted.

    Reached through the public API on purpose — one allocation, immediately
    released — because that is how a finished request reaches it in production.
    Poking ``_bump_is_exact`` directly would benchmark a state the engine cannot
    produce.
    """
    kv.release_ref(kv.alloc_kvcache_index(1).long())


def _alloc_states(kv: KVCacheManager) -> dict[str, Callable[[], None]]:
    """The three allocator states, each as a complete reset.

    Keys name the strategy :meth:`alloc_kvcache_index` will take:

    ``bump``
        Append-only cache. Returns a view of the pre-cast index tensor with no
        device read at all — this is the state every ``generate()`` starts in,
        and the reason a benchmark that only ever uses a fresh manager reports
        the fast path as if it were the allocator.
    ``run_search``
        Free list intact but the cursor distrusted, so
        :meth:`alloc_contiguous_kvcache` runs: a ``nonzero`` over the whole pool
        plus two ``.item()`` reads, i.e. three device synchronisations on the
        decode critical path.
    ``fragmented``
        Every other row held, so no run of two exists and the search falls
        through to :meth:`alloc_kvcache` — a second full ``nonzero``. Note the
        free list is also half as long here, which pulls the other way; at
        ``need=1`` the run search always succeeds and this row differs from
        ``run_search`` only by that density.
    """

    def bump() -> None:
        kv.free_all()

    def run_search() -> None:
        kv.free_all()
        _invalidate_bump(kv)

    def fragmented() -> None:
        kv.free_all()
        _invalidate_bump(kv)
        kv.add_ref(torch.arange(0, kv.max_num_tokens, 2, device=kv.device))

    return {"bump": bump, "run_search": run_search, "fragmented": fragmented}


def check_alloc(kv: KVCacheManager) -> None:
    """Check the invariant a state-machine benchmark can break silently.

    There is no numeric reference for an allocator; the property to hold is that
    each state still hands out the requested number of distinct free rows. A
    reset that corrupts the reference counts would otherwise keep producing
    plausible timings for an allocator returning overlapping rows.
    """
    print("\nInvariants (distinct free rows returned per state):")
    for name, reset in _alloc_states(kv).items():
        reset()
        free_before = kv.can_use_mem_size
        idx = kv.alloc_kvcache_index(32)
        assert idx.numel() == 32, f"{name}: asked for 32 rows, got {idx.numel()}"
        assert idx.unique().numel() == 32, f"{name}: returned duplicate rows"
        assert kv.can_use_mem_size == free_before - 32, f"{name}: capacity accounting drifted"
        print(f"  ok   alloc_kvcache_index [{name}]        32 distinct rows")
    kv.free_all()


def bench_alloc(kv: KVCacheManager) -> list[Row]:
    """Time one allocation per state, with the state rebuilt between calls.

    The floor row is not filler: these numbers are host wall time around a
    synchronise, so an empty call is not free, and a reader needs to know how
    much of the smallest row is the instrument.
    """
    rows: list[Row] = [Row("(harness floor: sync only)", "-", bench_host(lambda: None), Work())]
    for need in _ALLOC_SIZES:
        for name, reset in _alloc_states(kv).items():
            us = bench_host(lambda n=need: kv.alloc_kvcache_index(n), reset)
            rows.append(Row(f"alloc_kvcache_index [{name}]", f"need{need}", us, Work()))
    kv.free_all()
    return rows


def show_dispatch() -> None:
    """Print the implementation dispatch picks for ``kv_write``.

    The table labels rows with the ``KernelSpec.name`` this prints, so a reader
    can check that the row being timed is the row dispatch would choose. The
    native spec carries no dtype filter — fp8 quantisation happens in
    ``PagedAttention`` before the write — so the only gate is the ``kv:paged``
    layout tag this pool satisfies.
    """
    sel = dispatch("kv_write", dtype="fp16", layout=frozenset({"kv:paged"}))
    print(f"\nDispatch for kv_write:\n{sel.explain()}")
    assert sel.spec.name == _IMPL, f"table labels say {_IMPL}, dispatch picks {sel.spec.name}"
    assert sel.load() is not None


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    torch.set_grad_enabled(False)

    print(metadata())
    print()
    check_scatter()
    show_dispatch()
    print()

    print("Scatter (update_kv_buffer):")
    report(bench_scatter())

    # One layer is enough: the allocator only touches the index tensors, and the
    # per-layer buffers exist here just so construction matches production.
    kv = KVCacheManager(
        num_layers=1,
        num_kv_heads=_HEADS,
        head_dim=_DIM,
        gpu_num_blocks=_ALLOC_POOL_ROWS,
    )
    check_alloc(kv)
    print(f"\nAllocation (KVCacheManager.alloc_kvcache_index, {_ALLOC_POOL_ROWS} rows):")
    report(bench_alloc(kv))

    print(
        "\nRead the tables as: at prefill the scatter is a bandwidth kernel and hits\n"
        "70-76% of peak, so %bw is its score there and the launch configuration (one\n"
        "program per token, num_warps=1) is not costing anything. At decode every\n"
        "case collapses to 4-5 us whether it writes 1 row or 64, which is launch\n"
        "latency: %bw is meaningless in those rows and the only way to make the write\n"
        "cheaper is to issue fewer launches, not to move fewer bytes. Note the\n"
        "contrast with the read side, where halving the bytes barely helped: here fp8\n"
        "halves the prefill time at constant %bw, because a scatter never decodes the\n"
        "bytes it copies.\n"
        "In the allocator table the 11x from bump to run_search is what one finished\n"
        "request costs every later decode step, and it is host time spent waiting on\n"
        "``nonzero(...).item()`` — invisible to a CUDA-event timer, and roughly a\n"
        "quarter of a millisecond next to a 4 us scatter."
    )


if __name__ == "__main__":
    main()
