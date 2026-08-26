"""Microbenchmark ``attention.decode`` over the paged KV cache.

Decode attention is the kernel every generated token waits on, and it is
bandwidth-bound: one query row per sequence against the whole cached history, so
the score is achieved GB/s against the pool's minimum traffic. Three plausible
ways of building the inputs could each make the gather cheaper than production's,
which is why the same kernel is timed over four constructions rather than one.
What each construction is worth on an A10 is recorded below — two of the three
turned out to be worth nothing, and that result is more useful than the warning
it replaced:

``production``
    Fragmented pool, strided K/V views out of the combined per-layer buffer.
    This is what ``lite_llama/modules/attention.py`` passes, so this row is the
    kernel's speed and the only row to quote. It reaches 63-67% of peak
    bandwidth at batch >= 8, and 55% at batch 1 where there is not enough work to
    cover launch latency.
``contiguous``
    Same kernel, sequence rows laid out consecutively. Measures 2-4% faster than
    ``production``: at 2 KiB per cache row a random gather runs near streaming
    speed, so paging is not what a decode regression would be hiding in. Kept as
    the bound it establishes — it says the whole of paging is worth a few
    percent, so a change claiming more than that is measuring something else.
``split_alloc``
    Two separately allocated caches instead of views into one buffer, halving the
    row stride. Never beats ``production``: equal at three of the four shapes and
    8% *slower* at one, which is the wrong direction for a stride explanation. 8
    heads x 128 dim fp16 is already 16 cache lines per side, so halving the
    stride changes no line's useful payload. Worth keeping as a regression guard
    for smaller head geometries (MQA, or 64-dim heads), where the row approaches
    a single line and the stride starts to matter.
``fp8_pool``
    e4m3 bytes in a uint8 container with caller-side scales. Halves the traffic
    but only takes 6-10% off the time, so its %bw drops from 67% to 37%: the fp8
    path is limited by the dequant, not by memory. That inversion is the reason
    it is a case of its own and not a dtype column — read as a dtype variant, a
    lower GB/s looks like a regression when it is a different bottleneck.

Usage:
    python benchmarks/kernels/bench_paged_decode.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kv_pool import PagedPool, paged_pool
from microbench import Row, Work, bench, metadata, report, verify

# Import registers the native KernelSpec rows, so dispatch() below finds them.
from lite_llama.kernels.backends import native as _native
from lite_llama.kernels.flashdecoding import flash_decoding
from lite_llama.kernels.ops import dispatch
from tests.reference import paged_decode_attention

#: ``(batch, seq_len, num_q_heads, num_kv_heads, head_dim)``. GQA 4x with 128-dim
#: heads is the Qwen3/Llama-3 decode geometry; the batch/length pairs walk the
#: serving trade-off at roughly constant cached-token count, which is what the
#: kernel's runtime tracks.
CASES: list[tuple[int, int, int, int, int]] = [
    (1, 4096, 32, 8, 128),
    (8, 2048, 32, 8, 128),
    (32, 1024, 32, 8, 128),
    (64, 512, 32, 8, 128),
]

#: Stage 1 accumulates the fp16 qk product in fp32, so the kernel is more
#: accurate than a plain fp16 dot; these are the tolerances tests/kernels use.
_RTOL, _ATOL = 1e-2, 1e-2

#: ``KernelSpec.name`` of the row under test, so a table entry maps onto a
#: registry entry rather than onto a bare function name.
_IMPL = "native/flash_decoding"


def decode_work(pool: PagedPool, num_q_heads: int) -> Work:
    """Theoretical cost of one decode step over ``pool``.

    Two matmuls per attended token (``q @ k`` then ``p @ v``), each a
    multiply-accumulate over ``head_dim``, counted per query head. Traffic is the
    cached K/V read once plus the query in and the output out — nothing else
    crosses the interface in an ideal implementation, whatever the kernel's
    split-K partial buffers actually do.
    """
    head_dim = pool.buffer.shape[2]
    batch = len(pool.seq_lens)
    esz = torch.empty((), dtype=torch.float16).element_size()
    q_and_out = 2 * batch * num_q_heads * head_dim * esz
    return Work(
        flops=4 * num_q_heads * head_dim * pool.total_tokens,
        moved=pool.kv_bytes_read() + q_and_out,
    )


def run(pool: PagedPool, q: torch.Tensor, **scales: float):
    """Invoke the kernel exactly as :mod:`lite_llama.modules.attention` does."""
    return flash_decoding(
        q,
        pool.k,
        pool.v,
        1.0 / math.sqrt(pool.buffer.shape[2]),
        pool.table,
        pool.req_idx,
        pool.seq_lens,
        pool.max_seq_len,
        **scales,
    )


def check_correctness() -> None:
    """Verify against the paged reference before any number is printed.

    Deliberately small: the reference materialises the gathered history per
    sequence in fp32, so running it on the benchmark's own shapes would cost
    minutes and verify nothing extra. The fragmented layout is the case worth
    checking — it is the one where an index bug hides.
    """
    print("Correctness (tests.reference.paged_decode_attention):")
    for layout in ("fragmented", "contiguous"):
        pool = paged_pool([37, 128, 512], num_kv_heads=2, head_dim=64, layout=layout)
        q = torch.randn(3, 8, 64, device="cuda", dtype=torch.float16) * 0.3
        out = run(pool, q)
        ref = paged_decode_attention(
            q, pool.k, pool.v, pool.table, pool.seq_lens, 1.0 / math.sqrt(64)
        )
        verify(f"{_IMPL} [{layout}]", out, ref, rtol=_RTOL, atol=_ATOL)
    check_fp8()


def check_fp8() -> None:
    """Verify the fp8 row too, against the same bytes widened by torch.

    The fp8 case needs its own check and cannot borrow the fp16 one: it exercises
    a different code path inside the kernel (the e4m3 bit-trick dequant), and an
    unverified row is exactly the fast-but-wrong number the harness's
    correctness-first rule exists to keep out of a table. Comparing against the
    *same* uint8 bytes widened by ``torch`` isolates the kernel's dequant from
    fp8 rounding, which a comparison against the fp16 pool would confound.
    """
    pool = paged_pool([37, 128, 512], num_kv_heads=2, head_dim=64, dtype=torch.uint8)
    q = torch.randn(3, 8, 64, device="cuda", dtype=torch.float16) * 0.3
    widened = replace(
        pool,
        k=pool.k.view(torch.float8_e4m3fn).to(torch.float16),
        v=pool.v.view(torch.float8_e4m3fn).to(torch.float16),
    )
    verify(f"{_IMPL} [fp8_pool]", run(pool, q), run(widened, q), rtol=_RTOL, atol=_ATOL)


def show_dispatch() -> None:
    """Print which implementation dispatch would pick for this op.

    A benchmark table is only actionable if its rows name implementations the
    dispatcher can actually choose between, so the decision chain goes in the log
    next to the numbers, and the assertion below pins the table's labels to it —
    a second registered row for this op would otherwise leave the table naming
    one kernel while dispatch runs another. ``kv:paged`` is the layout tag this
    pool satisfies; without it the native row is filtered out and the explain
    line says so.
    """
    sel = dispatch(
        "attention.decode",
        dtype="fp16",
        shape={"num_kv_heads": 8, "head_dim": 128},
        layout=frozenset({"kv:paged"}),
    )
    print(f"\nDispatch for attention.decode:\n{sel.explain()}")
    assert sel.spec.name == _IMPL, f"table labels say {_IMPL}, dispatch picks {sel.spec.name}"
    assert sel.load() is not None


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    torch.set_grad_enabled(False)

    print(metadata())
    print()
    check_correctness()
    show_dispatch()
    print()

    rows: list[Row] = []
    for batch, seq_len, hq, hkv, head_dim in CASES:
        lens = [seq_len] * batch
        q = torch.randn(batch, hq, head_dim, device="cuda", dtype=torch.float16) * 0.3
        case = f"b{batch}_s{seq_len}"

        for label, kwargs in (
            ("production", {"layout": "fragmented"}),
            ("contiguous", {"layout": "contiguous"}),
            ("fp8_pool", {"layout": "fragmented", "dtype": torch.uint8}),
        ):
            pool = paged_pool(lens, num_kv_heads=hkv, head_dim=head_dim, **kwargs)
            us = bench(lambda p=pool, q=q: run(p, q))
            rows.append(Row(f"{_IMPL} [{label}]", case, us, decode_work(pool, hq)))
            del pool

        # The measurement mistake, quantified: same kernel, k/v as two separate
        # allocations. Built by copying the views out, which is precisely the
        # ``.contiguous()`` kv_pool refuses to do for the production row. It came
        # out equal to production or slower here, which is a result about this
        # geometry, not a reason to stop checking.
        pool = paged_pool(lens, num_kv_heads=hkv, head_dim=head_dim, layout="fragmented")
        split = PagedPool(
            buffer=pool.buffer,
            k=pool.k.contiguous(),
            v=pool.v.contiguous(),
            table=pool.table,
            req_idx=pool.req_idx,
            seq_lens=pool.seq_lens,
            max_seq_len=pool.max_seq_len,
            layout="fragmented",
        )
        us = bench(lambda p=split, q=q: run(p, q))
        rows.append(Row(f"{_IMPL} [split_alloc]", case, us, decode_work(pool, hq)))
        del pool, split, q
        torch.cuda.empty_cache()

    report(rows)
    print(
        "\nRead the table as: production is the kernel's speed, and the rows below it\n"
        "are bounds that turned out to be tight — contiguous is 2-4% ahead and\n"
        "split_alloc never ahead at all, so neither paging nor the combined-buffer\n"
        "stride is where decode time goes at this head geometry. fp8_pool is the row\n"
        "to read differently: half the bytes for 6-10% less time means it is\n"
        "dequant-bound, so its lower %bw is a different bottleneck rather than a\n"
        "regression."
    )


if __name__ == "__main__":
    main()
