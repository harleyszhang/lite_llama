"""MLA decode: reference vs native Triton vs FlashMLA at DeepSeek geometry.

The kernels are called directly — no projections, no module tree: the bench
compares implementations of the same op over the same latent pool. The
geometry is the real one (``kv_lora_rank + qk_rope_head_dim = 576``-wide
rows), and ``_traffic`` computes the bytes the kernel must move, so measured
milliseconds and roofline seconds sit side by side.

Usage:
    python benchmarks/kernels/bench_mla_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microbench import Row, Work, bench, metadata, report, verify

from rapid_llm.kernels.dispatcher import MLA_LATENT_TAGS, dispatch
from rapid_llm.kernels.ops.attention.mla import (
    QK_ROPE_HEAD_DIM,
    mla_decode,
    mla_decode_reference,
)

#: DeepSeek-V2-Lite-flavoured geometry: the latent row is the 512-wide c_kv
#: plus the 64-wide rope key, and the absorbed q is 576 wide per head.
BATCH, HEADS, KV_LORA = 8, 16, 512
ROPE = QK_ROPE_HEAD_DIM
PAGE_SIZE = 64
CACHE_LENS = (512, 2048, 8192)
DT = torch.bfloat16
DEVICE = "cuda"


def build_case(cache_len: int) -> tuple:
    """One latent pool at ``cache_len`` plus the query batch attending it."""
    torch.manual_seed(0)
    max_pages = (CACHE_LENS[-1] + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = (cache_len + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = torch.randn(BATCH * max_pages, PAGE_SIZE, KV_LORA + ROPE, dtype=DT, device=DEVICE)
    # Disjoint page runs per sequence; unused trailing pages stay garbage,
    # which is the point — a kernel that reads past cache_seqlens shows up
    # as a diff against the reference, not as a silent success.
    block_table = torch.arange(BATCH * max_pages, dtype=torch.int32, device=DEVICE).view(BATCH, -1)
    cache_seqlens = torch.full((BATCH,), cache_len, dtype=torch.int32, device=DEVICE)
    q = torch.randn(BATCH, HEADS, KV_LORA + ROPE, dtype=DT, device=DEVICE)
    return q, kv_cache, block_table, cache_seqlens, num_pages


def main() -> None:
    from rapid_llm.kernels.backend import flashmla

    print(metadata())
    rows: list[Row] = []
    for cache_len in CACHE_LENS:
        q, kv_cache, block_table, cache_seqlens, _ = build_case(cache_len)
        case = f"b{BATCH}_l{cache_len}_h{HEADS}_d{KV_LORA}"
        sm_scale = (KV_LORA + ROPE) ** -0.5
        with torch.no_grad():
            ref = mla_decode_reference(
                q,
                kv_cache,
                block_table,
                cache_seqlens,
                max_seq_len=cache_len,
                sm_scale=sm_scale,
            )

            # Default-arg bindings pin the loop variables at definition time
            # (B023): bench() consumes the closure inside this iteration, but
            # the static check cannot see that.
            def run_ref(
                q=q,
                kv_cache=kv_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_len=cache_len,
                sm_scale=sm_scale,
            ) -> torch.Tensor:
                return mla_decode_reference(
                    q,
                    kv_cache,
                    block_table,
                    cache_seqlens,
                    max_seq_len=cache_len,
                    sm_scale=sm_scale,
                )

            rows.append(
                Row(
                    "reference/mla_decode",
                    case,
                    bench(run_ref),
                    Work(moved=_traffic(kv_cache, cache_len, ref)),
                )
            )

            def run_native(
                q=q,
                kv_cache=kv_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_len=cache_len,
                sm_scale=sm_scale,
            ) -> torch.Tensor:
                return mla_decode(
                    q,
                    kv_cache,
                    block_table,
                    cache_seqlens,
                    max_seq_len=cache_len,
                    sm_scale=sm_scale,
                )

            got = run_native()
            verify(f"native mla_decode l={cache_len}", got, ref, rtol=2e-2, atol=2e-2)
            rows.append(
                Row(
                    "triton/mla_decode",
                    case,
                    bench(run_native),
                    Work(moved=_traffic(kv_cache, cache_len, got)),
                )
            )

            if not flashmla.available():
                continue
            flashmla_decode = dispatch(
                "attention.mla_decode", dtype=DT, layout=MLA_LATENT_TAGS, backend="flashmla"
            ).load()

            def run_fi(
                q=q,
                kv_cache=kv_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_len=cache_len,
                sm_scale=sm_scale,
                fn=flashmla_decode,
            ) -> torch.Tensor:
                return fn(
                    q,
                    kv_cache,
                    block_table,
                    cache_seqlens,
                    max_seq_len=cache_len,
                    sm_scale=sm_scale,
                )

            got = run_fi()
            # The tolerance window is generous on purpose: bf16 attention over
            # 8k keys; the *printed* max-abs-diff is the figure the golden
            # record freezes, the assert just catches structural breakage.
            verify(f"flashmla mla_decode l={cache_len}", got, ref, rtol=2e-2, atol=2e-2)
            rows.append(
                Row(
                    "flashmla/mla_decode",
                    case,
                    bench(run_fi),
                    Work(moved=_traffic(kv_cache, cache_len, got)),
                )
            )

    report(rows)
    if not flashmla.available():
        print(f"\nflashmla not importable here: {flashmla.INSTALL.how_to_get_it()}")
        print("Reference and native rows validated; the flashmla row needs an sm90+ box.")
    else:
        print("\nThe diffs above are the flashmla row's pending GoldenRecord:")
        print("freeze them (verified=True, max_abs_diff=...) after a real run.")


def _traffic(kv_cache: torch.Tensor, cache_len: int, out: torch.Tensor) -> int:
    """Minimum traffic: the cached rows read once, q and out once each."""
    return (
        BATCH * cache_len * (KV_LORA + ROPE) + out.numel() + BATCH * HEADS * (KV_LORA + ROPE)
    ) * kv_cache.element_size()


if __name__ == "__main__":
    main()
