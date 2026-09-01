"""MLA decode: reference vs FlashMLA — the flashmla row's golden source.

``build_case`` makes one latent-compressed KV cache per history length
and ``_traffic`` computes the bytes the kernel must move, so measured
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

import lite_llama.kernels
from lite_llama.models.mla_single_layer import (
    MinimalMlaLayer,
    flashmla_decode_fn,
)

#: DeepSeek-V2-Lite-flavoured geometry (the harness keeps qk == v == lora:
#: no absorption projections), cached length walked from short to long.
BATCH, HEADS, KV_LORA, HIDDEN = 8, 16, 512, 2048
PAGE_SIZE = 64
CACHE_LENS = (512, 2048, 8192)
DT = torch.bfloat16
DEVICE = "cuda"


def build_case(cache_len: int) -> tuple:
    """One harness instance with a filled latent cache at ``cache_len``."""
    torch.manual_seed(0)
    max_pages = (CACHE_LENS[-1] + PAGE_SIZE - 1) // PAGE_SIZE
    layer = MinimalMlaLayer(HIDDEN, HEADS, KV_LORA, dtype=DT, device=DEVICE).to(DEVICE).eval()
    num_pages = (cache_len + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = torch.randn(BATCH * max_pages, PAGE_SIZE, KV_LORA, dtype=DT, device=DEVICE)
    # Disjoint page runs per sequence; unused trailing pages stay garbage,
    # which is the point — a kernel that reads past cache_seqlens shows up
    # as a diff against the reference, not as a silent success.
    block_table = torch.arange(BATCH * max_pages, dtype=torch.int32, device=DEVICE).view(BATCH, -1)
    cache_seqlens = torch.full((BATCH,), cache_len, dtype=torch.int32, device=DEVICE)
    x = torch.randn(BATCH, HIDDEN, dtype=DT, device=DEVICE)
    return layer, kv_cache, block_table, cache_seqlens, x


def main() -> None:
    from lite_llama.kernels.backend import flashmla

    print(metadata())
    rows: list[Row] = []
    for cache_len in CACHE_LENS:
        layer, kv_cache, block_table, cache_seqlens, x = build_case(cache_len)
        case = f"b{BATCH}_l{cache_len}_h{HEADS}_d{KV_LORA}"
        with torch.no_grad():
            ref = layer.decode(x, kv_cache, block_table, cache_seqlens, max_seq_len=cache_len)

            # Default-arg bindings pin the loop variables at definition time
            # (B023): bench() consumes the closure inside this iteration, but
            # the static check cannot see that.
            def run_ref(
                layer=layer,
                x=x,
                kv_cache=kv_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_len=cache_len,
            ) -> torch.Tensor:
                return layer.decode(x, kv_cache, block_table, cache_seqlens, max_seq_len=cache_len)

            rows.append(
                Row(
                    "reference/mla_decode",
                    case,
                    bench(run_ref),
                    Work(moved=_traffic(kv_cache, cache_len, ref)),
                )
            )

            if not flashmla.available():
                continue
            layer.mla_decode_fn = flashmla_decode_fn
            got = layer.decode(x, kv_cache, block_table, cache_seqlens, max_seq_len=cache_len)

            def run_fi(
                layer=layer,
                x=x,
                kv_cache=kv_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_len=cache_len,
            ) -> torch.Tensor:
                return layer.decode(x, kv_cache, block_table, cache_seqlens, max_seq_len=cache_len)

            rows.append(
                Row(
                    "flashmla/mla_decode",
                    case,
                    bench(run_fi),
                    Work(moved=_traffic(kv_cache, cache_len, got)),
                )
            )
            # The tolerance window is generous on purpose: bf16 attention over
            # 8k keys; the *printed* max-abs-diff is the figure the golden
            # record freezes, the assert just catches structural breakage.
            verify(f"mla_decode l={cache_len}", got, ref, rtol=2e-2, atol=2e-2)
            layer.mla_decode_fn = layer.reference_decode

    report(rows)
    if not flashmla.available():
        print(f"\nflashmla not importable here: {flashmla.INSTALL.how_to_get_it()}")
        print("Reference path validated; the row's golden diff needs an sm90+ box.")
    else:
        print("\nThe diffs above are the flashmla row's pending GoldenRecord:")
        print("freeze them (verified=True, max_abs_diff=...) after a real run.")


def _traffic(kv_cache: torch.Tensor, cache_len: int, out: torch.Tensor) -> int:
    """Minimum traffic: the cached latents read once, q and out once each."""
    return (
        BATCH * cache_len * KV_LORA + out.numel() + BATCH * HEADS * KV_LORA
    ) * kv_cache.element_size()


if __name__ == "__main__":
    main()
