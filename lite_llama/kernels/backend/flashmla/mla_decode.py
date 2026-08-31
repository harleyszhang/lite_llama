"""FlashMLA decode — the ``flashmla/mla_decode`` row's wrapper.

The contract is
:class:`~lite_llama.kernels.ops.interfaces.MlaDecodeOp` — an op with no native
row at all: MLA's latent cache (``kv_cache`` pages are ``[page_size,
kv_lora_dim]``, no head axis) is a different layout from the per-head paged
pool the native attention rows share, and the tree has no MLA model to host a
Triton fallback. FlashMLA is the only implementation.

Two shape conversions, both views: ``q`` ``[bsz, heads, qk_dim]`` becomes
``[bsz, 1, heads, qk_dim]``, and the latent cache gains its implicit single KV
head — ``[pages, page, lkv]`` → ``[pages, page, 1, lkv]``.

``get_mla_metadata`` is the tile-scheduler handle: the wrapper recomputes it
per call from ``cache_seqlens`` (one small kernel) and feeds it to
``flash_mla_with_kvcache``, so callers never manage a handle across the
decode loop.

The row is ``verified=False``: this wrapper follows the upstream README's
calling convention but has never run on Hopper — the minimal single-layer MLA
harness (``models/mla_single_layer.py``) is the vehicle that will produce the
golden comparison.
"""

from __future__ import annotations

import torch


def mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    max_seq_len: int,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    """Decode attention over the MLA latent KV cache via FlashMLA.

    Args:
        q: ``[batch, num_heads, qk_head_dim]`` query for this step.
        kv_cache: ``[num_pages, page_size, kv_lora_dim]`` latent cache.
        block_table: ``[batch, max_pages]`` page ids per sequence.
        cache_seqlens: ``[batch]`` cached length per sequence.
        max_seq_len: Longest row, sizing the kernel grid.
        sm_scale: Softmax scale.

    Returns:
        ``[batch, num_heads, v_head_dim]`` attention output.
    """
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    if q.dim() != 3:
        raise ValueError(f"mla_decode wants q [batch, heads, qk_dim], got {tuple(q.shape)}")
    _batch, num_heads, qk_dim = q.shape
    # The latent cache has no head axis; FlashMLA's kernel wants the implicit
    # single KV head (MQA over c_kv) materialised.
    kv = kv_cache if kv_cache.dim() == 4 else kv_cache.unsqueeze(2)
    num_kv_heads = kv.shape[2]

    # Per-batch tile-scheduler handle; recomputed here so the caller's decode
    # loop stays handle-free.
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, num_heads * qk_dim // 64, num_kv_heads
    )
    out = flash_mla_with_kvcache(
        q.unsqueeze(1),  # [bsz, 1, heads, qk_dim]
        kv,
        block_table,
        cache_seqlens,
        max_seq_len,
        sm_scale,
        tile_scheduler_metadata,
        num_splits,
    )  # [bsz, 1, heads, v_dim]
    return out.squeeze(1)
