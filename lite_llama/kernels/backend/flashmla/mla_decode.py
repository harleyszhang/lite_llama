"""FlashMLA decode — the ``flashmla/mla_decode`` row's wrapper.

``mla_decode`` maps the paged KV buffer, block table and per-sequence
lengths onto FlashMLA's arguments, so latent-attention decode runs
against the same cache the native kernels use.

Usage:
    out = mla_decode(q, kv_cache, block_table, cache_seqlens)
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
