"""Minimal single-layer MLA harness — the flashmla row's verification vehicle.

:class:`MinimalMlaLayer` is one attention layer with latent KV compression
and no MLP, small enough to diff against a reference while still driving
the real ``mla_decode`` call.

Usage:
    layer = MinimalMlaLayer(hidden_size, num_heads, kv_lora_rank)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..kernels.dispatcher import MLA_LATENT_TAGS, dispatch
from ..kernels.ops.attention.mla import mla_decode_reference


def flashmla_decode_fn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    max_seq_len: int,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    """The flashmla row, selected explicitly behind the contract signature.

    The row is ``verified=False`` and gated on ``kv:mla_latent`` plus the
    library availability check, so default dispatch would never hand it out; asking for it
    by backend bypasses the golden gate (never the physical ones) and returns
    a callable with :class:`MlaDecodeOp`'s exact signature.
    """
    sel = dispatch("attention.mla_decode", dtype=q.dtype, layout=MLA_LATENT_TAGS, backend="flashmla")
    return sel.load()(
        q, kv_cache, block_table, cache_seqlens, max_seq_len=max_seq_len, sm_scale=sm_scale
    )


class MinimalMlaLayer(nn.Module):
    """Three projections plus a pluggable decode fn over the latent cache.

    Args:
        hidden_size: Residual width feeding the projections.
        num_heads: Query heads (the latent cache is single-KV-head, MQA).
        kv_lora_rank: Latent dimension — also the qk and v head dim here,
            since the absorption projections are out of the harness's scope.
        dtype: Weight dtype; bf16 in practice.
        device: Weight device.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = kv_lora_rank
        self.v_head_dim = kv_lora_rank
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)
        self.q_a_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False, dtype=dtype, device=device)
        self.q_b_proj = nn.Linear(
            kv_lora_rank, num_heads * kv_lora_rank, bias=False, dtype=dtype, device=device
        )
        self.kv_a_proj = nn.Linear(
            hidden_size, kv_lora_rank, bias=False, dtype=dtype, device=device
        )
        # The decode seam: swap in flashmla_decode_fn to exercise the row.
        self.mla_decode_fn = self.reference_decode

    def project_q(self, x: torch.Tensor) -> torch.Tensor:
        """``[batch, hidden]`` -> ``[batch, num_heads, qk_head_dim]``."""
        q = self.q_b_proj(self.q_a_proj(x))
        return q.view(-1, self.num_heads, self.qk_head_dim)

    @staticmethod
    def reference_decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        *,
        max_seq_len: int,
        sm_scale: float = 1.0,
    ) -> torch.Tensor:
        """Pure-PyTorch decode over the paged latent cache.

        The 1.2 kernel reference with the rope segment width pinned to zero —
        this harness's cache rows are all latent, no ``[c_kv | k_pe]`` split —
        which degenerates to exactly the V-is-the-whole-row semantics the
        golden diff wants. One implementation, kept honest by the kernel's
        own tests, instead of a second one drifting beside it.
        """
        return mla_decode_reference(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            max_seq_len=max_seq_len,
            sm_scale=sm_scale,
            qk_rope_head_dim=0,
        )

    def write_kv(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        """Project this step's token to the latent and write it into its page.

        One token per sequence (the decode geometry): the write position is
        ``cache_seqlens[b]``, which then advances — the same "write then
        attend" order :class:`~lite_llama.modules.attention.PagedAttention`
        keeps for the per-head pool.
        """
        c_kv = self.kv_a_proj(x)  # [batch, lora]
        page_size = kv_cache.shape[1]
        for b in range(c_kv.shape[0]):
            pos = int(cache_seqlens[b])
            page = int(block_table[b, pos // page_size])
            kv_cache[page, pos % page_size] = c_kv[b]
            cache_seqlens[b] += 1

    def decode(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        *,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Project ``x`` to q and run the decode seam.

        Returns:
            ``[batch, num_heads, v_head_dim]``.
        """
        q = self.project_q(x)
        return self.mla_decode_fn(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            max_seq_len=max_seq_len,
            sm_scale=self.scale,
        )
