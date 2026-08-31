"""Minimal single-layer MLA harness — the flashmla row's verification vehicle.

A single MLA layer (DeepSeek-V2 style latent attention) with random weights,
deliberately ahead of the v0.10 wiring (ROADMAP: verifying MLA's structure
and numeric path needs no 671B checkpoint). It is not in
:mod:`lite_llama.models.registry` — today it answers exactly two questions:

* does the flashmla row have a golden vehicle? :meth:`MinimalMlaLayer.
  reference_decode` is the pure-PyTorch baseline; running the flashmla row
  next to it yields the ``max_abs_diff`` the row's ``GoldenRecord`` is waiting
  for (``benchmarks/kernels/bench_mla_decode.py`` does exactly that);
* what does the latent cache look like in this repo? ``q_a``/``q_b``/``kv_a``
  projections plus a head-less paged latent cache — the shape
  :class:`~lite_llama.kernels.ops.interfaces.MlaDecodeOp`'s contract spells
  out, with its own ``kv:mla_latent`` layout tag so it can never be
  dispatched against the per-head paged pool.

Structure-wise this is MLA minimised, not faithfully replicated: the
absorption projections are out of scope (``qk_head_dim == v_head_dim ==
kv_lora_rank``), q/k norms are absent, and there is no tensor parallelism.
``mla_decode_fn`` is the seam — the layer is constructed with the reference
implementation and never knows whether a caller swaps in the flashmla row.

Usage:
    layer = MinimalMlaLayer(hidden, heads, kv_lora)
    layer.write_kv(x, kv_cache, block_table, cache_seqlens)   # one step
    out = layer.decode(x, kv_cache, block_table, cache_seqlens, max_seq_len=L)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..kernels.dispatcher import dispatch

#: Layout tag of the latent cache — matches the flashmla row's requirement.
MLA_LATENT = frozenset({"kv:mla_latent"})


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
    library probe, so default dispatch would never hand it out; asking for it
    by backend bypasses the golden gate (never the physical ones) and returns
    a callable with :class:`MlaDecodeOp`'s exact signature.
    """
    sel = dispatch("attention.mla_decode", dtype=q.dtype, layout=MLA_LATENT, backend="flashmla")
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

        The golden baseline the flashmla row's ``max_abs_diff`` is measured
        against: gather each sequence's pages into a dense ``[len, lora]``
        key matrix, softmax the single-KV-head logits, mix the keys.
        """
        batch = q.shape[0]
        page_size = kv_cache.shape[1]
        out = torch.empty_like(q)
        for b in range(batch):
            length = int(cache_seqlens[b])
            num_pages = (length + page_size - 1) // page_size
            pages = block_table[b, :num_pages].tolist()
            keys = torch.cat([kv_cache[p] for p in pages], dim=0)[:length]
            logits = torch.einsum("hd,sd->hs", q[b].float(), keys.float()) * sm_scale
            probs = logits.softmax(dim=-1)
            out[b] = torch.einsum("hs,sd->hd", probs, keys.float()).to(q.dtype)
        return out

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
