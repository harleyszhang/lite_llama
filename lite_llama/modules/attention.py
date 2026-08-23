"""Attention modules: fused-QKV projection, RoPE, paged KV cache I/O.

:class:`Attention` is the self-attention building block shared by every decoder
model; :class:`PagedAttention` is its lower half — the part that owns the KV
cache write and picks the prefill/decode kernel. They are separate classes so a
model variant can swap the cache strategy without touching the projections.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..distributed.parallel_state import divide, get_tp_world_size
from ..kernels import (
    flash_attention2_no_pad,
    flash_decoding,
    rope_emb_forward,
    skip_rmsnorm,
    update_kv_buffer,
)
from ..models.config import ModelConfig
from ..models.quantization import QuantConfig
from ..models.quantization.params import quantize_fp8_per_tensor
from .linear import ColumnParallelLinear, RowParallelLinear

# The prefill kernel evaluates exp2 rather than exp, so its softmax scale has to
# absorb log2(e). The decode kernel uses exp directly and takes the plain scale.
_LOG2E = 1.4426950408889634


class PagedAttention(nn.Module):
    """Writes K/V into the paged cache and runs the phase-appropriate kernel.

    Both phases share the cache layout ``[max_tokens, 2 * num_kv_heads, head_dim]``
    where the K heads occupy the first half and the V heads the second.

    Args:
        num_kv_heads: Number of key/value heads (may be smaller than the query
            head count for grouped-query attention).
        head_dim: Size of a single attention head.
        kv_cache_dtype: Element type of the cache — ``torch.float16`` stores
            K/V verbatim; ``torch.uint8`` stores e4m3 bytes (vLLM's fp8 KV
            cache), quantised here on write and widened by the decode kernel.
        k_scale: Per-tensor dequantisation scale of the fp8 key cache.
        v_scale: Same for the value cache.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: torch.dtype = torch.float16,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.kv_cache_dtype = kv_cache_dtype
        self.k_scale = k_scale
        self.v_scale = v_scale

    def _write_cache(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> None:
        if self.kv_cache_dtype == torch.uint8:
            # fp8 KV cache: e4m3 bytes travel in a uint8 container. Quantising
            # on write keeps update_kv_buffer a pure byte scatter for both
            # dtypes; the decode kernel widens on read.
            xk = quantize_fp8_per_tensor(xk, self.k_scale)
            xv = quantize_fp8_per_tensor(xv, self.v_scale)
        # (tokens, 2 * num_kv_heads, head_dim) — K heads first, then V heads.
        combined_kv = torch.cat([xk, xv], dim=-2)
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )

    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Prefill: causal attention over the freshly computed q/k/v.

        Returns:
            ``[tokens, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        return flash_attention2_no_pad(
            xq,
            xk,
            xv,
            self.scale * _LOG2E,
            atten_info.b_start_loc,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
        )

    def token_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Decode: attend the single new query against the whole cached history.

        Returns:
            ``[batch, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        kv_buffer = atten_info.kv_buffer[layer_index]
        return flash_decoding(
            xq,
            kv_buffer[:, : self.num_kv_heads, :],
            kv_buffer[:, self.num_kv_heads :, :],
            self.scale,
            atten_info.b_req_tokens_table,
            atten_info.b_req_idx,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        if is_prefill:
            return self.context_forward(xq, xk, xv, atten_info, layer_index)
        return self.token_forward(xq, xk, xv, atten_info, layer_index)


class Attention(nn.Module):
    """Fused-QKV self-attention with RoPE and optional per-head q/k normalisation.

    Under tensor parallelism the heads are dealt out across ranks: q/k/v are
    column-parallel (each rank owns ``num_heads / tp`` query heads and the KV heads
    that go with them, so its KV cache is that much smaller too) and ``o_proj`` is
    row-parallel, contributing this block's only all-reduce.

    Args:
        config: Model config supplying the head geometry.
        qkv_bias: Whether q/k/v projections carry a bias (true for Qwen2).
        use_qk_norm: Whether q and k are RMSNormed per head before RoPE (Qwen3).
        quant: Quantisation layout of the projections, or ``None`` for fp16.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        quant: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.num_heads = divide(config.num_heads, tp_size, "attention heads")
        self.num_kv_heads = divide(config.num_kv_heads, tp_size, "key/value heads")
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        # Attention width is independent of the residual stream width (Qwen3),
        # and these are this rank's share of it.
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.use_qk_norm = use_qk_norm

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, config.q_size, bias=qkv_bias, quant=quant, what="query features"
        )
        # K and V fused along dim 0 so one split yields both. Each rank assembles
        # its own fused pair from its slice of k_proj and of v_proj.
        self.kv_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * config.kv_size,
            bias=qkv_bias,
            quant=quant,
            what="key/value features",
        )
        self.o_proj = RowParallelLinear(
            config.q_size, self.hidden_size, quant=quant, what="query features"
        )

        if use_qk_norm:
            # Normalises over head_dim, so it is replicated rather than sharded.
            self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float16))
            self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float16))

        self.attn = PagedAttention(
            self.num_kv_heads, self.head_dim, kv_cache_dtype=config.kv_cache_torch_dtype
        )

    def _project_qkv(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, reshape to per-head layout, normalise (optionally) and apply RoPE."""
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.hidden_size)

        xq = self.q_proj(x)
        xkv = self.kv_proj(x)
        xk, xv = torch.split(xkv, self.kv_size, dim=-1)

        num_tokens = batch_size * seq_len
        xq = xq.view(num_tokens, self.num_heads, self.head_dim)
        xk = xk.view(num_tokens, self.num_kv_heads, self.head_dim)
        xv = xv.view(num_tokens, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            # RMSNorm over head_dim, i.e. independently per head.
            xq, _ = skip_rmsnorm(xq, None, self.q_norm_weight, self.rms_norm_eps)
            xk, _ = skip_rmsnorm(xk, None, self.k_norm_weight, self.rms_norm_eps)

        cos, sin = position_embeddings
        xq, xk = rope_emb_forward(xq, xk, cos, sin, batch_size, seq_len)
        return xq, xk, xv

    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self._project_qkv(x, position_embeddings)

        # seq_len == 1 means we are decoding and can read the whole cached history.
        attn_output = self.attn(xq, xk, xv, atten_info, layer_index, is_prefill=seq_len > 1)
        # Back to the residual-stream layout before the output projection.
        attn_output = attn_output.view(batch_size, seq_len, self.q_size)
        return self.o_proj(attn_output)
