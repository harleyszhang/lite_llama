"""Paged attention: the KV-cache write plus the phase-appropriate kernel call.

:class:`PagedAttention` owns no math: it scatters fresh K/V rows into the
paged buffer, then dispatches prefill (varlen) or decode (paged) to the
kernel layer using the step's :class:`AttentionMetadata`.

Usage:
    attn = PagedAttention(num_kv_heads, head_dim, kv_cache_dtype, dtype)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..kernels import dispatch
from ..kernels.dispatcher import PAGED_KV_TAGS
from .quantization.kv_cache import get_kv_cache_method


class PagedAttention(nn.Module):
    """Writes K/V into the paged cache and runs the phase-appropriate kernel.

    Both phases share the cache layout ``[2 * max_tokens, num_kv_heads, head_dim]``:
    K rows occupy the first half, V rows the second. ``kv_cache_dtype`` of
    ``torch.uint8`` stores fp8-e4m3 bytes; the dequantisation scales come from
    the strategy object that :func:`~lite_llama.modules.quantization.kv_cache.get_kv_cache_method`
    returns, so write and read cannot disagree about them.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: torch.dtype = torch.bfloat16,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_method = get_kv_cache_method(kv_cache_dtype)
        # Scales live on the strategy, not on this layer: the quantise-on-write
        # and dequantise-on-read halves are only correct as a pair.
        method = self.kv_cache_method
        self.k_scale = method.k_scale if method is not None else 1.0
        self.v_scale = method.v_scale if method is not None else 1.0

        # One dispatch decision per op, held for the module's lifetime. kv_write
        # carries no dtype window on purpose: with an fp8 cache the K/V arrive
        # already quantised, so uint8 rows are as legal as bf16 ones. The decode
        # op's scheme key encodes the cache dtype this module writes.
        self._kv_write = dispatch("kv_write", dtype=kv_cache_dtype, layout=PAGED_KV_TAGS).load()
        self._prefill = dispatch("attention.prefill", dtype=dtype).load()
        self._chunked = dispatch("attention.chunked_prefill", dtype=dtype).load()
        self._decode = dispatch(
            "attention.decode",
            dtype=dtype,
            scheme="fp8_kv" if kv_cache_dtype == torch.uint8 else "unquantized",
            layout=PAGED_KV_TAGS,
        ).load()

    def _write_cache(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> None:
        """Scatter this step's K/V into their allocated cache rows."""
        if self.kv_cache_method is not None:
            xk, xv = self.kv_cache_method.quantize_kv(xk, xv)

        self._kv_write(xk, xv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index])

    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Prefill: causal attention over the freshly computed q/k/v.

        The plain kernel reads the unquantised tensors it was handed, not the
        cache — the write above only has to make them available to later decode
        steps. A chunk resuming on cached rows (``b_prefix_len`` armed) cannot
        use it: self-attention over the grid alone would drop the prefix. Its
        queries instead run through the chunked kernel, whose keys and values
        are the slot's own cache rows — prefix plus this chunk, contiguous from
        ``b_kv_base`` — at tensor-core prefill prices rather than the one-row-
        per-token extend path.

        Returns:
            ``[tokens, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        if atten_info.b_prefix_len is not None:
            kv_buffer = atten_info.kv_buffer[layer_index]
            return self._chunked(
                xq,
                kv_buffer[:, : self.num_kv_heads, :],
                kv_buffer[:, self.num_kv_heads :, :],
                self.scale,
                atten_info.b_start_loc,
                atten_info.b_kv_base,
                atten_info.b_prefix_len,
                atten_info.b_seq_len,
                atten_info.max_chunk_len,
            )
        return self._prefill(
            xq,
            xk,
            xv,
            self.scale,
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
        return self._decode(
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
