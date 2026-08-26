"""Paged attention: the KV-cache write plus the phase-appropriate kernel call.

:class:`PagedAttention` is the lower half of an attention block. It owns
everything that depends on *how K/V are stored* — the write into the paged
buffer, the fp8 quantisation on the way in, the dequantisation scales the
decode kernel needs, and the prefill/decode split — and nothing that depends on
how q/k/v were produced. The projections, q/k norm and RoPE live one level up
in :class:`~lite_llama.models.base.Attention`, so a model with a different
composition (MLA) can reuse this half, and a different cache strategy can
replace it without touching any projection.

The kernels are called by name (``update_kv_buffer``, ``flash_attention2_no_pad``,
``flash_decoding``): reading this file tells you exactly which kernel runs.
Which *backend* provides them is declared as data in
:mod:`lite_llama.kernels.backends` and resolved by
:mod:`lite_llama.kernels.ops`.

Usage:
    attn = PagedAttention(num_kv_heads, head_dim, kv_cache_dtype=torch.bfloat16)
    out = attn(xq, xk, xv, atten_info, layer_index, is_prefill=True)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..kernels import flash_attention2_no_pad, flash_decoding, update_kv_buffer
from .quantization.kv_cache import get_kv_cache_method


class PagedAttention(nn.Module):
    """Writes K/V into the paged cache and runs the phase-appropriate kernel.

    Both phases share the cache layout ``[2 * max_tokens, num_kv_heads, head_dim]``
    where the K rows occupy the first half and the V rows the second.

    Args:
        num_kv_heads: Number of key/value heads on this rank (may be smaller
            than the query head count for grouped-query attention). Needed here
            to split the buffer's K half from its V half.
        head_dim: Size of a single attention head; sets the softmax scale.
        kv_cache_dtype: Element type of the cache — the activation dtype stores
            K/V verbatim; ``torch.uint8`` stores e4m3 bytes (vLLM's fp8 KV
            cache), quantised here on write and widened by the decode kernel.
            The dequantisation scales come with the strategy object that
            :func:`~lite_llama.modules.quantization.kv_cache.get_kv_cache_method`
            returns, so write and read cannot disagree about them.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_method = get_kv_cache_method(kv_cache_dtype)
        # Read the scales off the strategy that applied them. They used to be two
        # more constructor arguments, which no caller ever passed: the fp8 cache
        # was numerically correct only because that default and the method's own
        # default were both 1.0, and a per-tensor scale would have gone missing
        # silently the moment either side started computing one.
        method = self.kv_cache_method
        self.k_scale = method.k_scale if method is not None else 1.0
        self.v_scale = method.v_scale if method is not None else 1.0

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

        update_kv_buffer(xk, xv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index])

    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
    ) -> torch.Tensor:
        """Prefill: causal attention over the freshly computed q/k/v.

        The kernel reads the unquantised tensors it was handed, not the cache —
        the write above only has to make them available to later decode steps.

        Returns:
            ``[tokens, num_heads, head_dim]``.
        """
        self._write_cache(xk, xv, atten_info, layer_index)
        return flash_attention2_no_pad(
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
