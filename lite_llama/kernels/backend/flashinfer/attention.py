"""FlashInfer attention wrappers: both phases behind the native signatures.

Prefill and decode share one lazily allocated workspace and keep one wrapper
per phase (cleared via ``_reset_cache``); the functions mirror the native
kernels' signatures so dispatch can swap them in.

Usage:
    out = prefill_attention(q, k, v, sm_scale, b_start_loc, b_seq_len,
                            max_seq_len)
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Workspace FlashInfer's wrappers plan into, allocated on first use.
_WORKSPACE_BYTES = 128 * 1024 * 1024
_prefill_wrapper = None
_decode_wrapper = None
_workspace: torch.Tensor | None = None


def _reset_cache() -> None:
    """Drop cached wrappers (tests swapping devices/fakes)."""
    global _prefill_wrapper, _decode_wrapper, _workspace
    _prefill_wrapper = None
    _decode_wrapper = None
    _workspace = None


def _get_workspace() -> torch.Tensor:
    global _workspace
    if _workspace is None:
        _workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    return _workspace


def _get_wrapper(which: str) -> Callable:
    global _prefill_wrapper, _decode_wrapper
    if which == "prefill":
        if _prefill_wrapper is None:
            from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

            _prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(_get_workspace(), "NHD")
        return _prefill_wrapper
    if _decode_wrapper is None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        _decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(_get_workspace(), "NHD")
    return _decode_wrapper


def prefill_attention(q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len):
    """Causal prefill over a packed ragged batch, via FlashInfer.

    Args follow :func:`~lite_llama.kernels.ops.attention.flashattention2_nopad.
    flash_attention2_no_pad` exactly; ``max_seq_len`` only sizes the native
    kernel's grid and is re-derived here from the length vector, so callers
    cannot disagree with themselves.
    """
    _total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    # The ragged wrapper needs compact rows; the engine's prefill grid is
    # padded to the widest chunk (slot_batch.begin_prefill), so a batch of
    # unequal prompt lengths only agrees with this layout by accident. When
    # the declared packed total differs from the rows actually here, hand the
    # pass to the native kernel, which addresses each sequence at
    # b_start_loc and bounds it by b_seq_len — the exact padded contract.
    declared = int(b_start_loc[-1].item()) + int(b_seq_len[-1].item())
    if _total_q != declared:
        from ...ops.attention.flashattention2_nopad import flash_attention2_no_pad

        return flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len)
    qo_indptr = torch.cat(
        [b_start_loc.new_zeros(1), b_start_loc[1:], b_start_loc[-1:] + b_seq_len[-1:]]
    ).to(torch.int32)
    # Prefill attends over the same freshly projected rows: one kv chunk per
    # sequence, aligned with the query rows.
    kv_indptr = qo_indptr

    wrapper = _get_wrapper("prefill")
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    out = torch.empty_like(q)
    wrapper.run(q, k, v, out=out)
    return out


def decode_attention(
    q,
    k_cache,
    v_cache,
    qk_scale,
    b_req_tokens_table,
    b_req_idx,
    b_seq_len,
    max_actual_seq_len,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """Decode attention, one token per sequence, via FlashInfer.

    Args follow :func:`~lite_llama.kernels.ops.attention.flashdecoding.
    flash_decoding` exactly. The slot table is flattened into FlashInfer's
    ``indptr``/``indices`` (page_size 1: one cache row per page), and the
    fp8-KV dequantisation scales pass through as FlashInfer's per-tensor
    ``k_scale``/``v_scale`` on the run call.
    """
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    seq_lens = b_seq_len[:batch].to(torch.int32)
    # Cache-row ids of every attended token, request order flattened.
    rows = [b_req_tokens_table[b_req_idx[i], : seq_lens[i]].to(torch.int32) for i in range(batch)]
    indices = torch.cat(rows)
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    torch.cumsum(seq_lens, dim=0, out=indptr[1:])
    # page_size == 1: every page is full, so the last page of each sequence
    # holds exactly one token.
    last_page_len = torch.ones(batch, dtype=torch.int32, device=q.device)

    wrapper = _get_wrapper("decode")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=1,
        sm_scale=qk_scale,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
    )
    out = torch.empty_like(q)
    wrapper.run(
        q,
        # Zero-copy page views: [T, H, D] -> [T, 1, H, D] per pool.
        (k_cache.unsqueeze(1), v_cache.unsqueeze(1)),
        out=out,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return out
