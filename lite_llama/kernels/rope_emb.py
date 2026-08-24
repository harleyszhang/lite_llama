"""Rotary position embedding (RoPE) applied to q and k in a fused Triton kernel.

Rotates each head's channel pairs by the per-position angle in place, consuming the
``(cos, sin)`` tables built by :mod:`lite_llama.modules.rotary_embedding`.

Each token is located through ``stride(0)`` alone, so q and k may be *column slices of
a fused qkv buffer*: their heads still sit next to each other, only the distance from
one token to the next is larger. Such a tensor is rotated where it lies instead of being
copied out and back, which is what keeps the fused QKV projection
(:class:`~lite_llama.modules.linear.QKVParallelLinear`) a net win.

Usage:
    q, k = rope_emb_forward(q, k, cos, sin, batch_size, seq_len)
"""

import triton
import triton.language as tl


@triton.jit
def _triton_rope_emb(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_b_stride,
    cos_s_stride,
    sin,
    sin_b_stride,
    sin_s_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // sl
    cos_row_idx = pid % sl

    # 定位到 q, k 行起点
    q_ptr += pid * q_row_stride
    k_ptr += pid * k_row_stride

    # 定位到 cos, sin 对应 batch_id 的 cos_row_idx 行
    cos_ptr = cos + batch_id * cos_b_stride + cos_row_idx * cos_s_stride
    sin_ptr = sin + batch_id * sin_b_stride + cos_row_idx * sin_s_stride

    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos_ptr + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin_ptr + cos_offsets, mask=cos_mask, other=0)

    # 计算 head 和 dim 偏移
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]

    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def _rows_are_contiguous(tensor) -> bool:
    """Whether ``head * head_dim + channel`` addresses a token's row correctly.

    The kernel walks tokens with ``stride(0)`` and heads with ``head_dim``, so anything
    whose heads are adjacent within a row qualifies — including a slice of a fused
    ``[q | k | v]`` output, whose row stride is simply wider than its row.
    """
    return tensor.stride(2) == 1 and tensor.stride(1) == tensor.shape[2]


def rope_emb_forward(q, k, cos, sin, batch_size, seq_len):
    """
    q: (batch_size * seq_len, n_q_heads, head_dim)
    k: (batch_size * seq_len, n_k_heads, head_dim)
    cos, sin: (batch_size, seq_len, head_dim)

    q and k are rotated in place and returned; a tensor whose heads are not adjacent
    within a row is materialised first, in which case the copy is what comes back.
    """
    N, n_qh, HEAD_DIM = q.shape
    _, n_kh, _ = k.shape
    assert batch_size * seq_len == N

    pad_hd = triton.next_power_of_2(HEAD_DIM)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    BLOCK_SIZE = max(pad_n_qh, pad_n_kh)

    if HEAD_DIM >= 128:
        num_warps = 8
    else:
        num_warps = 4

    q = q if _rows_are_contiguous(q) else q.contiguous()
    k = k if _rows_are_contiguous(k) else k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_rope_emb[(N,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        cos,
        cos.stride(0),
        cos.stride(1),
        sin,
        sin.stride(0),
        sin.stride(1),
        seq_len,
        batch_size,
        n_qh,
        n_kh,
        HEAD_DIM,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, k
