"""FlashAttention-2 (Triton) for variable-length, unpadded prefill batches.

Sequences are packed without padding and indexed by cumulative-length offsets.
Uses the v2 work partition (parallelise over query blocks, rescale softmax once).

``sm_scale`` is the plain ``1 / sqrt(head_dim)``: the inner loop evaluates
``exp2`` rather than ``exp``, and folding ``log2(e)`` into the scale is this
function's own business — the caller passes the same number every attention
backend takes.

Usage:
    out = flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len)
"""

import torch
import triton
import triton.language as tl
from torch.amp import custom_fwd

#: ``exp(x) == exp2(x * log2(e))`` — the kernel takes the exp2 route.
_LOG2E = 1.4426950408889634

configs_tma = [
    triton.Config({"BLOCK_M_SIZE": BM, "BLOCK_N_SIZE": BN}, num_stages=stages, num_warps=warps)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for warps in [4, 8, 16]
    for stages in [2, 3, 4, 6]
]


def keep_tma(conf):
    BLOCK_M_SIZE = conf.kwargs["BLOCK_M_SIZE"]
    BLOCK_N_SIZE = conf.kwargs["BLOCK_N_SIZE"]
    return not (
        torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M_SIZE * BLOCK_N_SIZE < 128 * 128
        and conf.num_warps == 8
    )


# key 参数列表(['B_Seqlen', 'HEAD_DIM'])的值会直接影响最佳配置的选择，因为不同的输入尺寸或问题规模可能需要不同的内核调度策略。
# @triton.autotune(
#     configs=list(filter(keep_tma, configs_tma)),
#     key=['B_Seqlen', 'HEAD_DIM']
# )
@triton.jit
def flash_attention2_nopad_kernel(
    Q,
    K,
    V,
    O,
    B_Start_Loc,
    B_Seqlen,
    sm_scale,
    heads,
    num_kv_groups,  # group of kv heads
    stride_q_bs,
    stride_q_heads,
    stride_q_dim,  # Q 的 strides
    stride_k_bs,
    stride_k_heads,
    stride_k_dim,  # K 的 strides
    stride_v_bs,
    stride_v_heads,
    stride_v_dim,  # V 的 strides
    stride_o_bs,
    stride_o_heads,
    stride_o_dim,
    HEAD_DIM: tl.constexpr,  # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr,  # BLOCK size of m_size dimension，即 Q 矩阵行数分成了m_size // BLOCK_M_SIZE 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr,  # n_size dimension
):
    """
    flashattentionv1 内核实现, 支持 nopad 计算, 输入为 3 维张量
    """
    block_m_idx = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch_idx = cur_bh // heads
    cur_head_idx = cur_bh % heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups

    # 计算当前批次的序列长度和请求序列的起始位置
    cur_seq_len = tl.load(B_Seqlen + cur_batch_idx)
    # cur_seq_start_loc = tl.load(b_req_tokens_table + cur_batch_idx * stride_req_to_tokens_b)
    cur_seq_start_loc = tl.load(B_Start_Loc + cur_batch_idx)

    block_start_loc = block_m_idx * BLOCK_M_SIZE  # 计算当前 block 的起始和结束索引

    offs_n = tl.arange(0, BLOCK_N_SIZE)  # head_dim 维度偏移
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M_SIZE)

    # Compute offsets for the first block on matrix Q K V Output
    q_offs = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_q_bs
        + cur_head_idx * stride_q_heads
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_seq_len, other=0.0)

    k_offs = (
        offs_n[None, :] * stride_k_bs
        + cur_kv_head_idx * stride_k_heads
        + offs_d[:, None] * stride_k_dim
    )
    v_offs = (
        offs_n[:, None] * stride_v_bs
        + cur_kv_head_idx * stride_v_heads
        + offs_d[None, :] * stride_v_dim
    )

    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    # 初始化用于计算 softmax 归一化项的 m 和 d, 意义见 online-softmax, 这里
    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, HEAD_DIM), dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M_SIZE, cur_seq_len)

    # 每次循环按 BLOCK_N_SIZE 来处理 K, V 的列（即 key/value 的序列维度）。
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)
        # 计算 qk^t
        k = tl.load(
            k_ptrs + (cur_seq_start_loc + start_n) * stride_k_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )

        qk = tl.dot(q, k)

        # 应用因果遮罩, 下三角矩阵 causal mask
        casual_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(casual_mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))  # 求 qk 的最大值
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)  # qk - m_ij[:, None]更新为安全的 qk 分子项
        d_ij = tl.sum(p, 1)  # 1d vector

        # -- 更新归一化项 d_new
        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij

        # -- update output accumulator --
        acc = acc * alpha[:, None]  # acc scaling

        # compute O = PV
        v = tl.load(
            v_ptrs + (cur_seq_start_loc + start_n) * stride_v_bs,
            mask=(start_n + offs_n[:, None]) < block_end_loc,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        # update the normalizer (l and d) for next iteration
        m_i = m_ij

    acc = acc / d_i[:, None]
    off_o = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_o_bs
        + cur_head_idx * stride_o_heads
        + offs_d[None, :] * stride_o_dim
    )
    out_ptrs = O + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_seq_len)


# --------------------------------------
# Flashattention NoPad 实现（Triton 内核）
# --------------------------------------
@torch.no_grad()
@custom_fwd(device_type="cuda")
def flash_attention2_no_pad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale,
    b_start_loc,
    b_seq_len,
    max_seq_len,
):
    """Causal prefill attention over a packed (no-pad) ragged batch.

    Args:
        q: ``[total_tokens, num_heads, head_dim]`` packed query rows.
        k: ``[total_kv, num_kv_heads, head_dim]`` freshly projected keys; ``v``
            has the same layout. Grouped-query attention is handled inside the
            kernel from the head-count ratio.
        v: Value rows, laid out like ``k``.
        sm_scale: Plain softmax scale, ``1 / sqrt(head_dim)``.
        b_start_loc: ``[batch]`` first packed row of each sequence.
        b_seq_len: ``[batch]`` true length of each sequence.
        max_seq_len: Longest sequence, sizing the query-block grid.

    Returns:
        ``[total_tokens, num_heads, head_dim]`` attention output. fp32 inputs
        are not supported (``custom_fwd`` casts them down).
    """
    output = torch.empty_like(q)
    batchs = b_seq_len.shape[0]
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]

    # Autotune lookup: use persisted best config if available, else heuristic.
    from ...dispatcher.autotune import get_best_config

    dtype_key = "bf16" if q.dtype == torch.bfloat16 else "fp16"
    tuned = get_best_config(
        "flash_attn_nopad", m=max_seq_len, n=HEAD_DIM, k=HEAD_DIM, dtype=dtype_key
    )
    if tuned is not None:
        BLOCK_M = tuned.get("BLOCK_M_SIZE", 64)
        BLOCK_N = tuned.get("BLOCK_N_SIZE", 64)
        num_warps = tuned.get("num_warps", 4)
        num_stages = tuned.get("num_stages", 1)
    else:
        BLOCK_M = 64  # For Ampere Architecture, 3090ti, set 128
        BLOCK_N = 64
        num_warps = 4 if HEAD_DIM <= 64 else 8
        num_stages = 1

    num_kv_groups = q.shape[1] // k.shape[1]  # num_q_heads // num_k_heads
    grid = (triton.cdiv(max_seq_len, BLOCK_M), batchs * n_heads, 1)

    flash_attention2_nopad_kernel[grid](
        q,
        k,
        v,
        output,
        b_start_loc,
        b_seq_len,
        # Fold log2(e) here, once, instead of trusting every caller to know
        # that this kernel's softmax runs on exp2.
        sm_scale * _LOG2E,
        n_heads,
        num_kv_groups,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=HEAD_DIM,
        BLOCK_M_SIZE=BLOCK_M,
        BLOCK_N_SIZE=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
