import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_decoding_stage1_kernel(
    Q,
    K,
    V,
    sm_scale,
    actual_seq_len,  # 实际序列长度
    Mid_O,
    Mid_O_LogExpSum,
    q_bs_stride,
    q_heads_stride,
    q_dim_stride,  # Q 的 strides
    k_bs_stride,
    k_heads_stride,
    k_dim_stride,  # K 的 strides
    v_bs_stride,
    v_heads_stride,
    v_dim_stride,  # V 的 strides
    mido_batch_stride,
    mido_heads_stride,
    mido_partitions_stride,
    mido_dim_stride,
    mido_les_batch_stride,
    mido_les_heads_stride,
    mido_les_partitions_stride,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Flash Attention Stage1 Triton Kernel"""
    # 获取当前程序的 block 在各个维度上的索引
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_block_idx = tl.program_id(2)

    # 计算当前批次的起始位置
    cur_batch_start_loc = batch_idx * actual_seq_len

    # 计算当前分区的起始和结束索引
    cur_batch_partition_start_index = seq_block_idx * BLOCK_SEQ
    cur_batch_partition_end_index = tl.minimum(
        actual_seq_len, cur_batch_partition_start_index + BLOCK_SEQ
    )

    # 计算需要处理的块数
    num_blocks = (
        cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1
    ) // BLOCK_N

    # 初始化偏移向量
    offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_DMODEL)  # [BLOCK_DMODEL]

    # 计算 Q 的偏移量
    q_offs = batch_idx * q_bs_stride + head_idx * q_heads_stride + offs_d * q_dim_stride

    # 计算 K 和 V 的偏移量
    k_offs = (
        (cur_batch_start_loc + offs_n[:, None]) * k_bs_stride
        + head_idx * k_heads_stride
        + offs_d[None, :] * k_dim_stride
    )

    v_offs = (
        (cur_batch_start_loc + offs_n[:, None]) * v_bs_stride
        + head_idx * v_heads_stride
        + offs_d[None, :] * v_dim_stride
    )

    # 获取指针
    q_ptrs = Q + q_offs
    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    # 加载 Q 向量
    q = tl.load(q_ptrs)  # [BLOCK_DMODEL]

    # 初始化归一化项和累加器
    d_i = 0.0  # 标量
    m_i = -float("inf")  # 标量
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)  # [BLOCK_DMODEL]

    # 迭代处理每个块
    for start_n in range(num_blocks):
        offs_n_new = start_n * BLOCK_N + offs_n  # [BLOCK_N]
        # 生成 K 的掩码
        k_mask = offs_n_new < cur_batch_partition_end_index  # [BLOCK_N]

        # 加载 K 和 V
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_DMODEL]
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_DMODEL]

        # 计算 qk^T
        qk = tl.sum(q * k, axis=1)  # [BLOCK_N]
        qk = qk * sm_scale
        qk = tl.where(k_mask, qk, float("-inf"))  # [BLOCK_N]

        # 更新最大值项和 qk 项
        current_max = tl.max(qk)  # 标量
        m_ij = tl.maximum(m_i, current_max)  # 标量
        qk = qk - m_ij  # [BLOCK_N]

        # 更新归一化项
        p = tl.exp(qk)  # [BLOCK_N]
        alpha = tl.exp(m_i - m_ij)  # 标量
        d_i = d_i * alpha + tl.sum(p)  # 标量

        # 更新 attention 输出累加器
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)  # [BLOCK_DMODEL]
        # acc = acc * alpha + tl.dot(p, v)  # [BLOCK_DMODEL]

        # 更新归一化器
        m_i = m_ij

        # 更新 K 和 V 的指针
        k_ptrs += BLOCK_N * k_bs_stride
        v_ptrs += BLOCK_N * v_bs_stride

    # 计算是否需要存储
    need_store = num_blocks > 0  # 标量布尔值

    # 计算存储的偏移量
    off_mid_o = (
        batch_idx * mido_batch_stride
        + head_idx * mido_heads_stride
        + seq_block_idx * mido_partitions_stride
        + offs_d * mido_dim_stride
    )

    off_mid_o_les = (
        batch_idx * mido_les_batch_stride
        + head_idx * mido_les_heads_stride
        + seq_block_idx * mido_les_partitions_stride
    )

    # 计算最终的 attention 输出和 log-sum-exp
    part_atten_out = acc / d_i  # [BLOCK_DMODEL]
    logexpsum = m_i + tl.log(d_i)  # 标量

    # 条件存储
    part_atten_out = tl.where(need_store, part_atten_out, 0.0)  # [BLOCK_DMODEL]
    logexpsum = tl.where(need_store, logexpsum, float("-inf"))  # 标量

    # 存储结果
    tl.store(Mid_O + off_mid_o, part_atten_out, mask=need_store)
    tl.store(Mid_O_LogExpSum + off_mid_o_les, logexpsum, mask=need_store)


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,  # Q: [batchs, num_heads, head_dim], K, V: [batchs * seq_len, num_heads, head_dim]
    actual_seq_len,  # 实际的序列长度
    mid_o,
    mid_o_logexpsum,  # Mid_O: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE), head_dim], Mid_O_LogExpSum: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE)]
    PARTITION_SIZE,
):
    BLOCK_N_SIZE = 32
    BLOCK_DMODEL = q.shape[-1]
    assert PARTITION_SIZE % BLOCK_N_SIZE == 0, (
        "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数"
    )

    batchs, num_heads, head_dim = q.shape
    sm_scale = 1.0 / (head_dim**0.5)
    grid = (batchs, num_heads, triton.cdiv(actual_seq_len, PARTITION_SIZE))

    _flash_decoding_stage1_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        actual_seq_len,  # 使用实际序列长度
        mid_o,
        mid_o_logexpsum,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        BLOCK_SEQ=PARTITION_SIZE,
        BLOCK_N=BLOCK_N_SIZE,
        BLOCK_DMODEL=head_dim,
        num_warps=1,
        num_stages=2,
    )


import torch

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 假设头维度为 64，批次为 2，头数为 4，序列长度为 128
batchs, num_heads, head_dim, seq_len = 2, 4, 64, 128
partition_size = 32

# 随机初始化 Q, K, V
q = torch.randn(batchs, num_heads, head_dim, device="cuda", dtype=torch.float32)
k = torch.randn(
    batchs * seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32
)
v = torch.randn(
    batchs * seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32
)

# 初始化 mid_o 和 mid_o_logexpsum
mid_o = torch.zeros(
    batchs,
    num_heads,
    (seq_len + partition_size - 1) // partition_size,
    head_dim,
    device="cuda",
    dtype=torch.float32,
)
mid_o_logexpsum = torch.zeros(
    batchs,
    num_heads,
    (seq_len + partition_size - 1) // partition_size,
    device="cuda",
    dtype=torch.float32,
)

# 调用修复后的函数
flash_decode_stage1(
    q,
    k,
    v,
    actual_seq_len=seq_len,
    mid_o=mid_o,
    mid_o_logexpsum=mid_o_logexpsum,
    PARTITION_SIZE=partition_size,
)

# 打印输出结果
print("Mid_O:", mid_o)
print("Mid_O_LogExpSum:", mid_o_logexpsum)
