import torch, math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
from typing import List, Optional, Union
import torch.nn.functional as F


@triton.jit
def _flash_decoding_stage2_kernel(
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Ouput,  # attention 输出首地址
    mido_batch_stride,
    mido_heads_stride,
    mido_partitions_stride,
    mido_dim_stride,
    mido_les_batch_stride,
    mido_les_heads_stride,
    mido_les_partitions_stride,
    o_bs_stride,
    o_heads_stride,
    o_dim_stride,
    actual_seq_len,  # TODO 支持 PagedAttention 和连续批处理
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,  # type: ignore
):
    """Reduction (online softmax)"""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # 初始化偏移
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_part_v = (
        batch_idx * mido_batch_stride
        + head_idx * mido_heads_stride
        + offs_d * mido_dim_stride
    )

    offs_part_max = batch_idx * mido_les_batch_stride + head_idx * mido_les_heads_stride

    part_v_ptrs = Mid_O + offs_part_v
    part_max_ptrs = Mid_O_LogExpSum + offs_part_max

    # Reduce kv 分块相关变量值. num_partitions 是 kv 分块数量
    d_i = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    num_partitions = (actual_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    for _ in range(0, num_partitions, 1):
        part_v = tl.load(part_v_ptrs)
        part_max = tl.load(part_max_ptrs)

        # -- 更新局部最大值 和 exp 分子项 p-- #
        m_ij = tl.maximum(part_max, m_i)
        p = tl.exp(part_v - m_ij)

        # -- 计算 alpha = exp(m{j-1} - m{j}) 值 -- #
        alpha = tl.exp(m_i - m_ij)

        # -- 更新归一化项和 attention 输出累加器 -- #
        d_i = d_i * alpha + p

        acc *= alpha
        acc += p * part_v

        # 更新 max 值和指针偏移
        m_i = m_ij
        part_v_ptrs += mido_partitions_stride
        part_max_ptrs += mido_les_partitions_stride

    # -- 更新 attention 输出累加器 -- #
    offs_out = (
        batch_idx * o_bs_stride + head_idx * o_heads_stride + offs_d * o_dim_stride
    )
    tl.store(Ouput + offs_out, acc / d_i)


@torch.no_grad()
def flash_decode_stage2(
    mid_o,
    mid_o_logexpsum,  # 存储每个批次、每个头、每个分区的中间分数输出及 log(sum(exp(scores)))
    atten_output,  # attention 输出首地址
    actual_seq_len,  # kv cache 在 seq_len 维度的最大长度
    PARTITION_SIZE,
):
    HEAD_DIM = mid_o.shape[-1]

    batchs, num_heads = mid_o.shape[0], mid_o.shape[1]
    grid = (batchs, num_heads)

    _flash_decoding_stage2_kernel[grid](
        mid_o,  # [batch, head, seq_block_num, head_dim]
        mid_o_logexpsum,  # [batch, head, seq_block_num]
        atten_output,  # attention 输出首地址
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        *atten_output.stride(),
        actual_seq_len,  # TODO 支持 PagedAttention 和连续批处理
        BLOCK_DMODEL=HEAD_DIM,
        BLOCK_SEQ=PARTITION_SIZE,  # type: ignore
        num_warps=4,
        num_stages=2,
    )


import torch


# 定义 PyTorch 对照实现
def pytorch_flash_decode_stage2(mid_o, mid_o_logexpsum, actual_seq_len, partition_size):
    batchs, num_heads, seq_block_num, head_dim = mid_o.shape
    atten_output_pt = torch.zeros(
        batchs, num_heads, head_dim, device="cuda", dtype=torch.float32
    )

    for batch in range(batchs):
        for head in range(num_heads):
            d_i = torch.zeros(head_dim, device="cuda", dtype=torch.float32)
            m_i = torch.full(
                (head_dim,), -float("inf"), device="cuda", dtype=torch.float32
            )  # 初始化为 [head_dim]
            acc = torch.zeros(head_dim, device="cuda", dtype=torch.float32)
            for partition in range(seq_block_num):
                part_v = mid_o[batch, head, partition]  # [head_dim]
                part_max = mid_o_logexpsum[batch, head, partition].item()  # scalar

                # Broadcast part_max to [head_dim] for comparison
                part_max_tensor = torch.full(
                    (head_dim,), part_max, device="cuda", dtype=torch.float32
                )
                m_ij = torch.maximum(part_max_tensor, m_i)  # [head_dim]
                p = torch.exp(part_v - m_ij)  # [head_dim]

                alpha = torch.exp(m_i - m_ij)  # [head_dim]

                d_i = d_i * alpha + p  # [head_dim]
                acc = acc * alpha + p * part_v  # [head_dim]

                m_i = m_ij  # [head_dim]

            # Avoid division by zero by setting zero where d_i is zero
            mask = d_i > 0
            atten_output_pt[batch, head][mask] = acc[mask] / d_i[mask]
            atten_output_pt[batch, head][~mask] = 0.0  # Handle division by zero

    return atten_output_pt


# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 假设头维度为 64，批次为 2，头数为 4，分区数量为 4，实际序列长度为 128，分区大小为 32
batchs, num_heads, seq_block_num, head_dim = (
    2,
    4,
    4,
    64,
)  # head_dim 必须等于 BLOCK_DMODEL_CONST
actual_seq_len = 128
partition_size = 32

# 随机初始化 Mid_O 和 Mid_O_LogExpSum
mid_o = torch.randn(
    batchs, num_heads, seq_block_num, head_dim, device="cuda", dtype=torch.float32
)
mid_o_logexpsum = torch.randn(
    batchs, num_heads, seq_block_num, device="cuda", dtype=torch.float32
)

# 初始化 atten_output
atten_output = torch.zeros(
    batchs, num_heads, head_dim, device="cuda", dtype=torch.float32
)

# 调用修复后的 Triton 函数
flash_decode_stage2(
    mid_o,
    mid_o_logexpsum,
    atten_output,
    actual_seq_len=actual_seq_len,
    PARTITION_SIZE=partition_size,
)

# 调用 PyTorch 实现
pt_atten_output = pytorch_flash_decode_stage2(
    mid_o, mid_o_logexpsum, actual_seq_len, partition_size
)

# 比较 Triton 和 PyTorch 的输出
diff_atten_output = torch.abs(atten_output - pt_atten_output).max()
print(f"Difference in Atten_Output: {diff_atten_output.item()}")

# 断言差异在合理范围内
assert diff_atten_output < 1e-3, "Atten_Output 的差异超出容忍范围"
print("Triton 内核与 PyTorch 实现的数值对比通过。")
