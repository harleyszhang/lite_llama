# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438
# https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

import torch, math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd


configs_tma = [
    triton.Config(
        {"BLOCK_M_SIZE": BM, "BLOCK_N_SIZE": BN}, num_stages=stages, num_warps=warps
    )
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for warps in [4, 8, 16]
    for stages in [2, 3, 4, 6]
]


def keep_tma(conf):
    BLOCK_M_SIZE = conf.kwargs["BLOCK_M_SIZE"]
    BLOCK_N_SIZE = conf.kwargs["BLOCK_N_SIZE"]
    if (
        torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M_SIZE * BLOCK_N_SIZE < 128 * 128
        and conf.num_warps == 8
    ):
        return False
    return True


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
@custom_fwd(cast_inputs=torch.float16)
def flash_attention2_no_pad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale,
    b_start_loc,
    b_seq_len,
    max_seq_len,
):
    """Compute Flash-attention, can't support fp32 input
    参数:
        q: Query tensor, shape: [bs*m_size, n_heads, head_dim], decode 阶段, q 的 seq_len 和 k v 不一致, 其值为 1
        k: Key tensor,  shape: [bs*n_size, n_heads, head_dim].
        v: Value tensor, shape is consistent with k.
    """
    output = torch.empty_like(q)
    batchs = b_seq_len.shape[0]
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]

    BLOCK_SIZE = 64  # For Ampere Architecture, 3090ti, set 128
    num_warps = 4 if HEAD_DIM <= 64 else 8
    num_stages = 1

    num_kv_groups = q.shape[1] // k.shape[1]  # num_q_heads // num_k_heads
    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE), batchs * n_heads, 1)

    flash_attention2_nopad_kernel[grid](
        q,
        k,
        v,
        output,
        b_start_loc,
        b_seq_len,
        sm_scale,
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
        BLOCK_M_SIZE=BLOCK_SIZE,  # 使用或者关闭 autotune 针对不同机器和上下文长度自动优化内核配置
        BLOCK_N_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


# --------------------------------------
# 标准 Attention Prefill 实现（纯 PyTorch版）
# --------------------------------------
def _naive_attention(q, k, v):
    import math

    bs, seqlen, num_head, head_dim = q.shape
    device = q.device
    mask = 1.0 - torch.tril(
        torch.ones((seqlen, seqlen), device=device), diagonal=0
    ).unsqueeze(0).unsqueeze(0)
    mask.masked_fill_(mask.to(torch.bool), -100000000.0)
    q = q.transpose(1, 2)  # (bs, num_head, seqlen, head_dim)
    k = k.transpose(1, 2)  # (bs, num_head, seqlen, head_dim)
    v = v.transpose(1, 2)  # (bs, num_head, seqlen, head_dim)
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    scores = torch.nn.functional.softmax(scores.float() + mask, dim=-1).to(q.dtype)
    output = (
        torch.matmul(scores, v)
        .transpose(1, 2)
        .contiguous()
        .reshape(bs, seqlen, num_head, head_dim)
    )
    return output


def _sdpa(q, k, v):
    bs, seqlen, num_head, head_dim = q.shape
    q = q.transpose(1, 2)  # (bs, num_head, seqlen, head_dim)
    k = k.transpose(1, 2)  # (bs, num_head, seqlen, head_dim)
    v = v.transpose(1, 2)  # (bs, num_head, seqlen, head_dim)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, num_head, head_dim)
    return output


def standard_attention_prefill(q, k, v, b_start_loc, b_seq_len, sdpa=True):
    out = torch.empty_like(q)
    Z = b_start_loc.shape[0]
    for i in range(Z):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        qi = q[start:end].unsqueeze(0)
        ki = k[start:end].unsqueeze(0)
        vi = v[start:end].unsqueeze(0)
        if sdpa:
            oi = _sdpa(qi, ki, vi)
        else:
            oi = _naive_attention(qi, ki, vi)
        out[start:end] = oi.squeeze(0)
    return out


# =============================================================================
# 内核精度验证与性能比较函数封装
# =============================================================================
def run_flash_attention2_no_pad_benchmark(
    batch=4, n_heads=32, head_dim=128, max_seq_len_list=[1024, 2048, 4096]
):
    """
    构造输入 q/k/v 张量形状为 [batch, n_heads, head_dim] (q) 
    和 [max_seq_len, n_heads, head_dim] (k, v),
    验证 flash_attention2_no_pad 输出结果与标准 attention 对齐（允许一定误差），
    并比较两者在 decode 阶段的性能，输出性能对比曲线。
    返回一个字典，包含验证误差及各 max_seq_len 下的平均执行时间。
    """
    import matplotlib.pyplot as plt

    # =============================================================================
    # 1, 内核精度验证
    # =============================================================================
    device = "cuda"
    sm_scale = 1.0 / math.sqrt(head_dim) * 1.4426950408889634
    max_seq_len = max_seq_len_list[0]

    # q 的 shape: [batch, n_heads, head_dim] (decode 阶段 q 的 seq_len=1)
    shape = (batch * max_seq_len, n_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)
    # 构造 b_start_loc 和 b_seq_len (假设每个 batch 从 0 开始，序列长度均为 max_seq_len)
    b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 512, 1536, 2048], dtype=torch.int32, device="cuda")

    triton_output = flash_attention2_no_pad(
        q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len
    )
    torch_output = standard_attention_prefill(
        q, k, v, b_start_loc, b_seq_len, sdpa=False
    )
    print(
        f"The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}"
    )

    # =============================================================================
    # 2, 内核运行速度性能比较
    # =============================================================================
    flash_times = []
    standard_times = []
    iterations = 50

    for seq_len in max_seq_len_list:
        # q 的 shape: [batch, n_heads, head_dim] (decode 阶段 q 的 seq_len=1)
        shape = (batch * seq_len, n_heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        # 构造 b_start_loc 和 b_seq_len (假设每个 batch 从 0 开始，序列长度均为 max_seq_len)
        b_start_loc = torch.tensor(
            [0, seq_len, 2 * seq_len, 3 * seq_len], dtype=torch.int32, device="cuda"
        )  # batch = 4
        b_seq_len = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
        # b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda")
        # b_start_loc = torch.tensor([0, 512, 1536, 2048], dtype=torch.int32, device="cuda")

        # 预热
        _ = flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, seq_len)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            _ = flash_attention2_no_pad(
                q, k, v, sm_scale, b_start_loc, b_seq_len, seq_len
            )
        end_event.record()
        torch.cuda.synchronize()
        flash_time = start_event.elapsed_time(end_event) / iterations
        flash_times.append(flash_time)

        # 标准 attention 预热
        _ = standard_attention_prefill(q, k, v, b_start_loc, b_seq_len)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(iterations):
            _ = standard_attention_prefill(q, k, v, b_start_loc, b_seq_len)
        end_event.record()
        torch.cuda.synchronize()
        standard_time = start_event.elapsed_time(end_event) / iterations
        standard_times.append(standard_time)

        print(
            f"max_seq_len = {seq_len:4d}: flash_attn = {flash_time:.3f} ms, standard_attn = {standard_time:.3f} ms"
        )

    # 绘制性能对比曲线
    plt.figure(figsize=(8, 5))
    plt.plot(max_seq_len_list, flash_times, marker="o", label="Flash Attentionv2")
    plt.plot(max_seq_len_list, standard_times, marker="s", label="Standard Attention")
    plt.xlabel("max_seq_len (kv cache length)")
    plt.ylabel("Average execution time (ms)")
    plt.title("Prefill Stage Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("./flashattentionv2_nopad_benchamrk.png")

    return {
        "max_seq_len_list": max_seq_len_list,
        "flash_times": flash_times,
        "standard_times": standard_times,
    }


# =============================================================================
# 如果直接运行该脚本，则执行验证与性能比较
# =============================================================================
if __name__ == "__main__":
    stats = run_flash_attention2_no_pad_benchmark()
    print("Benchmark statistics:", stats)
