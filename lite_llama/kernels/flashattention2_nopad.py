# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438

import torch,math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd

# TODO: integrating rope with flash-attn
@triton.jit
def flash_attention2_nopad_kernel(
    Q, K, V, O,
    B_Start_Loc, B_Seqlen, 
    sm_scale, heads, num_kv_groups,       # group of kv heads
    stride_q_bs, stride_q_heads, stride_q_dim,  # Q 的 strides
    stride_k_bs, stride_k_heads, stride_k_dim,  # K 的 strides
    stride_v_bs, stride_v_heads, stride_v_dim,  # V 的 strides
    stride_o_bs, stride_o_heads, stride_o_dim,
    BLOCK_DHEAD_SIZE: tl.constexpr, # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr, # BLOCK size of m_size dimension，即 Q 矩阵行数分成了m_size // BLOCK_M_SIZE 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr, # n_size dimension    
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

    block_start_loc = block_m_idx * BLOCK_M_SIZE # 计算当前 block 的起始和结束索引

    offs_n = tl.arange(0, BLOCK_N_SIZE) # head_dim 维度偏移
    offs_d = tl.arange(0, BLOCK_DHEAD_SIZE)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M_SIZE)

    # Compute offsets for the first block on matrix Q K V Output
    q_offs = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_q_bs
        + cur_head_idx * stride_q_heads
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_seq_len, other=0.0)

    k_offs = offs_n[None, :] * stride_k_bs + cur_kv_head_idx * stride_k_heads + offs_d[:, None] * stride_k_dim
    v_offs = offs_n[:, None] * stride_v_bs + cur_kv_head_idx * stride_v_heads + offs_d[None, :] * stride_v_dim
    
    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    # 初始化用于计算 softmax 归一化项的 m 和 d, 意义见 online-softmax, 这里
    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)
        
    block_mask = tl.where(block_start_loc < cur_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M_SIZE, cur_seq_len)

    # 每次循环按 BLOCK_N_SIZE 来处理 K, V 的列（即 key/value 的序列维度）。
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)
        # 计算 qk^t
        k = tl.load(
            k_ptrs + (cur_seq_start_loc + start_n) * stride_k_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc, other = 0.0
        )

        qk = tl.dot(q, k)
        
        # 应用因果遮罩, 下三角矩阵 causal mask 
        casual_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(casual_mask, qk*sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1)) # 求 qk 的最大值
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)  # qk - m_ij[:, None]更新为安全的 qk 分子项
        d_ij = tl.sum(p, 1) # 1d vector

        # -- 更新归一化项 d_new
        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij
        
        # -- update output accumulator --
        acc = acc * alpha[:, None] # acc scaling 

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
    BLOCK_SIZE = 64 # For Ampere Architecture, 3090ti
    output = torch.empty_like(q)
    batchs = b_seq_len.shape[0]
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]

    num_kv_groups = q.shape[1] // k.shape[1] # num_q_heads // num_k_heads
    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE), batchs * n_heads, 1)
    num_warps = 2 if HEAD_DIM <= 64 else 4
    num_stages = 1
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
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_DHEAD_SIZE=HEAD_DIM,
        BLOCK_M_SIZE=BLOCK_SIZE,
        BLOCK_N_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

def torch_context_attention_fwd2(q, k, v, o, b_start_loc, b_seq_len):
    import torch.nn.functional as F
    batch = b_start_loc.shape[0]
    for i in range(batch):
        start_loc = b_start_loc[i]
        seq_len = b_seq_len[i]
        cur_q = q[start_loc : start_loc + seq_len, :, :]
        cur_q = cur_q.clone().to(torch.float32)
        print(cur_q.shape)

        cur_k = k[start_loc : start_loc + seq_len, :, :]
        cur_k = cur_k.clone().to(torch.float32)

        cur_v = v[start_loc : start_loc + seq_len, :, :]
        cur_v = cur_v.clone().to(torch.float32)

        cur_q = cur_q.transpose(0, 1)
        cur_k = cur_k.transpose(0, 1)
        cur_v = cur_v.transpose(0, 1)
        dk = cur_q.shape[-1]

        p = torch.matmul(cur_q, cur_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        q_index = torch.arange(cur_q.shape[1]).unsqueeze(-1).to(p.device)
        k_index = torch.arange(cur_k.shape[1]).unsqueeze(0).to(p.device)
        mask = (q_index >= k_index).int()
        mask = mask.unsqueeze(0).expand(cur_q.shape[0], -1, -1)

        p = p.masked_fill(mask == 0, float("-inf"))
        s = F.softmax(p, dim=-1)
        o[start_loc : start_loc + seq_len, :, :] = torch.matmul(s, cur_v).transpose(0, 1)

def test():
    import torch
    import numpy as np

    Z, H, N_CTX, D_HEAD = 4, 16, 1024, 128
    sm_scale = 1.0 / (D_HEAD ** 0.5) * 1.4426950408889634

    dtype = torch.float16
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.3)
    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.4)
    torch_o = torch.empty((Z * N_CTX , H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    
    max_input_len = N_CTX
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")

    for i in range(Z):
        b_seq_len[i] = N_CTX
        if i != 0:
            b_start_loc[i] = b_start_loc[i - 1] + N_CTX
    torch_context_attention_fwd2(q, k, v, torch_o, b_start_loc, b_seq_len)

    import time

    torch.cuda.synchronize()
    a = time.time()
    
    triton_o = flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, max_input_len)
    print(triton_o)
    # torch.cuda.synchronize()
    b = time.time()
    # print(o.shape, torch_out.shape)
    print((b - a))
    if torch.isnan(triton_o).any(): # 检查 NaNs
        print(f"NaNs detected in context_forward output at layer") 
    print("max ", torch.max(torch.abs(torch_o - triton_o)))
    print("mean ", torch.mean(torch.abs(torch_o - triton_o)))
    assert torch.allclose(torch_o, triton_o, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test()

# def naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_start_loc, b_seq_len):
#     """
#     参考实现：与 `flash_decoding` 在功能上对应，可用于数值正确性对比。
#     - q: [bs, num_heads, head_dim]
#     - k_cache, v_cache: [total_tokens, num_heads, head_dim]
#     - b_start_loc: [bs], 每个 batch 对应在 k_cache/v_cache 的起始位置
#     - b_seq_len: [bs], 每个 batch 当前已用序列长度
#     """
#     device = q.device
#     batch_size, num_heads, head_dim = q.shape
    
#     # 最终输出: [bs, num_heads, head_dim]
#     output = torch.zeros_like(q)

#     for b in range(batch_size):
#         start_loc = b_start_loc[b] # batch b 的 K/V 起始位置
#         seq_len = b_seq_len[b].item()

#         # 取出 [seq_len, num_heads, head_dim]
#         k_this = k_cache[start_loc : start_loc + seq_len]  # shape: [seq_len, num_heads, head_dim]
#         v_this = v_cache[start_loc : start_loc + seq_len]  # shape: [seq_len, num_heads, head_dim]

#         # q[b, ...]: shape [num_heads, head_dim]
#         q_b = q[b]  # [num_heads, head_dim]

#         # 计算注意力
#         # Q K^T => [num_heads, head_dim] x [seq_len, num_heads, head_dim] => 先调换 K 维度 => ...
#         # 这里为了演示，把 num_heads 这维也计算在 for 里了，或可直接 reshape 到 [num_heads, head_dim].
#         # 参考： scores = (q_b * qk_scale) @ k_this.transpose(-1, -2)，此处需注意 batch 维和 heads 维对齐

#         # 先把 q_b 改成 [num_heads, 1, head_dim], k_this => [seq_len, num_heads, head_dim]
#         # => scores shape: [num_heads, seq_len]
#         scores = torch.empty((num_heads, seq_len), device=device)
#         for h in range(num_heads):
#             # q_b[h, :].shape => [head_dim]
#             # k_this[:, h, :].shape => [seq_len, head_dim]
#             qk = torch.matmul(q_b[h] * qk_scale, k_this[:, h].transpose(0, 1))  # [seq_len]
#             scores[h] = qk

#         # softmax => [num_heads, seq_len]
#         attn_probs = torch.softmax(scores, dim=-1)

#         # out => [num_heads, head_dim]
#         out_b = torch.zeros((num_heads, head_dim), device=device)
#         for h in range(num_heads):
#             # v_this[:, h, :] => [seq_len, head_dim]
#             out_b[h] = torch.matmul(attn_probs[h], v_this[:, h, :])  # [head_dim]

#         output[b] = out_b

#     return output

# def test_flash_decoding_correctness():
#     torch.manual_seed(0)
#     device = "cuda"

#     # ========== 测试维度配置 ========== #
#     batch_size = 4
#     num_heads = 8
#     head_dim = 32
#     max_seq_len = 128  # k_cache, v_cache 最大序列长度

#     # ========== 构造输入张量 ========== #
#     # Q shape: [batch_size, num_heads, head_dim], decode阶段 Q 只有 seq=1
#     q = torch.randn((batch_size, num_heads, head_dim), device=device, dtype=torch.float32)

#     # K, V cache shape: [max_seq_len * batch_size, num_heads, head_dim]
#     # 此处简单起见，假设 b_req_tokens_table 依次排布，不做复杂的 paged_layout
#     total_tokens = max_seq_len * batch_size
#     k_cache = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=torch.float32)
#     v_cache = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=torch.float32)

#     # 对于每个 batch，设置它的 k/v 起始位置(b_req_tokens_table) 和 已用长度(b_seq_len)
#     # 这里假设第 b 个 batch 的起始位置为 b*max_seq_len, 实际可随机或者更灵活的分配
#     # b_req_tokens_table = torch.arange(batch_size * max_seq_len, device=device).view(batch_size, max_seq_len)
#     b_start_loc = torch.arange(batch_size, device='cuda', dtype=torch.int32) * max_seq_len
#     # 每个 batch 当前已经用了多少长度(小于等于 max_seq_len)，可随机生成
#     b_seq_len = torch.randint(1, max_seq_len+1, (batch_size,), device=device, dtype=torch.long)

#     # 缩放因子
#     qk_scale = 1.0 / (head_dim ** 0.5) * 1.4426950408889634

#     # ========== Triton Flash Decoding ========== #
#     triton_output = flash_attention2_no_pad(q, k_cache, v_cache, b_start_loc, b_seq_len, max_seq_len, qk_scale)

#     # ========== Naive 参考实现 ========== #
#     naive_output = naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_start_loc, b_seq_len)

#     # ========== 结果比对 ========== #
#     max_abs_diff = (triton_output - naive_output).abs().max().item()
#     print(f"[Unit Test] Max abs diff = {max_abs_diff:.6f}")

#     # 设置一个容忍度，通常闪电注意力与 Naive 在 float32 下的结果不会相差太大
#     assert max_abs_diff < 1e-3, f"Difference too large: {max_abs_diff}"
#     print("[Unit Test] flash_decoding correctness check passed!\n")

# def benchmark_flash_decoding(
#     batch_sizes = [1, 4, 8],
#     head_dims   = [32, 64],
#     seq_lengths = [128, 256, 512],
#     num_heads   = 8,
#     warmup      = 3,
#     rep         = 10
# ):
#     import time
#     import numpy as np
#     device = "cuda"
#     qk_scale = 1.0 / (64 ** 0.5)  # 只示例一个 scale，可根据 dim 不同动态调整

#     results = []  # 用于存储性能结果，后续可视化

#     for bs in batch_sizes:
#         for d in head_dims:
#             for seq_len in seq_lengths:
#                 # total_tokens = bs * seq_len (简化做法)
#                 total_tokens = bs * seq_len
#                 qk_scale = 1.0 / (d ** 0.5) * 1.4426950408889634
#                 # 随机构造数据
#                 q = torch.randn((bs, num_heads, d), device=device, dtype=torch.float32)
#                 k_cache = torch.randn((total_tokens, num_heads, d), device=device, dtype=torch.float32)
#                 v_cache = torch.randn((total_tokens, num_heads, d), device=device, dtype=torch.float32)
#                 # b_req_tokens_table = torch.arange(bs * seq_len, device=device).view(bs, seq_len)
#                 b_start_loc = torch.arange(bs, device='cuda', dtype=torch.int32) * seq_len
#                 b_seq_len = torch.full((bs,), seq_len, device=device, dtype=torch.long)  # 全部用满

#                 # 预热 (warmup)
#                 for _ in range(warmup):
#                     _ = flash_attention2_no_pad(q, k_cache, v_cache, b_start_loc, b_seq_len, seq_len, qk_scale)
#                     _ = naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_start_loc, b_seq_len)

#                 # 统计 Triton 时间
#                 triton_times = []
#                 for _ in range(rep):
#                     torch.cuda.synchronize()
#                     start = time.time()
#                     _ = flash_attention2_no_pad(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, seq_len)
#                     torch.cuda.synchronize()
#                     end = time.time()
#                     triton_times.append(end - start)

#                 # 统计 Naive 时间
#                 naive_times = []
#                 for _ in range(rep):
#                     torch.cuda.synchronize()
#                     start = time.time()
#                     _ = naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len)
#                     torch.cuda.synchronize()
#                     end = time.time()
#                     naive_times.append(end - start)

#                 triton_mean = np.mean(triton_times)
#                 naive_mean = np.mean(naive_times)
#                 speedup = naive_mean / triton_mean if triton_mean > 0 else 1.0

#                 results.append({
#                     "batch_size": bs,
#                     "head_dim": d,
#                     "num_heads": num_heads,
#                     "seq_len": seq_len,
#                     "triton_mean_time": triton_mean,
#                     "naive_mean_time": naive_mean,
#                     "speedup": speedup
#                 })

#                 print(f"bs={bs}, head_dim={d}, seq_len={seq_len} => "
#                       f"Triton: {triton_mean:.6f}s, Naive: {naive_mean:.6f}s, Speedup: {speedup:.2f}")
    
#     return results

# def plot_benchmark_results(results, fix_bs=None, fix_dim=None):
#     """
#     示例：若需要固定 batch_size 和 head_dim, 观察随 seq_len 变化的时间 / speedup。
#     可根据实际需求定制更丰富的可视化。
#     """
#     import matplotlib.pyplot as plt
#     # 过滤出满足 fix_bs 和 fix_dim 的记录
#     filtered = [r for r in results 
#                 if (fix_bs is None or r["batch_size"] == fix_bs) 
#                 and (fix_dim is None or r["head_dim"] == fix_dim)]
    
#     if not filtered:
#         print("No matched results to plot!")
#         return
    
#     # 按照 seq_len 排序
#     filtered.sort(key=lambda x: x["seq_len"])

#     seq_lens = [f["seq_len"] for f in filtered]
#     triton_time = [f["triton_mean_time"] for f in filtered]
#     naive_time = [f["naive_mean_time"] for f in filtered]
#     speedup = [f["speedup"] for f in filtered]

#     fig, ax1 = plt.subplots(figsize=(8, 5))
#     ax2 = ax1.twinx()

#     # Plot time
#     ax1.plot(seq_lens, triton_time, 'o--', label="Triton Time", color='blue')
#     ax1.plot(seq_lens, naive_time, 's--', label="Naive Time", color='red')
#     ax1.set_xlabel("Sequence Length")
#     ax1.set_ylabel("Time (s)")
#     ax1.set_title(f"FlashAttention2 nopad Benchmark (bs={fix_bs}, dim={fix_dim})")

#     # Plot speedup
#     ax2.plot(seq_lens, speedup, 'd-', label="Speedup", color='green')
#     ax2.set_ylabel("Speedup (Naive / Triton)")

#     # legends
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

#     plt.savefig("./flashattentionv2_nopad_benchamrk.png")
    
# if __name__ == "__main__":
#     # 1. 单元测试
#     test_flash_decoding_correctness()

#     # 2. 基准测试
#     results = benchmark_flash_decoding(
#         batch_sizes = [1, 4],   # 可根据实际需要增减
#         head_dims   = [32, 64],
#         seq_lengths = [128, 256],
#         num_heads   = 8,
#         warmup      = 2,
#         rep         = 5
#     )

#     # 3. 可视化
#     # 例：固定 batch_size=4, head_dim=32
#     plot_benchmark_results(results, fix_bs=4, fix_dim=32)