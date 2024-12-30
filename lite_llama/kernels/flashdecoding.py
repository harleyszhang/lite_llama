import triton, torch
import triton.language as tl
from torch.cuda.amp import custom_fwd

@triton.jit
def detect_nan_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    is_nan = x!=x # NaN != NaN
    tl.store(output_ptr + offsets, is_nan, mask=mask)

def detect_nan(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    output = torch.zeros_like(input_tensor, dtype=torch.int32)
    detect_nan_kernel[grid](input_tensor, output, N, BLOCK_SIZE)
    return output

@triton.jit
def _flash_decoding_stage1_kernel(
    Q, K, V, qk_scale,
    b_req_tokens_table, B_Seqlen, 
	num_kv_groups, # group of kv heads
    Mid_O, Mid_O_LogExpSum,
	stride_req_to_tokens_b, stride_req_to_tokens_s,
    q_bs_stride, q_heads_stride, q_dim_stride,  # Q 的 strides
    k_bs_stride, k_heads_stride, k_dim_stride,  # K 的 strides
    v_bs_stride, v_heads_stride, v_dim_stride,  # V 的 strides
    mido_batch_stride, mido_heads_stride, mido_partitions_stride, mido_dim_stride,
    mido_les_batch_stride, mido_les_heads_stride, mido_les_partitions_stride,
    BLOCK_SEQ: tl.constexpr, # 默认 128
    BLOCK_N: tl.constexpr,   # 默认 32
    BLOCK_DMODEL: tl.constexpr,
):
    """Flash Attention Stage1 Triton Kernel"""
    # 获取当前程序的 block 在各个维度上的索引
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_block_pid = tl.program_id(2)
    kv_head_pid = head_pid // num_kv_groups

    # 计算当前批次的起始位置
    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)
    cur_req_start_loc = tl.load(b_req_tokens_table + stride_req_to_tokens_b * batch_pid)

    # 计算当前分区的起始和结束索引
    cur_batch_partition_start_index = seq_block_pid * BLOCK_SEQ
    cur_batch_partition_end_index = tl.minimum(cur_batch_seq_len, cur_batch_partition_start_index + BLOCK_SEQ)

    # 计算需要处理的块数
    num_blocks = tl.where(cur_batch_partition_end_index - cur_batch_partition_start_index <= 0, 
                        0, (cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1) // BLOCK_N)

    # 初始化偏移向量
    offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_DMODEL)  # [BLOCK_DMODEL]

    # 计算 Q K 的偏移量
    q_offs = (
        batch_pid * q_bs_stride 
        + head_pid * q_heads_stride
        + offs_d * q_dim_stride
    )
    k_offs = kv_head_pid * k_heads_stride + offs_d[None, :] * k_dim_stride 

    q_ptrs = Q + q_offs # 获取 Q 指针
    q = tl.load(q_ptrs)  # # 加载 Q 向量 [BLOCK_DMODEL]

    # 初始化归一化项和累加器
    d_i = 0.0  # 标量 # 使用小的正数而不是0
    m_i = -float("inf")  # 标量
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)  # [BLOCK_DMODEL]

    # 迭代处理每个块
    for start_n in range(0, num_blocks, 1):
        # k 位置索引计算
        offs_n_new = offs_n + start_n * BLOCK_N  # [BLOCK_N]
        k_loc = tl.load(b_req_tokens_table + stride_req_to_tokens_b * batch_pid + offs_n_new, mask=offs_n_new < cur_batch_partition_end_index, other=0.0)
        k_ptrs = k_loc[:, None] * k_bs_stride + k_offs
        
        k_mask = offs_n_new < cur_batch_partition_end_index  # [BLOCK_N]
        
        k = tl.load(K + k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(V + k_ptrs, mask=k_mask[:, None], other=0.0)
        
        # 计算 qk^T
        qk = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_N]
        qk *= qk_scale
        qk = tl.where(k_mask, qk, float("-inf"))  # [BLOCK_N]

        # 更新最大值项和 qk 项
        current_max = tl.max(qk)  # 标量
        m_ij = tl.maximum(m_i, current_max)  # 标量
        p = tl.exp(qk - m_ij)  # [BLOCK_N]
        
        # 更新归一化项
        alpha = tl.exp(m_i - m_ij) 
        d_i = alpha * d_i + tl.sum(p, axis=0)

        # 更新 attention 输出累加器
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)  # [BLOCK_DMODEL]
        # acc = acc * alpha + tl.dot(p, v)  # [BLOCK_DMODEL]
        
        # 更新归一化器
        m_i = m_ij
        
    # 计算是否需要存储
    need_store = num_blocks > 0  # 标量布尔值

    # 计算存储的偏移量
    off_mid_o = (
        batch_pid * mido_batch_stride
        + head_pid * mido_heads_stride
        + seq_block_pid * mido_partitions_stride
        + offs_d * mido_dim_stride
    )

    off_mid_o_les = (
        batch_pid * mido_les_batch_stride
        + head_pid * mido_les_heads_stride
        + seq_block_pid * mido_les_partitions_stride
    )

    # 计算最终的 attention 输出和 log-sum-exp
    need_store = tl.where(num_blocks == 0, 0, 1)
    for _ in range(0, need_store, 1):
        tl.store(Mid_O + off_mid_o, acc / d_i)
        tl.store(Mid_O_LogExpSum + off_mid_o_les, m_i + tl.log(d_i))

@torch.no_grad()
def flash_decode_stage1(
    q, k, v,         		# Q: [batchs, num_heads, head_dim], K, V: [batchs * seq_len, num_heads, head_dim]
    qk_scale, 
    b_req_tokens_table,
	b_seq_len, 
	max_actual_seq_len,     # 最大的实际序列长度
    mid_o, mid_o_logexpsum, # Mid_O: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE), head_dim], Mid_O_LogExpSum: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE)]
    PARTITION_SIZE,
):
	BLOCK_N_SIZE = 16

	# BLOCK_DMODEL = q.shape[-1]
	assert PARTITION_SIZE % BLOCK_N_SIZE == 0, "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数"

	batchs, num_heads, head_dim = q.shape # decode 阶段 q 张量的 seq_len = 1, 这里的 batchs 实际就是 batch_size
	
	# grid 配置的并行度比 flashattention1-2 多了 kv cache seq 维度
	grid = (batchs, num_heads, triton.cdiv(max_actual_seq_len + PARTITION_SIZE - 1, PARTITION_SIZE))
	num_kv_groups = q.shape[1] // k.shape[1] # num_q_heads // num_k_heads

	_flash_decoding_stage1_kernel[grid](
		q, k, v, qk_scale,
	   	b_req_tokens_table,
        b_seq_len, 
		num_kv_groups,   # kv 组数量
		mid_o, mid_o_logexpsum,
		*b_req_tokens_table.stride(),
		*q.stride(),
		*k.stride(),
		*v.stride(),
		*mid_o.stride(),
		*mid_o_logexpsum.stride(),
		BLOCK_SEQ = PARTITION_SIZE,
		BLOCK_N = BLOCK_N_SIZE,
		BLOCK_DMODEL = head_dim,
		num_warps = 2,
		num_stages = 2,
	)

@triton.jit
def _flash_decoding_stage2_kernel(
	Mid_O,  		  # [batch, head, seq_block_num, head_dim]
	Mid_O_LogExpSum,  # [batch, head, seq_block_num]
	Ouput,            # attention 输出首地址
	mido_batch_stride, mido_heads_stride, mido_partitions_stride, mido_dim_stride,
	mido_les_batch_stride, mido_les_heads_stride, mido_les_partitions_stride,
	o_bs_stride, o_heads_stride, o_dim_stride,
	B_Seqlen,   # TODO 支持 PagedAttention 和连续批处理
	BLOCK_DMODEL: tl.constexpr,
	BLOCK_SEQ: tl.constexpr, # type: ignore
):
    """Reduction (online softmax)
    """
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)
    
    # 初始化偏移 
    offs_d = tl.arange(0, BLOCK_DMODEL)

	# 最后一个维度 stride 为 1 可省略, 如 mido_dim_stride
    offs_part_v = batch_pid * mido_batch_stride \
                + head_pid * mido_heads_stride \
                + offs_d

    offs_part_max = batch_pid * mido_les_batch_stride \
                + head_pid * mido_les_heads_stride

    part_v_ptrs = Mid_O + offs_part_v
    part_max_ptrs = Mid_O_LogExpSum + offs_part_max

    # Reduce kv 分块相关变量值. num_partitions 是 kv 分块数量
    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    num_partitions = (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    for block_seq_n in range(0, num_partitions, 1): # TODO 有 bug 需要修复
        part_v = tl.load(part_v_ptrs + block_seq_n * mido_partitions_stride)
        part_max = tl.load(part_max_ptrs + block_seq_n) # mido_les_partitions_stride = 1

        # -- 更新局部最大值 -- #
        m_ij = tl.maximum(part_max, m_i)
        # -- 计算 alpha = exp(m{j-1} - m{j}) 值 -- #
        alpha = tl.exp(m_i - m_ij)

        # -- 更新归一化项和 attention 输出累加器 -- #
        p = tl.exp(part_max - m_ij)
        acc = alpha * acc + p * part_v

        # alpha * d_i: 缩放 d_i, p * weight: 当前元素的指数值 * 权重
        d_i = alpha * d_i + p

        # 更新 max 值和指针偏移
        m_i = m_ij

    # -- 更新 attention 输出累加器 -- #
    offs_out = batch_pid * o_bs_stride + head_pid * o_heads_stride + offs_d * o_dim_stride
    tl.store(Ouput + offs_out, acc / d_i)

@torch.no_grad()
def flash_decode_stage2(
    mid_o, mid_o_logexpsum, # 存储每个批次、每个头、每个分区的中间分数输出及 log(sum(exp(scores)))
	atten_output,           # attention 输出首地址
	b_seq_len,  	        # kv cache 在 seq_len 维度的长度向量
    PARTITION_SIZE
):	
	batchs, num_heads, head_dim = mid_o.shape[0], mid_o.shape[1], mid_o.shape[-1]
	grid = (batchs, num_heads)
	
	_flash_decoding_stage2_kernel[grid](
		mid_o,  	     # [batch, head, seq_block_num, head_dim]
		mid_o_logexpsum, # [batch, head, seq_block_num]
		atten_output,           # attention 输出首地址
		*mid_o.stride(),
		*mid_o_logexpsum.stride(),
		*atten_output.stride(),
		b_seq_len,   # TODO 支持 PagedAttention 和连续批处理
		BLOCK_DMODEL = head_dim,
		BLOCK_SEQ = PARTITION_SIZE, # type: ignore	
		num_warps = 4,
		num_stages = 2,
	)

@torch.no_grad()
def flash_decoding(
    q, 			 # q 查询向量，形状为 [bsz, num_head, head_dim]
    k_cache, v_cache, 	     # 键/值向量缓存，形状为 [max_tokens, kv_num_head, head_dim]
    qk_scale,
    b_req_tokens_table, b_seq_len, # start locations and sequence lengths for kv cache in a batch
    max_actual_seq_len
):
	# q.view(-1, num_heads, head_dim)
	assert q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1]
	PARTITION_SIZE = 128
	batchs, num_heads, head_dim = q.shape # decode 阶段 q 的 seq_len = 1, 

	# 最大可用分区数量计算
	max_num_partitions = (max_actual_seq_len + PARTITION_SIZE -1) // PARTITION_SIZE

	# mid_o: 存储每个批次、每个头、每个分区的中间输出
	mid_o = torch.empty((batchs, num_heads, max_num_partitions, head_dim), dtype=torch.float32, device=q.device)
	# 存储每个批次、每个头、每个分区的 log(sum(exp(scores)))，用于后续 decode_stage2 的归一化
	mid_o_logexpsum = torch.empty((batchs, num_heads, max_num_partitions), dtype=torch.float32, device=q.device)

	# decode stage 1: attention in partitions
	flash_decode_stage1(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, max_actual_seq_len, mid_o, mid_o_logexpsum, PARTITION_SIZE)
	
	# decode stage 2: reduction among partitions
	atten_output = torch.empty_like(q)

	flash_decode_stage2(mid_o, mid_o_logexpsum, atten_output, b_seq_len, PARTITION_SIZE)

	return atten_output


def naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len):
    """
    参考实现：与 `flash_decoding` 在功能上对应，可用于数值正确性对比。
    - q: [bs, num_heads, head_dim]
    - k_cache, v_cache: [total_tokens, num_heads, head_dim]
    - b_req_tokens_table: [bs], 每个 batch 对应在 k_cache/v_cache 的起始位置
    - b_seq_len: [bs], 每个 batch 当前已用序列长度
    """
    device = q.device
    batch_size, num_heads, head_dim = q.shape
    
    # 最终输出: [bs, num_heads, head_dim]
    output = torch.zeros_like(q)

    for b in range(batch_size):
        start_loc = b_req_tokens_table[b][0] # batch b 的 K/V 起始位置
        seq_len = b_seq_len[b].item()

        # 取出 [seq_len, num_heads, head_dim]
        k_this = k_cache[start_loc : start_loc + seq_len]  # shape: [seq_len, num_heads, head_dim]
        v_this = v_cache[start_loc : start_loc + seq_len]  # shape: [seq_len, num_heads, head_dim]

        # q[b, ...]: shape [num_heads, head_dim]
        q_b = q[b]  # [num_heads, head_dim]

        # 计算注意力
        # Q K^T => [num_heads, head_dim] x [seq_len, num_heads, head_dim] => 先调换 K 维度 => ...
        # 这里为了演示，把 num_heads 这维也计算在 for 里了，或可直接 reshape 到 [num_heads, head_dim].
        # 参考： scores = (q_b * qk_scale) @ k_this.transpose(-1, -2)，此处需注意 batch 维和 heads 维对齐

        # 先把 q_b 改成 [num_heads, 1, head_dim], k_this => [seq_len, num_heads, head_dim]
        # => scores shape: [num_heads, seq_len]
        scores = torch.empty((num_heads, seq_len), device=device)
        for h in range(num_heads):
            # q_b[h, :].shape => [head_dim]
            # k_this[:, h, :].shape => [seq_len, head_dim]
            qk = torch.matmul(q_b[h] * qk_scale, k_this[:, h].transpose(0, 1))  # [seq_len]
            scores[h] = qk

        # softmax => [num_heads, seq_len]
        attn_probs = torch.softmax(scores, dim=-1)

        # out => [num_heads, head_dim]
        out_b = torch.zeros((num_heads, head_dim), device=device)
        for h in range(num_heads):
            # v_this[:, h, :] => [seq_len, head_dim]
            out_b[h] = torch.matmul(attn_probs[h], v_this[:, h, :])  # [head_dim]

        output[b] = out_b

    return output

def test_flash_decoding_correctness():
    torch.manual_seed(0)
    device = "cuda"

    # ========== 测试维度配置 ========== #
    batch_size = 4
    num_heads = 8
    head_dim = 32
    max_seq_len = 128  # k_cache, v_cache 最大序列长度

    # ========== 构造输入张量 ========== #
    # Q shape: [batch_size, num_heads, head_dim], decode阶段 Q 只有 seq=1
    q = torch.randn((batch_size, num_heads, head_dim), device=device, dtype=torch.float32)

    # K, V cache shape: [max_seq_len * batch_size, num_heads, head_dim]
    # 此处简单起见，假设 b_req_tokens_table 依次排布，不做复杂的 paged_layout
    total_tokens = max_seq_len * batch_size
    k_cache = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=torch.float32)
    v_cache = torch.randn((total_tokens, num_heads, head_dim), device=device, dtype=torch.float32)

    # 对于每个 batch，设置它的 k/v 起始位置(b_req_tokens_table) 和 已用长度(b_seq_len)
    # 这里假设第 b 个 batch 的起始位置为 b*max_seq_len, 实际可随机或者更灵活的分配
    b_req_tokens_table = torch.arange(batch_size * max_seq_len, device=device).view(batch_size, max_seq_len)
    # 每个 batch 当前已经用了多少长度(小于等于 max_seq_len)，可随机生成
    b_seq_len = torch.randint(1, max_seq_len+1, (batch_size,), device=device, dtype=torch.long)

    # 缩放因子
    qk_scale = 1.0 / (head_dim ** 0.5)

    # ========== Triton Flash Decoding ========== #
    triton_output = flash_decoding(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, max_seq_len)

    # ========== Naive 参考实现 ========== #
    naive_output = naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len)

    # ========== 结果比对 ========== #
    max_abs_diff = (triton_output - naive_output).abs().max().item()
    print(f"[Unit Test] Max abs diff = {max_abs_diff:.6f}")

    # 设置一个容忍度，通常闪电注意力与 Naive 在 float32 下的结果不会相差太大
    assert max_abs_diff < 1e-3, f"Difference too large: {max_abs_diff}"
    print("[Unit Test] flash_decoding correctness check passed!\n")

def benchmark_flash_decoding(
    batch_sizes = [1, 4, 8],
    head_dims   = [32, 64],
    seq_lengths = [128, 256, 512],
    num_heads   = 8,
    warmup      = 3,
    rep         = 10
):
    import time
    import numpy as np
    device = "cuda"
    qk_scale = 1.0 / (64 ** 0.5)  # 只示例一个 scale，可根据 dim 不同动态调整

    results = []  # 用于存储性能结果，后续可视化

    for bs in batch_sizes:
        for d in head_dims:
            for seq_len in seq_lengths:
                # total_tokens = bs * seq_len (简化做法)
                total_tokens = bs * seq_len

                # 随机构造数据
                q = torch.randn((bs, num_heads, d), device=device, dtype=torch.float32)
                k_cache = torch.randn((total_tokens, num_heads, d), device=device, dtype=torch.float32)
                v_cache = torch.randn((total_tokens, num_heads, d), device=device, dtype=torch.float32)
                b_req_tokens_table = torch.arange(bs * seq_len, device=device).view(bs, seq_len)
        
                b_seq_len = torch.full((bs,), seq_len, device=device, dtype=torch.long)  # 全部用满

                # 预热 (warmup)
                for _ in range(warmup):
                    _ = flash_decoding(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, seq_len)
                    _ = naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len)

                # 统计 Triton 时间
                triton_times = []
                for _ in range(rep):
                    torch.cuda.synchronize()
                    start = time.time()
                    _ = flash_decoding(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, seq_len)
                    torch.cuda.synchronize()
                    end = time.time()
                    triton_times.append(end - start)

                # 统计 Naive 时间
                naive_times = []
                for _ in range(rep):
                    torch.cuda.synchronize()
                    start = time.time()
                    _ = naive_flash_decoding_reference(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len)
                    torch.cuda.synchronize()
                    end = time.time()
                    naive_times.append(end - start)

                triton_mean = np.mean(triton_times)
                naive_mean = np.mean(naive_times)
                speedup = naive_mean / triton_mean if triton_mean > 0 else 1.0

                results.append({
                    "batch_size": bs,
                    "head_dim": d,
                    "num_heads": num_heads,
                    "seq_len": seq_len,
                    "triton_mean_time": triton_mean,
                    "naive_mean_time": naive_mean,
                    "speedup": speedup
                })

                print(f"bs={bs}, head_dim={d}, seq_len={seq_len} => "
                      f"Triton: {triton_mean:.6f}s, Naive: {naive_mean:.6f}s, Speedup: {speedup:.2f}")
    
    return results

def plot_benchmark_results(results, fix_bs=None, fix_dim=None):
    """
    示例：若需要固定 batch_size 和 head_dim，观察随 seq_len 变化的时间 / speedup。
    可根据实际需求定制更丰富的可视化。
    """
    import matplotlib.pyplot as plt
    # 过滤出满足 fix_bs 和 fix_dim 的记录
    filtered = [r for r in results 
                if (fix_bs is None or r["batch_size"] == fix_bs) 
                and (fix_dim is None or r["head_dim"] == fix_dim)]
    
    if not filtered:
        print("No matched results to plot!")
        return
    
    # 按照 seq_len 排序
    filtered.sort(key=lambda x: x["seq_len"])

    seq_lens = [f["seq_len"] for f in filtered]
    triton_time = [f["triton_mean_time"] for f in filtered]
    naive_time = [f["naive_mean_time"] for f in filtered]
    speedup = [f["speedup"] for f in filtered]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Plot time
    ax1.plot(seq_lens, triton_time, 'o--', label="Triton Time", color='blue')
    ax1.plot(seq_lens, naive_time, 's--', label="Naive Time", color='red')
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Time (s)")
    ax1.set_title(f"Flash Decoding Benchmark (bs={fix_bs}, dim={fix_dim})")

    # Plot speedup
    ax2.plot(seq_lens, speedup, 'd-', label="Speedup", color='green')
    ax2.set_ylabel("Speedup (Naive / Triton)")

    # legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.savefig("./flashdecoding_benchamrk.png")
    
if __name__ == "__main__":
    # 1. 单元测试
    test_flash_decoding_correctness()

    # 2. 基准测试
    results = benchmark_flash_decoding(
        batch_sizes = [1, 4],   # 可根据实际需要增减
        head_dims   = [32, 64],
        seq_lengths = [128, 256],
        num_heads   = 8,
        warmup      = 2,
        rep         = 5
    )

    # 3. 可视化
    # 例：固定 batch_size=4, head_dim=32
    plot_benchmark_results(results, fix_bs=4, fix_dim=32)