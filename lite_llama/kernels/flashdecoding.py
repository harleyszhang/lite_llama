import triton, torch
import triton.language as tl
from torch.cuda.amp import custom_fwd


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
		num_warps = 1, 
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
	batchs, num_heads, HEAD_DIM = mid_o.shape[0], mid_o.shape[1], mid_o.shape[-1]
	grid = (batchs, num_heads)
	
	_flash_decoding_stage2_kernel[grid](
		mid_o,  	     # [batch, head, seq_block_num, head_dim]
		mid_o_logexpsum, # [batch, head, seq_block_num]
		atten_output,           # attention 输出首地址
		*mid_o.stride(),
		*mid_o_logexpsum.stride(),
		*atten_output.stride(),
		b_seq_len,   # TODO 支持 PagedAttention 和连续批处理
		BLOCK_DMODEL = HEAD_DIM,
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
	PARTITION_SIZE = 128  # 3090ti 显卡以上可设置为 256
	batchs, num_heads, head_dim = q.shape # decode 阶段 q 的 seq_len = 1, 

	# 最大可用分区数量计算
	max_num_partitions = (max_actual_seq_len + PARTITION_SIZE -1) // PARTITION_SIZE

	# mid_o: 存储每个批次、每个头、每个分区的中间输出
	mid_o = torch.empty((batchs, num_heads, max_num_partitions, head_dim), dtype=torch.float32, device=q.device)
	# 存储每个批次、每个头、每个分区的 log(sum(exp(scores)))，用于后续 decode_stage2 的归一化
	mid_o_logexpsum = torch.empty((batchs, num_heads, max_num_partitions), dtype=torch.float32, device=q.device)

	# decode stage 1: attention in partitions
	flash_decode_stage1(q, k_cache, v_cache, qk_scale, 
                        b_req_tokens_table, b_seq_len, max_actual_seq_len, 
                        mid_o, mid_o_logexpsum, PARTITION_SIZE
                    )
	
	# decode stage 2: reduction among partitions
	atten_output = torch.empty_like(q)

	flash_decode_stage2(mid_o, mid_o_logexpsum, atten_output, b_seq_len, PARTITION_SIZE)

	return atten_output

# --------------------------------------
# 标准 Attention Decode 实现（纯 PyTorch版）
# --------------------------------------
def _naive_attention(q, k, v):
    import math
    head_dim = q.shape[-1]
    q = q.transpose(0, 1)  #(nhead, 1, head_dim)
    k = k.transpose(0, 1)  #(nhead, seqlen, head_dim)
    v = v.transpose(0, 1)  #(nhead, seqlen, head_dim)
    scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    scores = torch.nn.functional.softmax(scores.float(), dim=-1).to(q.dtype)
    output = torch.matmul(scores, v).transpose(0, 1).contiguous() #(1, nhead, head_dim)
    return output

def torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len):
    out = torch.empty_like(q)
    Z = q.shape[0]
    for i in range(Z):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        q_i = q[i:i+1]            #(1, nhead, head_dim)
        k_i = k_cache[start:end]  #(seqlen, nhead, head_dim)
        v_i = v_cache[start:end]  #(seqlen, nhead, head_dim)
        o_i = _naive_attention(q_i, k_i, v_i)
        out[i:i+1] = o_i
    return out

# ----------------------------------
# 性能对比及曲线绘制函数封装（含 Warm up）
# ----------------------------------
def plot_performance_comparison(token_sizes, warmup_iterations=10, test_iterations=50):
    """
    对不同 token size 下的 Flash Decoding 与标准 Attention 的性能进行测试，
    并绘制性能对比曲线。
    
    参数:
      token_sizes: list[int]，不同的 kv cache 长度
      warmup_iterations: int, 预热迭代次数
      test_iterations: int, 正式测试迭代次数
    """
    import matplotlib.pyplot as plt
    device = torch.device('cuda')
    batch = 4
    num_heads = 32
    head_dim = 64
    qk_scale = 1.0 / (head_dim ** 0.5)
    q = torch.randn(batch*1, num_heads, head_dim, device=device)

    flash_times = []
    standard_times = []

    for tokens in token_sizes:
        print(f"\n测试 token size: {tokens}")
        k_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
        v_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
        b_req_tokens_table = torch.arange(0, tokens, device=device, dtype=torch.int32).repeat(batch, 1)
        b_start_loc = torch.tensor([0, tokens, 2*tokens, 3*tokens], dtype=torch.int32, device="cuda") # batch = 4
        b_seq_len = torch.full((batch,), tokens, device=device, dtype=torch.int32)
        max_actual_seq_len = tokens

        # Warm up Flash Decoding 内核
        for _ in range(warmup_iterations):
            _ = flash_decoding(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, max_actual_seq_len)
        # 测试 Flash Decoding
        torch.cuda.synchronize()
        flash_start = torch.cuda.Event(enable_timing=True)
        flash_end = torch.cuda.Event(enable_timing=True)
        flash_start.record()
        for _ in range(test_iterations):
            _ = flash_decoding(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, max_actual_seq_len)
        flash_end.record()
        torch.cuda.synchronize()
        flash_avg = flash_start.elapsed_time(flash_end) / test_iterations
        flash_times.append(flash_avg)
        print(f"Flash Decoding 平均时间: {flash_avg:.3f} ms")

        # Warm up 标准 Attention
        for _ in range(warmup_iterations):
            _ = torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)
        # 测试标准 Attention
        torch.cuda.synchronize()
        std_start = torch.cuda.Event(enable_timing=True)
        std_end = torch.cuda.Event(enable_timing=True)
        std_start.record()
        for _ in range(test_iterations):
            _ = torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)
        std_end.record()
        torch.cuda.synchronize()
        std_avg = std_start.elapsed_time(std_end) / test_iterations
        standard_times.append(std_avg)
        print(f"Standard Attention 平均时间: {std_avg:.3f} ms")

    # 绘制性能对比曲线
    plt.figure(figsize=(8, 6))
    plt.plot(token_sizes, flash_times, marker='o', label='Flash Decoding')
    plt.plot(token_sizes, standard_times, marker='o', label='Standard Attention')
    plt.xlabel('Token Size (kv cache length)')
    plt.ylabel('Average Time (ms)')
    plt.title('Performance Comparison: Flash Decoding vs Standard Attention')
    plt.legend()
    plt.grid(True)
    plt.savefig("./flashdecoding_benchamrk.png")

# -------------------------------
# 验证输出和调用性能对比函数
# -------------------------------
def main():
    torch.manual_seed(0)
    device = torch.device('cuda')
    
    # 测试参数
    batch = 4
    num_heads = 32
    head_dim = 64
    max_tokens = 2048 # 每个请求序列的最大 tokens 长度
    qk_scale = 1.0 / (head_dim ** 0.5)

    # 构造测试数据：固定 q，k_cache, v_cache, b_req_tokens_table, b_seq_len
    # 输入张量 q/k/v 的形状为 [batch * seq_len, num_heads, head_dim], 形状是三维的，为了兼容 flash_decoding 内核
    q = torch.randn(batch * 1, num_heads, head_dim, device=device)
    k_cache = torch.randn(batch * max_tokens, num_heads, head_dim, device=device)
    v_cache = torch.randn(batch * max_tokens, num_heads, head_dim, device=device)
    # 构造每个请求的 kv tokens 分配的显存空间对应的显存块索引
    b_req_tokens_table = torch.arange(0, max_tokens*batch, device=device, dtype=torch.int32).view(batch, max_tokens)
    b_seq_len = torch.full((batch,), max_tokens, device=device, dtype=torch.int32)
    b_start_loc = torch.tensor([0, max_tokens, 2*max_tokens, 3*max_tokens], dtype=torch.int32, device="cuda") # batch = 4
    
    # 单次验证 flash_decoding 输出形状及数值（与标准 Attention 接近）
    flash_out = flash_decoding(q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, max_tokens)
    standard_out = torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len)
    print("Flash Decoding output shape:", flash_out.shape)
    print("Standard Attention output shape:", standard_out.shape)
    if torch.allclose(flash_out, standard_out, atol=1e-3, rtol=1e-3):
        print("验证通过: Flash Decoding 输出与标准 Attention 接近。")
    else:
        diff = (flash_out - standard_out).abs().max().item()
        print(f"验证失败：最大误差为 {diff:.4f}")
    
    # 封装的性能对比曲线函数
    token_numbers = [64, 128, 256, 512, 1024, max_tokens]
    plot_performance_comparison(token_numbers, warmup_iterations=10, test_iterations=50)


if __name__ == '__main__':
    main()