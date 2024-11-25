import torch,math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
from typing import List, Optional, Union
import torch.nn.functional as F
import pdb

@triton.jit
def _flash_decoding_stage1_kernel(
    Q, K, V, sm_scale,

    actual_seq_len,  # 实际序列长度
	num_kv_groups, # group of kv heads
    Mid_O, Mid_O_LogExpSum,

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
	kv_head_pid = head_pid // num_kv_groups

	seq_block_pid = tl.program_id(2)

	# 计算当前批次的起始位置
	cur_batch_start_loc = batch_pid * actual_seq_len

	# 计算当前分区的起始和结束索引
	cur_batch_partition_start_index = seq_block_pid * BLOCK_SEQ
	cur_batch_partition_end_index = tl.minimum(actual_seq_len, cur_batch_partition_start_index + BLOCK_SEQ)

	# 计算需要处理的块数
	num_blocks = tl.where(cur_batch_partition_end_index - cur_batch_partition_start_index <= 0, 0, (cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1) // BLOCK_N)

	# 初始化偏移向量
	offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)  # [BLOCK_N]
	offs_d = tl.arange(0, BLOCK_DMODEL)  # [BLOCK_DMODEL]
    
	# 计算 Q 的偏移量
	q_offs = (
		batch_pid * q_bs_stride
		+ head_pid * q_heads_stride
		+ offs_d * q_dim_stride
	)

	# 计算 K 和 V 的偏移量
	k_offs = (
		(cur_batch_start_loc + offs_n[:, None]) * k_bs_stride
		+ kv_head_pid * k_heads_stride
		+ offs_d[None, :] * k_dim_stride
	)

	v_offs = (
		(cur_batch_start_loc + offs_n[:, None]) * v_bs_stride
		+ kv_head_pid * v_heads_stride
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
	
	# if (batch_pid == 1) & (kv_head_pid == 1) & (seq_block_pid == 1):
	# 	tl.device_print(f"cur_batch_partition_start_index", cur_batch_partition_start_index)
	# 	tl.device_print(f"cur_batch_partition_end_index", cur_batch_partition_end_index)
	# 	tl.device_print(f"offs_n", offs_n)
	# 	tl.device_print(f"offs_d", offs_d)
	# 	tl.device_print(f"k_offs", k_offs)
	# 	tl.device_print(f"cur_batch_start_loc", cur_batch_start_loc)

	# 迭代处理每个块
	for start_n in range(0, num_blocks, 1):
		offs_n_new = start_n * BLOCK_N + offs_n  # [BLOCK_N]
		# 生成 K 的掩码
		k_mask = offs_n_new < cur_batch_partition_end_index  # [BLOCK_N]

		# 加载 K 和 V
		k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_DMODEL]
		v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_DMODEL]

		# 计算 qk^T
		# qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
		qk = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_N]
		qk *= sm_scale
		qk = tl.where(k_mask, qk, -1.0e8)  # [BLOCK_N]

		# tl.device_print(f"q: ", q)
		# tl.device_print(f"k: ", k)
		# tl.device_print(f"qk: ", qk)
        
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
		# 更新 K 和 V 的指针
		k_ptrs += BLOCK_N * k_bs_stride
		v_ptrs += BLOCK_N * v_bs_stride

		# if (batch_pid == 1) & (head_pid == 1) & (seq_block_pid == 1):
		# 	tl.device_print(f"offs_n_new: ", offs_n_new)
		# 	tl.device_print(f"Loaded k", k)
		# 	tl.device_print(f"qk - m_ij", qk)
		# 	tl.device_print(f"current_max: ", current_max)
		# 	tl.device_print(f"exp(qk)", p)
		# 	tl.device_print(f"updated d_i", d_i)
		# 	tl.device_print(f"updated acc", acc)
        
	# need_store = tl.where(num_blocks == 0, 0, 1)
	# for _ in range(0, need_store, 1):
	# 	# 计算存储的偏移量
	# 	off_mid_o = (
	# 		batch_pid * mido_batch_stride
	# 		+ head_pid * mido_heads_stride
	# 		+ seq_block_pid * mido_partitions_stride
	# 		+ offs_d * mido_dim_stride
	# 	)

	# 	off_mid_o_les = (
	# 		batch_pid * mido_les_batch_stride
	# 		+ head_pid * mido_les_heads_stride
	# 		+ seq_block_pid * mido_les_partitions_stride
	# 	)
	# 	tl.store(Mid_O + off_mid_o, acc / d_i)
	# 	tl.store(Mid_O_LogExpSum + off_mid_o_les, m_i + tl.log(d_i))
        
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
    q, k, v,         # Q: [batchs, num_heads, head_dim], K, V: [batchs * seq_len, num_heads, head_dim]
	actual_seq_len,  # 实际的序列长度
    mid_o, mid_o_logexpsum, # Mid_O: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE), head_dim], Mid_O_LogExpSum: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE)]
    PARTITION_SIZE,
):
	BLOCK_N_SIZE = 16

	# BLOCK_DMODEL = q.shape[-1]
	assert PARTITION_SIZE % BLOCK_N_SIZE == 0, "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数"

	batchs, num_heads, head_dim = q.shape
	sm_scale = 1.0 / (head_dim ** 0.5)
	
	# grid 配置的并行度比 flashattention1-2 多了 kv cache seq 维度
	grid = (batchs, num_heads, triton.cdiv(actual_seq_len + PARTITION_SIZE - 1, PARTITION_SIZE))
	num_kv_groups = q.shape[1] // k.shape[1] # num_q_heads // num_k_heads

	_flash_decoding_stage1_kernel[grid](
		q, k, v, sm_scale,
		actual_seq_len,  # 使用实际序列长度
		num_kv_groups,   # kv 组数量
		mid_o, mid_o_logexpsum,
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
	Mid_O,  		# [batch, head, seq_block_num, head_dim]
	Mid_O_LogExpSum,  # [batch, head, seq_block_num]
	Ouput,          # attention 输出首地址
	mido_batch_stride, mido_heads_stride, mido_partitions_stride, mido_dim_stride,
	mido_les_batch_stride, mido_les_heads_stride, mido_les_partitions_stride,
	o_bs_stride, o_heads_stride, o_dim_stride,

	actual_seq_len,   # TODO 支持 PagedAttention 和连续批处理

	BLOCK_DMODEL: tl.constexpr,
	BLOCK_SEQ: tl.constexpr, # type: ignore
):
    """Reduction (online softmax)
    """
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)

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
    num_partitions = (actual_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    for block_seq_n in range(0, num_partitions, 1): # TODO 有 bug 需要修复
        part_v = tl.load(part_v_ptrs)
        part_max = tl.load(part_max_ptrs)

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
        part_v_ptrs += mido_partitions_stride
        part_max_ptrs += mido_les_partitions_stride

    # -- 更新 attention 输出累加器 -- #
    offs_out = batch_pid * o_bs_stride + head_pid * o_heads_stride + offs_d * o_dim_stride
    tl.store(Ouput + offs_out, acc / d_i)

@torch.no_grad()
def flash_decode_stage2(
    mid_o, mid_o_logexpsum, # 存储每个批次、每个头、每个分区的中间分数输出及 log(sum(exp(scores)))
	atten_output,           # attention 输出首地址
	actual_seq_len,  	    # kv cache 在 seq_len 维度的最大长度
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
		actual_seq_len,   # TODO 支持 PagedAttention 和连续批处理
		BLOCK_DMODEL = head_dim,
		BLOCK_SEQ = PARTITION_SIZE, # type: ignore	
		num_warps = 4,
		num_stages = 2,
	)

@torch.no_grad()
def flash_decoding(
    q, 			 # q 查询向量，形状为 [bsz, num_head, head_dim]
    k, v, 	     # 键/值向量缓存，形状为 [ntokens, kv_num_head, head_dim]
    actual_seq_len, # 最大序列长度, 即当前查询对应的 kv cache 大小。用于分区计算
):
	# q.view(-1, num_heads, head_dim)
	assert q.shape[-1] == k.shape[-1] == v.shape[-1]
	batchs, num_heads, head_dim = q.shape # decode 阶段 q 的 seq_len = 1, 
	# actual_seq_len = k.shape[0] // batchs
	
	kv_nums, _, _ = k.shape
	# middle results
	PARTITION_SIZE = 32
	# 最大可用分区数量计算
	max_num_partitions = (actual_seq_len + PARTITION_SIZE -1) // PARTITION_SIZE

	# mid_o: 存储每个批次、每个头、每个分区的中间输出
	mid_o = torch.empty((batchs, num_heads, max_num_partitions, head_dim), dtype=torch.float32, device=q.device)
	# 存储每个批次、每个头、每个分区的 log(sum(exp(scores)))，用于后续 decode_stage2 的归一化
	mid_o_logexpsum = torch.empty((batchs, num_heads, max_num_partitions), dtype=torch.float32, device=q.device)

	# decode stage 1: attention in partitions
	flash_decode_stage1(q, k, v, actual_seq_len, mid_o, mid_o_logexpsum, PARTITION_SIZE)
    
	# decode stage 2: reduction among partitions
	atten_output = torch.empty_like(q)

	flash_decode_stage2(mid_o, mid_o_logexpsum, atten_output, actual_seq_len, PARTITION_SIZE)

	return atten_output

def standard_attention(Q, K, V, sm_scale, mask=None):
	"""
	标准的 PyTorch 实现的自注意力机制。

	Args:
		Q (torch.Tensor): 查询张量，形状 (batch_size, num_heads, seq_length, head_dim)
		K (torch.Tensor): 键张量，形状 (batch_size, num_heads, seq_length, head_dim)
		V (torch.Tensor): 值张量，形状 (batch_size, num_heads, seq_length, head_dim)
		sm_scale (float): Softmax 缩放因子
		mask (torch.Tensor, optional): 遮罩张量，形状 (batch_size, num_heads, seq_length, seq_length)

	Returns:
		torch.Tensor: 注意力输出，形状与 Q 相同
	"""
	print(f"K V cache tensor have 0 numbers is ", torch.nonzero(K==0).numel(), torch.nonzero(V==0).numel())
	# 计算 QK^T
	attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale  # (batch_size, num_heads, seq_length, seq_length)

	if mask is not None:
		attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

	# print("attn_scores", attn_scores)
	attn_weights = F.softmax(attn_scores, dim=-1)

	# 计算注意力输出
	out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

	return out

def test_decode_stage():
	# 设置测试参数
	batch_size = 4
	num_heads = 32
	# 使用 padattention，所以 batch 中的每个 seq 长度相同
	kv_cache_seq_length = 512
	generated_seq_length = 16
	head_dim = 64
	dtype = torch.float16  # 改为 float32

	# 生成固定的初始输入张量
	torch.manual_seed(0)

	# torch_q = torch.randn(batch_size, num_heads, initial_seq_length, head_dim, device='cuda', dtype = dtype)
	torch_k_cache = torch.randn(batch_size, num_heads, kv_cache_seq_length, head_dim, device='cuda', dtype = dtype)
	torch_v_cache = torch.randn(batch_size, num_heads, kv_cache_seq_length, head_dim, device='cuda', dtype = dtype)

	# triton_q = torch_q.transpose(1, 2).view(-1, num_heads, head_dim)
	triton_k_cache = torch_k_cache.transpose(1, 2).reshape(-1, num_heads, head_dim)
	triton_v_cache = torch_v_cache.transpose(1, 2).reshape(-1, num_heads, head_dim)
	print(f"triton_k_cache shape is ", triton_k_cache.shape)

	torch_new_token_q = torch.randn(batch_size, num_heads, 1, head_dim, device='cuda', dtype = dtype)
	triton_new_token_q = torch_new_token_q.transpose(1, 2).reshape(-1, num_heads, head_dim)
	print(f"triton_new_token_q shape is ", triton_new_token_q.shape)

	# 初始化线性层，用于生成 Q、K、V
	# 为了测试，这里使用随机的线性层参数
	q_linear = torch.nn.Linear(head_dim, num_heads * head_dim, bias=False).to('cuda', dtype=dtype)
	k_linear = torch.nn.Linear(head_dim, num_heads * head_dim, bias=False).to('cuda', dtype=dtype)
	v_linear = torch.nn.Linear(head_dim, num_heads * head_dim, bias=False).to('cuda', dtype=dtype)
	# expand = 1
	# b_start_loc = torch.tensor([0, 1024, 2048, 3072], dtype=torch.int32, device="cuda") * expand
	# b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda") * expand

	# 模拟生成过程中逐步增加序列长度
	for step in range(1, generated_seq_length + 1):
		# 扩展 Q, K, V 和 Out
		# q_extended = torch.cat([q_initial, new_token_q], dim=2)

		# 计算 Softmax 缩放因子
		sm_scale_extended = 1.0 / math.sqrt(head_dim)

		# 计算 Triton 内核输出
		
		triton_new_token_q = flash_decoding(triton_new_token_q, triton_k_cache, triton_v_cache, actual_seq_len=kv_cache_seq_length)

		# 使用标准 PyTorch 实现计算扩展后的注意力输出
		torch_new_token_q = standard_attention(torch_new_token_q, torch_k_cache, torch_v_cache, sm_scale_extended)

		# 生成新的 token
		triton_k_cache = torch.cat([triton_k_cache, triton_new_token_q], dim=0)
		triton_v_cache = torch.cat([triton_v_cache, triton_new_token_q], dim=0)
		
		torch_k_cache = torch.cat([torch_k_cache, torch_new_token_q], dim=2)
		torch_v_cache = torch.cat([torch_v_cache, torch_new_token_q], dim=2)
		kv_cache_seq_length += 1

		torch_new_token_q_format = torch_new_token_q.transpose(1, 2).contiguous().view(-1, num_heads, head_dim)
		
		with open(f"/gemini/code/lite_llama/test/debug/{step}_torch_decode_out_tensor.txt", "w") as f:
			f.write(str(torch_new_token_q_format))

		with open(f"/gemini/code/lite_llama/test/debug/{step}_triton_decode_out_tensor.txt", "w") as f:
			f.write(str(triton_new_token_q))

		# 比较 Triton 内核输出与标准实现的输出
		if torch.allclose(triton_new_token_q, torch_new_token_q_format, atol=1e-1):
			max_difference = (triton_new_token_q - torch_new_token_q_format).abs().max()
			print(f"Decode Stage Step {step} Difference {max_difference} Test Passed: Triton output matches PyTorch standard implementation.")
		else:
			max_diff = (triton_new_token_q - torch_new_token_q_format).abs().max()
			print(f"Decode Stage Step {step} Test Failed: Maximum difference {max_diff}")
			# 可选择打印更多信息进行调试
			break  # 根据需要是否停止测试

if __name__ == "__main__":
    print("\nRunning Decode Stage Test...")
    test_decode_stage()