# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438

import torch,math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
from typing import List, Optional, Union
import torch.nn.functional as F

@triton.jit
def _attn_fwd_inner(
	acc, m_i, d_i, q,
	k_ptrs, v_ptrs, 
	k_seq_stride, v_seq_stride,
	offs_m,
	qk_scale, 
	n_size, # kv seq_len
	causal_mask, 
	BLOCK_M_SIZE: tl.constexpr, BLOCK_N_SIZE: tl.constexpr, fp8_v: tl.constexpr
):
	n_range_offs = tl.arange(0, BLOCK_N_SIZE) # head_dim 维度偏移

	# 在 SRAM 上完成计算
	for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
		block_n_start_idx = tl.multiple_of(block_n_start_idx, BLOCK_N_SIZE)
		block_n_offs = block_n_start_idx + n_range_offs
        
		k_mask = block_n_offs[:, None] < n_size
		k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask, other=0.0)

		# qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
		qk = tl.dot(q, tl.trans(k))

		# 应用因果遮罩
		if causal_mask:
			offs_k = block_n_offs
			# casual 模型的 causal mask 下三角矩阵
			mask = offs_m[:, None] >= offs_k[None, :]
			qk = qk * qk_scale + tl.where(mask, 0, -1.0e8)
			# qk = tl.where(mask, qk * qk_scale, -1.0e8)
			m_ij = tl.maximum(m_i, tl.max(qk, 1)) # 求 qk 的最大值
			qk -= m_ij[:, None] # 更新为安全的 qk
		else:
			m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
			qk = qk * qk_scale - m_ij[:, None]

		p = tl.math.exp2(qk)
		d_ij = tl.sum(p, 1) # 1d vector

		# -- 更新归一化项 d_new
		alpha = tl.math.exp2(m_i - m_ij)
		d_i = d_i * alpha + d_ij

		# -- 更新 attention 输出累加器 --
		acc = acc * alpha[:, None]

		# compute O = PV
		v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask, other=0.0)
		p = p.to(v.dtype)
		# acc += tl.dot(p, v)
		acc = tl.dot(p, v, acc)
		# update the normalizer (l and d) for next iteration
		m_i = m_ij

	return acc, d_i

@triton.jit
def flash_attention_v2_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,

    q_batch_stride,
    q_heads_stride,
    q_seq_stride,
    q_dim_stride,

    k_batch_stride,
    k_heads_stride,
    k_seq_stride,
    k_dim_stride, # matrix Q stride for columns, [seq_len, head_dim]

    v_batch_stride,
    v_heads_stride,
    v_seq_stride,
    v_dim_stride,

    out_batch_stride,
    out_heads_stride,
    out_seq_stride,
    out_dim_stride,

    n_heads,      # number of heads
    m_size,       # sequence length of q
    n_size,       # sequence length of k, also be rows of K matrix
    HEAD_DIM: tl.constexpr, # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr, # BLOCK size of m_size dimension，即 Q 矩阵行数分成了m_size // BLOCK_M_SIZE 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr, # n_size dimension
    qk_scale,
    causal_mask
    ):
	"""
	flashattention2 内核实现
	"""
	block_m_idx = tl.program_id(0)
	head_idx = tl.program_id(1) # 获取当前 CUDA 块在第二个维度（通常是 blockIdx.y）上的索引。head_idx 表示当前块对应的头（head）的索引。

	cur_batch_idx = head_idx // n_heads # 通过整数除法，将 head_idx 转换为当前批次（batch）的索引。
	cur_head_idx = head_idx % n_heads # 通过取模操作，计算出当前头在其所属批次中的具体索引。

	m_range_offs = tl.arange(0, BLOCK_M_SIZE) # seq_dim 维度偏移
	n_range_offs = tl.arange(0, BLOCK_N_SIZE) # bs*n_heads 维度偏移
	dhead_range_offs = tl.arange(0, HEAD_DIM) # head_dim 维度偏移

	offs_m = block_m_idx * BLOCK_M_SIZE + m_range_offs # 计算当前块在 M(seq_dim) 维度上的实际偏移量。

	# 二维偏移, Compute offsets for the first block on matrix Q K V Output
	offs_q = ( 
		cur_batch_idx * q_batch_stride 
		+ cur_head_idx * q_heads_stride
		+ (offs_m[:, None] * q_seq_stride + dhead_range_offs[None,:] * q_dim_stride))

	offs_k = (
		cur_batch_idx * k_batch_stride 
		+ cur_head_idx * k_heads_stride
		+ (n_range_offs[:,None] * k_seq_stride + dhead_range_offs[None,:] * k_dim_stride))

	offs_v = ( 
		cur_batch_idx * v_batch_stride 
		+ cur_head_idx * v_heads_stride
		+ (n_range_offs[:,None] * v_seq_stride + dhead_range_offs[None,:] * v_dim_stride))

	offs_o = ( 
		cur_batch_idx * out_batch_stride 
		+ cur_head_idx * out_heads_stride
		+ (offs_m[:,None] * out_seq_stride + dhead_range_offs[None,:] * out_dim_stride))

	q_ptrs = q_ptr + offs_q
	k_ptrs = k_ptr + offs_k
	v_ptrs = v_ptr + offs_v
	out_ptrs = o_ptr + offs_o

	q_mask = offs_m[:, None] < m_size
	q = tl.load(q_ptrs, mask=q_mask, other=0.0)

	# 初始化用于计算 softmax 归一化项的 m 和 d, 意义见 online-softmax, 这里
	m_i = tl.zeros([BLOCK_M_SIZE,], dtype=tl.float32) - float("inf")
	d_i = tl.zeros([BLOCK_M_SIZE,], dtype=tl.float32)
	acc = tl.zeros([BLOCK_M_SIZE, HEAD_DIM], dtype=tl.float32)

	# acc 是 attention 输出累加器, d_i 是 softmax 的归一化项（分母）, m_i 是最大值（分子）
	acc, d_i = _attn_fwd_inner(acc, m_i, d_i, q,
                                k_ptrs, v_ptrs,
                                k_seq_stride, v_seq_stride,
                                offs_m,
                                qk_scale, 
                                n_size, # kv seq_len
                                causal_mask,
                                BLOCK_M_SIZE, BLOCK_N_SIZE,
                                v_ptr.dtype.element_ty == tl.float8e5
								)

	acc = acc / d_i[:, None]
	out_mask = offs_m[:, None] < m_size
	tl.store(out_ptrs, acc, mask=out_mask)

@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ):
    """Compute Flash-attention, can't support fp32 input
    参数:
        q: Query tensor, shape: [bs, n_heads, m_size, head_dim], decode 阶段, q 的 seq_len 和 k v 不一致, 其值为 1
        k: Key tensor,  shape: [bs, n_heads, n_size, head_dim]. 
        v: Value tensor, shape is consistent with k. 
        output: Attention ouput tensor, shape is consistent with q. 
        attention_mask: Attention mask matrix broadcastable to (batch, head_size, m_size, n_size).
    """
    output = torch.empty_like(q)
    assert q.device.type == 'cuda', "Input tensor q must be on CUDA device"
    assert k.device.type == 'cuda', "Input tensor keys must be on CUDA device"

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
            q.dtype == k.dtype == v.dtype == output.dtype
        ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"
    
    # sequence length of q, also be rows of Q matrix
    bs, n_heads, m_size, head_dim = q.size()
    causal_mask = False
    if m_size > 1:
        causal_mask: bool = True
        
    n_size = k.shape[2]
    qk_scale = 1 / (head_dim ** 0.5) * 1.4426950408889634 # 1/log(2)
    # BLOCK_M_SIZE = 128
    grid = lambda meta: (triton.cdiv(m_size, meta["BLOCK_M_SIZE"]), bs*n_heads, 1) # 二维 grid

    flash_attention_v2_kernel[grid](
        q,
        k,
        v, 
        output,
        *q.stride(),  # (batch, heads, m_size, head_dim)
        *k.stride(),  # (batch, heads, n_size, head_dim)
        *v.stride(),  # (batch, heads, n_size, head_dim)
        *output.stride(),  # (batch, heads, m_size, n_size)
        n_heads,
        m_size,
        n_size,
        head_dim,
        32,  # BLOCK_M_SIZE
        32,  # BLOCK_N_SIZE
        qk_scale,
        causal_mask
    )
    return output

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
    # 计算 QK^T
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale  # (batch_size, num_heads, seq_length, seq_length)
    
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    
    # print("attn_scores", attn_scores)
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # 计算注意力输出
    out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)
    
    return out

def test_prefill_stage():
    # 设置测试参数
    batch_size = 2
    num_heads = 4
    seq_length = 32
    head_dim = 64

    # 生成固定的输入张量（使用固定随机种子以确保可重复性）
    torch.manual_seed(0)
    q = torch.randn(batch_size, num_heads, seq_length, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_length, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_length, head_dim, device='cuda', dtype=torch.float32)

    # 计算 Softmax 缩放因子
    sm_scale = 1.0 / math.sqrt(head_dim)  # 1 / sqrt(d_k)

    # 调用 Triton 内核
    out = flash_attention_v2(q, k, v)

    # 使用标准 PyTorch 实现计算注意力输出
    # 创建下三角矩阵
    mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0).unsqueeze(0).type_as(q)  # (1, 1, seq, seq)
    standard_o = standard_attention(q, k, v, sm_scale, mask)

    # 比较 Triton 内核输出与标准实现的输出
    if torch.allclose(out, standard_o, atol=1e-2):
        print("Prefill Stage Test Passed: Triton output matches PyTorch standard implementation.")
    else:
        max_diff = (out - standard_o).abs().max()
        print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")
        # 可选择打印更多信息进行调试

def test_decode_stage():
    # 设置测试参数
    batch_size = 1
    num_heads = 4
    initial_seq_length = 16
    generated_seq_length = 16
    head_dim = 64

    # 生成固定的初始输入张量
    torch.manual_seed(0)
    q_initial = torch.randn(batch_size, num_heads, initial_seq_length, head_dim, device='cuda', dtype=torch.float32)
    k_initial = torch.randn(batch_size, num_heads, initial_seq_length, head_dim, device='cuda', dtype=torch.float32)
    v_initial = torch.randn(batch_size, num_heads, initial_seq_length, head_dim, device='cuda', dtype=torch.float32)
    o_initial = torch.zeros_like(q_initial, device='cuda', dtype=torch.float32)
    new_token_q = torch.randn(batch_size, num_heads, 1, head_dim, device='cuda', dtype=torch.float32)

    triton_k_extended = k_initial
    triton_v_extended = v_initial
    torch_k_extended = k_initial
    torch_v_extended = v_initial
    torch_new_token_q = new_token_q
    triton_new_token_q = new_token_q
    # 模拟生成过程中逐步增加序列长度
    for step in range(1, generated_seq_length + 1):
        # 生成新的 token
        triton_k_extended = torch.cat([triton_k_extended, triton_new_token_q], dim=2)
        triton_v_extended = torch.cat([triton_v_extended, triton_new_token_q], dim=2)
        
        torch_k_extended = torch.cat([torch_k_extended, torch_new_token_q], dim=2)
        torch_v_extended = torch.cat([torch_v_extended, torch_new_token_q], dim=2)

        # 扩展 Q, K, V 和 Out
        # q_extended = torch.cat([q_initial, new_token_q], dim=2)

        # 计算 Softmax 缩放因子
        sm_scale_extended = 1.0 / math.sqrt(head_dim)

        # 计算 Triton 内核输出
        triton_new_token_q = flash_attention_v2(new_token_q, triton_k_extended, triton_v_extended)

        # 使用标准 PyTorch 实现计算扩展后的注意力输出
        torch_new_token_q = standard_attention(new_token_q, torch_k_extended, torch_v_extended, sm_scale_extended)

        # 比较 Triton 内核输出与标准实现的输出
        if torch.allclose(triton_new_token_q, torch_new_token_q, atol=1e-1):
            print(f"Decode Stage Step {step} Test Passed: Triton output matches PyTorch standard implementation.")
        else:
            max_diff = (triton_new_token_q - torch_new_token_q).abs().max()
            print(f"Decode Stage Step {step} Test Failed: Maximum difference {max_diff}")
            # 可选择打印更多信息进行调试
            break  # 根据需要是否停止测试

if __name__ == "__main__":
    print("Running Prefill Stage Test...")
    test_prefill_stage()
    print("\nRunning Decode Stage Test...")
    test_decode_stage()

"""
Running Prefill Stage Test...
Prefill Stage Test Passed: Triton output matches PyTorch standard implementation.

Running Decode Stage Test...
Decode Stage Step 1 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 2 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 3 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 4 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 5 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 6 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 7 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 8 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 9 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 10 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 11 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 12 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 13 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 14 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 15 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 16 Test Passed: Triton output matches PyTorch standard implementation.
"""