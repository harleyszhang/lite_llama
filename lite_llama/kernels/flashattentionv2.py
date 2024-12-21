# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438

import torch
import triton
import triton.language as tl

# TESLA = "Tesla" in torch.cuda.get_device_name(0)

@triton.jit
def _attn_fwd_inner(
	acc, m_i, d_i, q,
	k_ptrs, v_ptrs, 
	k_seq_stride, v_seq_stride,
	offs_m,
	qk_scale, 
	n_size, # kv seq_len
	BLOCK_M_SIZE: tl.constexpr, BLOCK_N_SIZE: tl.constexpr,
    fp8_v: tl.constexpr
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
        offs_k = block_n_offs
        # casual 模型的 causal mask 下三角矩阵
        mask = offs_m[:, None] >= offs_k[None, :]
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e8)
        # qk = tl.where(mask, qk * qk_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1)) # 求 qk 的最大值
        qk -= m_ij[:, None] # 更新为安全的 qk
        
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

    num_kv_groups, # group of kv heads
    n_heads,      # number of heads
    m_size,       # sequence length of q
    n_size,       # sequence length of k, also be rows of K matrix
    HEAD_DIM: tl.constexpr, # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr, # BLOCK size of m_size dimension，即 Q 矩阵行数分成了m_size // BLOCK_M_SIZE 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr, # n_size dimension
    qk_scale,
    ):
    """
    flashattention2 内核实现
    """
    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1) # 获取当前 CUDA 块在第二个维度（通常是 blockIdx.y）上的索引。head_idx 表示当前块对应的头（head）的索引。

    cur_batch_idx = head_idx // n_heads # 通过整数除法，将 head_idx 转换为当前批次（batch）的索引。
    cur_head_idx = head_idx % n_heads # 通过取模操作，计算出当前头在其所属批次中的具体索引。

    cur_kv_head_idx = cur_head_idx // num_kv_groups # 支持 GQA 模型直接获取 kv heads index, 也兼容非 GQA 模型

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
        + cur_kv_head_idx * k_heads_stride
        + (n_range_offs[:,None] * k_seq_stride + dhead_range_offs[None,:] * k_dim_stride))

    offs_v = ( 
        cur_batch_idx * v_batch_stride 
        + cur_kv_head_idx * v_heads_stride
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
                                BLOCK_M_SIZE, BLOCK_N_SIZE,
                                v_ptr.dtype.element_ty == tl.float8e5)

    acc = acc / d_i[:, None]
    out_mask = offs_m[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)

@torch.no_grad()
def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
	qk_scale
    ):
    """Compute Flash-attention, can't support fp32 input
    参数:
        q: Query tensor, shape: [bs, n_heads, m_size, head_dim], decode 阶段, q 的 seq_len 和 k v 不一致, 其值为 1
        k: Key tensor,  shape: [bs, n_heads, n_size, head_dim]. 
        v: Value tensor, shape is consistent with k. 
        output: Attention ouput tensor, shape is consistent with q. 
        attention_mask: Attention mask matrix broadcastable to (batch, head_size, m_size, n_size).
    """
    BLOCK_SIZE = 64 # default: BLOCK_M_SIZE = 64
    num_kv_groups = q.shape[1] // k.shape[1] # num_q_heads // num_k_heads
    output = torch.empty_like(q)

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
            q.dtype == k.dtype == v.dtype == output.dtype
        ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"

    # sequence length of q, also be rows of Q matrix
    bs, n_heads, m_size, head_dim = q.size()

    n_size = k.shape[2]
    
    grid = lambda meta: (triton.cdiv(m_size, BLOCK_SIZE), bs*n_heads, 1) # 二维 grid

    flash_attention_v2_kernel[grid](
        q,
        k,
        v, 
        output,
        *q.stride(),  # (batch, heads, m_size, head_dim)
        *k.stride(),  # (batch, heads, n_size, head_dim)
        *v.stride(),  # (batch, heads, n_size, head_dim)
        *output.stride(),  # (batch, heads, m_size, n_size)
        num_kv_groups,
        n_heads,
        m_size,
        n_size,
        head_dim,
        BLOCK_SIZE,  # BLOCK_M_SIZE
        BLOCK_SIZE,  # BLOCK_N_SIZE
        qk_scale,
    )
    return output