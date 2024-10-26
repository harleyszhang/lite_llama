# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438

import torch,math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
from typing import List, Optional, Union

# TODO: integrating rope with flash-attn
@triton.jit
def flash_attention_v1_kernel(
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
    m_size,
    n_size,       # sequence length of k, also be rows of K matrix
    BLOCK_DHEAD_SIZE: tl.constexpr, # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr, # BLOCK size of m_size dimension，即 Q 矩阵行数分成了m_size // BLOCK_M_SIZE 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr, # n_size dimension
    sm_scale,
    ):
    """
    针对 prefill 阶段的 attention, 所以忽略了添加 mask 过程
    """
    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads

    m_range_offs = tl.arange(0, BLOCK_M_SIZE)
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)
    dhead_range_offs = tl.arange(0, BLOCK_DHEAD_SIZE)

    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs

    # Compute offsets for the first block on matrix Q K V Output
    q_offs = ( 
        cur_batch_idx * q_batch_stride 
        + cur_head_idx * q_heads_stride
        + (m_offs[:, None] * q_seq_stride + dhead_range_offs[None,:] * q_dim_stride))

    k_offs = (
        cur_batch_idx * k_batch_stride 
        + cur_head_idx * k_heads_stride
        + (n_range_offs[:,None] * k_seq_stride + dhead_range_offs[None,:] * k_dim_stride))
    
    v_offs = ( 
        cur_batch_idx * v_batch_stride 
        + cur_head_idx * v_heads_stride
        + (n_range_offs[:,None] * v_seq_stride + dhead_range_offs[None,:] * v_dim_stride))

    o_offs = ( 
        cur_batch_idx * out_batch_stride 
        + cur_head_idx * out_heads_stride
        + (m_offs[:,None] * out_seq_stride + dhead_range_offs[None,:] * out_dim_stride))
    
    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    out_ptrs = o_ptr + o_offs

    # 初始化用于计算 softmax 归一化项的 m 和 d, 意义见 online-softmax, 这里
    l_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)
    
    q_mask = m_offs[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
        block_n_offs = block_n_start_idx + n_range_offs
        k_mask = block_n_offs[:, None] < n_size
        k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask, other=0.0)
        
        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        l_j = tl.max(qk, 1)
        numerators = tl.exp(qk - l_j[:, None])
        d_j = tl.sum(numerators, 1) # 1d vector

        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha * d_i  + beta * d_j
        
        # compute softmax(qk)
        p_scale = beta / d_new
        p = numerators * p_scale[:, None]
        # acc scaling
        sigma = d_i / d_new * alpha
        acc = acc * sigma[:, None]
        
        # compute O = PV
        v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask, other=0.0)
        p = p.to(q_ptr.dtype.element_ty)

        acc += tl.dot(p, v)

        # update the normalizer (l and d) for next iteration
        l_i = l_new
        d_i = d_new
    
    out_mask = m_offs[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)

@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale,
    attention_mask: Optional[torch.Tensor] = None,
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
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
            q.dtype == k.dtype == v.dtype == output.dtype
        ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"
    
    # sequence length of q, also be rows of Q matrix
    bs, n_heads, m_size, head_dim = q.size()
    n_size = k.shape[2]
    # sm_scale = 1 / math.sqrt(head_dim)
    # BLOCK_M_SIZE = 128
    grid = lambda meta: (triton.cdiv(m_size, meta["BLOCK_M_SIZE"]), bs*n_heads, 1) # 二维 grid

    flash_attention_v1_kernel[grid](
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
        64,  # BLOCK_M_SIZE
        64,  # BLOCK_N_SIZE
        sm_scale
    )
    return output