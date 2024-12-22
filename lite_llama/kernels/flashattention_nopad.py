# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438

import torch,math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd

# TODO: integrating rope with flash-attn
@triton.jit
def flash_attention_v1_kernel(
    Q, K, V, O,
    B_Start_Loc, B_Seqlen, 
    sm_scale, num_kv_groups, max_seq_len,       # group of kv heads

    stride_q_bs, stride_q_heads, stride_q_dim,  # Q 的 strides
    stride_k_bs, stride_k_heads, stride_k_dim,  # K 的 strides
    stride_v_bs, stride_v_heads, stride_v_dim,  # V 的 strides
    stride_o_bs, stride_o_heads, stride_o_dim,
    HEAD_DIM, 
    BLOCK_DHEAD_SIZE: tl.constexpr, # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr, # BLOCK size of m_size dimension，即 Q 矩阵行数分成了m_size // BLOCK_M_SIZE 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr, # n_size dimension    
):
    """
    flashattentionv1 内核实现, 支持 nopad 计算, 输入为 3 维张量
    """
    block_m_idx = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch_idx = cur_bh // HEAD_DIM
    cur_head_idx = cur_bh % HEAD_DIM
    
    cur_kv_head_idx = cur_head_idx // num_kv_groups

    # 计算当前批次的序列长度和请求序列的起始位置
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch_idx)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch_idx)

    block_start_loc = block_m_idx * BLOCK_M_SIZE # 计算当前 block 的起始和结束索引

    offs_n = tl.arange(0, BLOCK_N_SIZE) # head_dim 维度偏移
    offs_d = tl.arange(0, BLOCK_DHEAD_SIZE)
    offs_m = block_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)

    # Compute offsets for the first block on matrix Q K V Output
    q_offs = (
        (cur_batch_start_loc + offs_m[:, None]) * stride_q_bs
        + cur_head_idx * stride_q_heads
        + offs_d[None, :] * stride_q_dim
    )

    k_offs = offs_n[None, :] * stride_k_bs + cur_kv_head_idx * stride_k_heads + offs_d[:, None] * stride_k_dim
    v_offs = offs_n[:, None] * stride_v_bs + cur_kv_head_idx * stride_v_heads + offs_d[None, :] * stride_v_dim
    
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    # 初始化用于计算 softmax 归一化项的 m 和 d, 意义见 online-softmax, 这里
    l_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)
        
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M_SIZE, cur_batch_seq_len)
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N_SIZE):
    # for start_n in range(0, block_mask * (block_m_idx + 1) * BLOCK_M_SIZE, BLOCK_N_SIZE):

        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)
        # block_n_offs = start_n + offs_n
        # k_mask = block_n_offs[None, :] < cur_batch_seq_len

        # 计算 qk^t
        k = tl.load(
            k_ptrs + (cur_batch_start_loc + start_n) * stride_k_bs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
        )

        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        
        # offs_k = block_n_offs
        # mask = offs_m[:, None] >= offs_k[None, :]  # casual 模型的 causal mask 下三角矩阵
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf")) # 应用因果遮罩

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
        # sigma = tl.where(offs_m >= start_n, sigma, 1.0)

        acc = acc * sigma[:, None]
        
        # compute O = PV # update acc
        v = tl.load(
            v_ptrs + (cur_batch_start_loc + start_n) * stride_v_bs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )

        p = p.to(v.dtype)

        acc += tl.dot(p, v)

        # update the normalizer (l and d) for next iteration
        l_i = l_new
        d_i = d_new
    
    o_offs = (
        (cur_batch_start_loc + offs_m[:, None]) * stride_o_bs
        + cur_head_idx * stride_o_heads
        + offs_d[None, :] * stride_o_dim
    )
    out_ptrs = O + o_offs
    out_mask = offs_m[:, None] < cur_batch_seq_len
    tl.store(out_ptrs, acc, mask=out_mask)
    return 

@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def flash_attention_v1_no_pad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc, 
    b_seq_len, 
    max_seq_len,
    sm_scale,
    ):
    """Compute Flash-attention, can't support fp32 input
    参数:
        q: Query tensor, shape: [bs*m_size, n_heads, head_dim], decode 阶段, q 的 seq_len 和 k v 不一致, 其值为 1
        k: Key tensor,  shape: [bs*n_size, n_heads, head_dim]. 
        v: Value tensor, shape is consistent with k. 
    """
    BLOCK_SIZE = 32 # For Ampere Architecture, 3090ti
    output = torch.empty_like(q)
    batchs = b_seq_len.shape[0]
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]

    num_kv_groups = q.shape[1] // k.shape[1] # num_q_heads // num_k_heads
    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M_SIZE"]), batchs * n_heads, 1)
    # grid = (batchs, n_heads, triton.cdiv(max_seq_len + BLOCK_SIZE - 1, BLOCK_SIZE))  # batch, head,
    num_warps = 2 if HEAD_DIM <= 64 else 4

    flash_attention_v1_kernel[grid](
        q,
        k,
        v, 
        output,
        b_start_loc,
        b_seq_len,
        sm_scale,
        num_kv_groups,  
        max_seq_len,

        *q.stride(),  # (batch * m_size, heads, head_dim)
        *k.stride(),  # (batch * n_size, heads, head_dim)
        *v.stride(),  # (batch * n_size, heads, head_dim)
        *output.stride(),  # (batch * m_size, heads, n_size)
        HEAD_DIM, 
        BLOCK_DHEAD_SIZE=HEAD_DIM, 
        BLOCK_M_SIZE=BLOCK_SIZE,
        BLOCK_N_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return output

def _naive_attention(q, k ,v):
    import math
    bs, seqlen, num_head, head_dim = q.shape
    device = q.device
    mask = 1.0 - torch.tril(torch.ones((seqlen, seqlen), device=device), diagonal=0).unsqueeze(0).unsqueeze(0)
    mask.masked_fill_(mask.to(torch.bool), -float("-inf"))
    q = q.transpose(1, 2) #(bs, num_head, seqlen, head_dim)
    k = k.transpose(1, 2) #(bs, num_head, seqlen, head_dim)
    v = v.transpose(1, 2) #(bs, num_head, seqlen, head_dim)
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    scores = torch.nn.functional.softmax(scores.float() + mask, dim=-1).to(q.dtype)
    output = torch.matmul(scores, v).transpose(1, 2).contiguous().reshape(bs, seqlen, num_head, head_dim)
    return output

def _sdpa(q, k, v):
    bs, seqlen, num_head, head_dim = q.shape
    q = q.transpose(1, 2) #(bs, num_head, seqlen, head_dim)
    k = k.transpose(1, 2) #(bs, num_head, seqlen, head_dim)
    v = v.transpose(1, 2) #(bs, num_head, seqlen, head_dim)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, num_head, head_dim)
    return output

def torch_attention(q, k, v, b_start_loc, b_seq_len, sdpa=False):
    out = torch.empty_like(q)
    Z = b_start_loc.shape[0]
    for i in range(Z):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        # 添加检查，避免越界
        if end > q.shape[0]:
            raise ValueError(f"Batch {i}: end index {end} exceeds tensor size {q.shape[0]}")
        qi = q[start:end].unsqueeze(0)
        ki = k[start:end].unsqueeze(0)
        vi = v[start:end].unsqueeze(0)
        if sdpa:
            oi = _sdpa(qi, ki, vi)
        else:
            oi = _naive_attention(qi, ki, vi)
        out[start:end] = oi.squeeze(0)
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    # inputs
    batchs, head, head_dim = 4 * 1024, 64, 128 #(ntoken, nhead, head_dim)
    shape = (batchs, head, head_dim)

    dtype = torch.float16
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")
    # meta data
    max_input_len = 1024
    
    b_start_loc = torch.tensor([0, 512, 1536, 2048], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda")
    # compute attention
    sm_scale = 1 / math.sqrt(head_dim)
    triton_output = flash_attention_v1_no_pad(q, k, v, b_start_loc, b_seq_len, max_input_len, sm_scale)
    torch_output = torch_attention(q, k, v, b_start_loc, b_seq_len, sdpa=True)
    print("triton_output", triton_output)
    print("torch_output", torch_output)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')
    # benchmark
    print("torch:", triton.testing.do_bench(lambda: torch_attention(q, k, v, b_start_loc, b_seq_len, sdpa=True)))
    print("triton:", triton.testing.do_bench(lambda: flash_attention_v1_no_pad(q, k, v, b_start_loc, b_seq_len, max_input_len, sm_scale)))