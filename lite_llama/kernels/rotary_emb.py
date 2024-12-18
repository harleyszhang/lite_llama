import torch
import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q,
    K,
    Cos,
    Sin,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    max_total_len,
    HEAD_Q,
    HEAD_K,  # N_CTX 代表要计算的上下文长度
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_head_index = tl.program_id(0)
    cur_seq_index = tl.program_id(1)

    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
    dim_range1 = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)

    off_q0 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range0[None, None, :] * stride_qd
    )
    off_q1 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range1[None, None, :] * stride_qd
    )

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd

    q0 = tl.load(
        Q + off_q0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q),
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q),
        other=0.0,
    )

    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

    tl.store(
        Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q)
    )
    tl.store(
        Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q)
    )

    off_k0 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range0[None, None, :] * stride_kd
    )
    off_k1 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range1[None, None, :] * stride_kd
    )

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd

    k0 = tl.load(
        K + off_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
        other=0.0,
    )
    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos

    tl.store(
        K + off_k0,
        out_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
    )
    tl.store(
        K + off_k1,
        out_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K),
    )
    return


@torch.no_grad()
def rotary_emb_fwd(q, k, cos, sin, partial_rotary_factor=1.):
    total_len = q.shape[0]
    head_num_q, head_num_k = q.shape[1], k.shape[1]
    head_dim = int(q.shape[2] * partial_rotary_factor)
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    assert k.shape[0] == cos.shape[0] and k.shape[0] == sin.shape[0], f"k shape {k.shape} cos shape {cos.shape}"

    BLOCK_SEQ = 64
    BLOCK_HEAD = 4
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    grid = (triton.cdiv(head_num_q, BLOCK_HEAD), triton.cdiv(total_len, BLOCK_SEQ))
    _rotary_kernel[grid](
        q,
        k,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        total_len,
        head_num_q,
        head_num_k,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, k

def torch_rotary_emb(x, cos, sin):
    seq_len, h, d = x.shape
    # cos, sin 的形状为 (seq_len, d//2)
    half_dim = cos.shape[-1]
    x0 = x[:, :, :half_dim]
    x1 = x[:, :, half_dim: 2*half_dim]

    cos = cos.view(seq_len, 1, half_dim)
    sin = sin.view(seq_len, 1, half_dim)

    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos

    if 2 * half_dim < d:
        out = torch.cat([o0, o1, x[:, :, 2*half_dim:]], dim=-1)
    else:
        out = torch.cat([o0, o1], dim=-1)

    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    batch_tokens = 24800
    x_shape = (batch_tokens, 32, 64)  # (seq_len, num_heads, head_dim)
    dtype = torch.float16
    q = torch.randn(x_shape, dtype=dtype, device='cuda')
    k = torch.clone(q)

    # 生成 cos 和 sin，与 head_dim 对应，这里 head_dim=64，因此 cos, sin=(seq_len, head_dim//2)=(128,32)
    cos_shape = (batch_tokens, 32)  
    y = torch.randn(cos_shape, dtype=dtype, device='cuda')
    cos = y.cos()
    sin = y.sin()

    output_torch = torch_rotary_emb(q, cos, sin)
    q_out, k_out = rotary_emb_fwd(q, k, cos, sin)
    print(output_torch)
    print(q_out)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - q_out))}')
    print('torch:', triton.testing.do_bench(lambda: torch_rotary_emb(q, cos, sin)))
    print('triton:', triton.testing.do_bench(lambda: torch_rotary_emb(q, cos, sin)))