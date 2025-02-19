import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # q size: (bsz, seq_len, num_q_heads, head_dim)
    # q stride: (seq_len * num_q_heads * head_dim, num_q_heads * head_dim, head_dim, 1)
    # k size: (bsz, seq_len, num_kv_heads, head_dim)
    # k stride: (seq_len * num_kv_heads * head_dim, num_kv_heads * head_dim, head_dim, 1)

    # cos size: (1, seq_len, head_dim)
    # stride: (seq_len * head_dim, head_dim, 1)
    pid = tl.program_id(0)

    # locate start address
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # 1. program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    # 2. We only need the left half of cos and sin matrix because the right half is just
    # a clone of the left half.
    cos_row_idx = pid % (sl)
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)

def rope_forward(q, k, cos, sin):
    """
    输入 q、k 参数是 4 维张量
    """
    # transpose it back to the physical shape because Triton looks at the physical storage
    # note: q and k are incontiguous before the transformation and will become contiguous after transpose
    # q = q.transpose(1, 2)
    # k = k.transpose(1, 2)
    
    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_rope[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q, k

@triton.jit
def _triton_rope_emb(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_b_stride,
    cos_s_stride,
    sin,
    sin_b_stride,
    sin_s_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // sl
    cos_row_idx = pid % sl

    # 定位到 q, k 行起点
    q_ptr += pid * q_row_stride
    k_ptr += pid * k_row_stride

    # 定位到 cos, sin 对应 batch_id 的 cos_row_idx 行
    cos_ptr = cos + batch_id * cos_b_stride + cos_row_idx * cos_s_stride
    sin_ptr = sin + batch_id * sin_b_stride + cos_row_idx * sin_s_stride

    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos_ptr + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin_ptr + cos_offsets, mask=cos_mask, other=0)

    # 计算 head 和 dim 偏移
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )

    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)

def rope_emb_forward(q, k, cos, sin, batch_size, seq_len):
    """
    q: (batch_size * seq_len, n_q_heads, head_dim)
    k: (batch_size * seq_len, n_k_heads, head_dim)
    cos, sin: (batch_size, seq_len, head_dim)
    """
    N, n_qh, HEAD_DIM = q.shape
    _, n_kh, _ = k.shape
    assert N == batch_size * seq_len

    pad_hd = triton.next_power_of_2(HEAD_DIM)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    BLOCK_SIZE = max(pad_n_qh, pad_n_kh)

    if HEAD_DIM >= 128:
        num_warps = 8
    else:
        num_warps = 4

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_rope_emb[(N,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        cos,
        cos.stride(0),
        cos.stride(1),
        sin,
        sin.stride(0),
        sin.stride(1),
        seq_len,
        batch_size,
        n_qh,
        n_kh,
        HEAD_DIM,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
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
    # 单元测试有 bug 等待修复
    torch.manual_seed(0)
    batch_size = 248
    seq_len = 100
    head_dim = 64
    batch_tokens = batch_size * seq_len
    x_shape = (batch_tokens, 32, 64)  # (seq_len, num_heads, head_dim)
    dtype = torch.float16
    q = torch.randn(x_shape, dtype=dtype, device='cuda')
    k = torch.clone(q)

    triton_q = q.view(batch_size, seq_len, 32, 64)
    triton_k = k.view(batch_size, seq_len, 32, 64)

    # 生成 cos 和 sin，与 head_dim 对应，这里 head_dim=64，因此 cos, sin=(seq_len, head_dim//2)=(128,32)
    cos_shape = (batch_tokens, 32)  
    y = torch.randn(cos_shape, dtype=dtype, device='cuda')
    cos = y.cos()
    sin = y.sin()
    
    triton_cos = cos.view(seq_len, 1, head_dim)
    triton_sin = sin.view(seq_len, 1, head_dim)

    output_torch = torch_rotary_emb(q, cos, sin)
    q_out, k_out, _, _ = rope_forward(triton_q, triton_k, triton_cos, triton_cos)
    triton_q_out = q_out.view(-1, 32, 64)
    print(f"output_torch shape {output_torch.shape}, triton_q_out shape {triton_q_out.shape}")
    
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - triton_q_out))}')
    print('torch:', triton.testing.do_bench(lambda: torch_rotary_emb(q, cos, sin)))
    print('triton:', triton.testing.do_bench(lambda: rope_forward(triton_q, triton_k, cos, sin)))