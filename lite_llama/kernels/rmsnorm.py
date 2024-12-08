# modified from https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
import triton,torch, os
import triton.language as tl 

@triton.jit
def _rmsnorm_kernel_fwd(
    x_ptr, # shape is [M, N]
    w_ptr, # gamma 参数地址
    z_ptr, # 输出结果首元素指针
    K,
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr = 8,
):
    # z = (x / (rms)) * w
    
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * K # 一行有 K 个元素，K 是最后一维
    w_row_ptr = w_ptr + row_idx * K
    z_row_ptr = z_ptr + row_idx * K
    
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_index in range(0, K, BLOCK_SIZE):
        col_offsets = col_index + tl.arange(0, BLOCK_SIZE)
        x_ptrs = x_row_ptr + col_offsets
        
        x = tl.load(x_ptrs, mask = col_offsets < K, other=0.0).to(tl.float32)
        _var += x*x
    
    var = tl.sum(_var, axis=0) / K
    rsqrt =  1 / tl.sqrt(var + eps)
    
    # Normalize and apply rmsnorm
    for col_index in range(0, K, BLOCK_SIZE):
        col_offsets = col_index + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < K
        
        x = tl.load(x_row_ptr + col_offsets, mask = mask, other=0.0)
        w = tl.load(w_ptr + col_offsets, mask = mask).to(tl.float32)
        
        normed = x * rsqrt
        normed = normed.to(w.dtype) # Exact copy from HF
        z =normed * w
        tl.store(z_row_ptr + col_offsets, z, mask = mask)
        
@torch.no_grad()
def rmsnorm(
    x,
    weight,
    eps = 1e-5
):
    # 只针对 nlp 领域的 layernorm，省去了 normalized_shape 参数
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert x.shape[-1] == weight.shape[0]
    
    out_shape = x.shape
    # 将 x 的所有维度压缩为二维张量, [B, L, K] -> [M, K], K 是隐藏层的维度。
    x = x.view((-1, x.shape[-1]))
    M, K = x.shape
    x = x.view((M, K))
    z = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(K))

    grid = (triton.cdiv(K, BLOCK_SIZE), 1)
    _rmsnorm_kernel_fwd[M, ](
        x,
        weight,
        z,
        K, 
        eps = eps,
        BLOCK_SIZE=BLOCK_SIZE,    
    )  
    return z.view(out_shape)

def torch_rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def test_rms_norm(M, N, dtype, eps=1e-5, device="cuda"):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    # forward pass
    y_tri = rmsnorm(x, weight, eps)
    y_ref = torch_rms_norm(x.to(torch.float32), weight.to(torch.float32), eps).to(dtype)

    # compare
    print("type:", y_tri.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    assert torch.allclose(y_tri, y_ref, atol=1e-3, rtol=0)
    return

if __name__ == "__main__":
    test_rms_norm(100, 256, torch.float16)