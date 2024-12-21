# modified from https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
import triton,torch, os
import triton.language as tl 
from ..utils import calculate_settings

@triton.jit
def _rmsnorm_kernel_fwd(
    x_ptr, # shape is [M, K]
    w_ptr, # gamma 参数地址
    z_ptr, # 输出结果首元素指针
    K,     # 权重 W 大小, 也是输入 X 的第二维度大小
    eps,   # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr = 8,
):
    """z = (x / (rms)) * w"""
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * K # 一行有 K 个元素，K 是最后一维
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
        
        x = tl.load(x_row_ptr + col_offsets, mask = mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col_offsets, mask = mask).to(tl.float32)
        
        normed = x * rsqrt
        normed = normed.to(w.dtype) # Exact copy from HF
        z = normed * w
        tl.store(z_row_ptr + col_offsets, z.to(z.dtype), mask = mask)
        
@torch.no_grad()
def rmsnorm(
    x,
    weight,
    eps = 1e-5
):
    z = torch.empty_like(x) # z 是三维的, [B, L, K]
    out_shape = x.shape
    x = x.view((-1, x.shape[-1])) # 将 x 的所有维度压缩为二维张量, [B, L, K] -> [M, K], K 是隐藏层的维度。
    M, K = x.shape
    
    # Less than 64KB per feature: enqueue fused kernel
    # MAX_FUSED_SIZE = 65536 // x.element_size() # 用于返回张量中单个元素的大小（以字节为单位）。 
    BLOCK_SIZE, num_warps = calculate_settings(K)
    if K > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _rmsnorm_kernel_fwd[M, ](
        x,
        weight,
        z,
        K, 
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps, 
    )  
    return z.view(out_shape)

def test_rms_layernorm(
    dim = 1024, eps = 1e-5, dtype = torch.float16,
    bsz = 21, random_state = 3407, seqlen = 3341,
):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    Y = layernorm(X)
    Y2 = rmsnorm(X, layernorm.weight, eps)

    assert(torch.amax(Y - Y2).item() <= 0.05)
    print("max delta:", torch.max(torch.abs(Y - Y2)))


def testing_suite_layernorm():
    for dim in [512, 1024, 2048]:
        for dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type = "cuda", dtype = dtype):
                for seqlen in [3341, 2048, 349]:
                    for random_state in [3407, 42]:
                        test_rms_layernorm(
                            dim = dim,
                            eps = 1e-5,
                            dtype = dtype,
                            bsz = 21,
                            random_state = random_state,
                            seqlen = seqlen,
                        )

if __name__ == "__main__":
    testing_suite_layernorm()