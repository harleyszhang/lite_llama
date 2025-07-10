import torch

import triton
import triton.language as tl


@triton.jit
def matmul_silu_kernel(
    # Pointers to matrices
    a_ptr,
    w1_ptr,
    w2_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_am,
    stride_ak,  # input
    stride_w1k,
    stride_w1n,  # weight 1
    stride_w2k,
    stride_w2n,  # weight 2
    stride_cm,
    stride_cn,  # output
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """
    Fused kernel for computing F.silu(w1(x)) * w2(x)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to pid_m and pid_n
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w2_ptrs = w2_ptr + (offs_k[:, None] * stride_w2k + offs_bn[None, :] * stride_w2n)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc1 += tl.dot(a, w1)
        w2 = tl.load(w2_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc2 += tl.dot(a, w2)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w2_ptrs += BLOCK_SIZE_K * stride_w2k

    # -----------------------------------------------------------
    # Fuse silu activation function
    # option 1: all in fp32
    c = (acc1 * tl.sigmoid(acc1)) * acc2

    # option 2: silu in fp32
    # acc1 = (acc1 * tl.sigmoid(acc1)).to(tl.float16)
    # acc2 = acc2.to(tl.float16)
    # c = acc1 * acc2

    # -----------------------------------------------------------
    # Write back the block of the output matrix
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def mlp_silu(x, w1, w2, w3):
    # Check constraints.
    assert x.shape[-1] == w1.shape[0], "Incompatible dimensions"
    assert x.shape[-1] == w2.shape[0], "Incompatible dimensions"
    assert w1.shape[1] == w2.shape[1], "Incompatible dimensions"

    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w1.is_contiguous(), "Matrix W1 must be contiguous"
    assert w2.is_contiguous(), "Matrix W2 must be contiguous"

    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim
    N = w1.shape[1]
    x = x.view(M, K)

    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 这里的 grid 针对 (M,K) 输出维度进行网格划分
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  # 用于中间N和最终K的分块大小
    BLOCK_SIZE_K = 128  # 用于中间K维的分块大小
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_silu_kernel[grid](
        x,
        w1,
        w2,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        num_stages=2,
        num_warps=4,
    )

    M, K = out.shape
    K, N = w3.shape

    # Allocates output.
    mlp_silu_out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        out,
        w3,
        mlp_silu_out,
        M,
        N,
        K,
        out.stride(0),
        out.stride(1),
        w3.stride(0),
        w3.stride(1),
        mlp_silu_out.stride(0),
        mlp_silu_out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        num_stages=2,
        num_warps=4,
    )

    mlp_silu_out = mlp_silu_out.view(batch, seq_len, -1)
    return mlp_silu_out


def triton_torch_mlp_silu(x, w1, w2, w3):
    # Check constraints.
    assert x.shape[-1] == w1.shape[0], "Incompatible dimensions"
    assert x.shape[-1] == w2.shape[0], "Incompatible dimensions"
    assert w1.shape[1] == w2.shape[1], "Incompatible dimensions"

    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w1.is_contiguous(), "Matrix W1 must be contiguous"
    assert w2.is_contiguous(), "Matrix W2 must be contiguous"

    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim
    N = w1.shape[1]
    x = x.view(M, K)

    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 这里的 grid 针对 (M,K) 输出维度进行网格划分
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  # 用于中间N和最终K的分块大小
    BLOCK_SIZE_K = 128  # 用于中间K维的分块大小
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_silu_kernel[grid](
        x,
        w1,
        w2,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        num_stages=2,
        num_warps=4,
    )

    mlp_silu_out = torch.mm(out, w3)  # MxK
    mlp_silu_out = mlp_silu_out.view(batch, seq_len, -1)
    return mlp_silu_out


import torch.nn as nn
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.kernels.swiglu import swiglu_forward


def torch_mlp_silu(x, w1, w2, w3):
    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim
    x = x.view(M, K)
    y1 = torch.mm(x, w1)  # MxN
    y2 = torch.mm(x, w2)  # MxN
    out = swiglu_forward(y1, y2)
    mlp_silu_out = torch.mm(out, w3)  # MxK
    mlp_silu_out = mlp_silu_out.view(batch, seq_len, -1)
    return mlp_silu_out


class FusedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16
        )

    def forward(self, x):
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    seq_len = 256
    hidden_size = 3584
    intermediate_size = 18944
    x = torch.randn(B, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    w1 = (
        torch.randn(
            (intermediate_size, hidden_size), device="cuda", dtype=torch.float16
        )
        * 0.01
    )
    w2 = (
        torch.randn(
            (intermediate_size, hidden_size), device="cuda", dtype=torch.float16
        )
        * 0.01
    )
    w3 = (
        torch.randn(
            (hidden_size, intermediate_size), device="cuda", dtype=torch.float16
        )
        * 0.01
    )

    w1_t = w1.t().contiguous()
    w2_t = w2.t().contiguous()
    w3_t = w3.t().contiguous()

    triton_output = mlp_silu(x, w1_t, w2_t, w3_t)
    triton_torch_output = triton_torch_mlp_silu(x, w1_t, w2_t, w3_t)
    torch_output = torch_mlp_silu(x, w1_t, w2_t, w3_t)
    torch_fused_mlp = FusedMLP(hidden_size, intermediate_size).cuda()
    torch_fused_mlp_out = torch_fused_mlp(x)

    # assert torch.allclose(torch_output, triton_output, atol=1e-2)
    # assert(torch.amax(torch_output - triton_output).item() <= 0.05)
    print(
        f"Max diff: {torch.max(torch.abs(torch_output - triton_output))}"
    )  # assert(torch.amax(Y - Y2).item() <= 0.05)
    print(
        f"Max diff: {torch.max(torch.abs(torch_output - triton_torch_output))}"
    )  # assert(torch.amax(Y - Y2).item() <= 0.05)
    print(
        f"Max diff: {torch.max(torch.abs(torch_output - torch_fused_mlp_out))}"
    )  # assert(torch.amax(Y - Y2).item() <= 0.05)

    print(
        "torch:", triton.testing.do_bench(lambda: torch_mlp_silu(x, w1_t, w2_t, w3_t))
    )
    print("triton:", triton.testing.do_bench(lambda: mlp_silu(x, w1_t, w2_t, w3_t)))
    print(
        "triton_torch:",
        triton.testing.do_bench(lambda: triton_torch_mlp_silu(x, w1_t, w2_t, w3_t)),
    )
    print("torch_fused_mlp:", triton.testing.do_bench(lambda: torch_fused_mlp(x)))
