import triton
import triton.language as tl
import torch
import torch.nn as nn
import numpy as np
from ..quantization.gptq.gptq import GPTQ

@triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
        ],
        key=["M", "N", "K"],
    )

@triton.jit
def int4_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    bscales_ptr, bzeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N

    a_ptrs = a_ptr + stride_am * offs_am[:, None] + stride_ak * offs_k[None, :]
    b_ptrs = b_ptr + stride_bn * offs_bn[None, :] + stride_bk * (offs_k[:, None] // 2)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    for k in range(0, K, BLOCK_SIZE_K):
        b_q = tl.load(b_ptrs, mask=b_mask)

        a = tl.load(a_ptrs, mask=a_mask).to(tl.float16)

        # Compute per-group index
        k_offset = k + offs_k  # shape: [BLOCK_SIZE_K]
        group_idx = k_offset // GROUP_SIZE  # [BLOCK_SIZE_K]

        # Load scale and zero for each [N, G]
        scale = tl.load(bscales_ptr + stride_bn * offs_bn[None, :] + group_idx[:, None]).to(tl.float16)  # [BLOCK_SIZE_K, BLOCK_SIZE_N]
        zero = tl.load(bzeros_ptr + stride_bn * offs_bn[None, :] + group_idx[:, None]).to(tl.float16)   # same shape

        # Extract int4 values from uint8
        shift = (k_offset[:, None] % 2) * 4
        q = (b_q.to(tl.uint8) >> shift) & 0xF
        b_deq = (q.to(tl.float16) - zero) * scale

        accumulator += tl.dot(a, b_deq, out_dtype=tl.float16)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_int4_gemm(
    inp: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 64
) -> torch.Tensor:


    weight = weight.t().contiguous()  # [K/2, N]
    c_shape = inp.shape[:-1] + weight.shape[-1:]
    inp = inp.view(-1, inp.shape[-1]).contiguous()

    PAD_TO = 256
    if inp.shape[0] % PAD_TO != 0:
        c_crop = inp.shape[0]
        new_inp = inp.new_zeros(((inp.shape[0] + PAD_TO - 1) // PAD_TO * PAD_TO, inp.shape[1]))
        new_inp[:c_crop] = inp
        inp = new_inp
    else:
        c_crop = None

    M, K = inp.shape
    N = weight.shape[1]



    c = torch.empty((M, N), device=inp.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    int4_gemm_kernel[grid](
        inp, weight, c,
        scales, zeros,
        M, N, K,
        inp.stride(0), inp.stride(1),
        weight.stride(0), weight.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE=group_size,
    )

    return c[:c_crop] if c_crop is not None else c.view(c_shape)

class GPTQLinear(nn.Module):
    """
    4-bit quantized linear layer using Triton kernels
    """

    def __init__(self, in_features, out_features, bias=True, groupsize=64, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.device = device

        self.tile_cols = groupsize
        self.original_out_features = out_features

        # Quantized params (assigned later)
        self.register_buffer("packed_weight", None)
        self.register_buffer("scales", None)
        self.register_buffer("zeros", None)
        self.register_buffer("bias", None if not bias else torch.empty(out_features))

    @staticmethod
    def pack_weight(weight):
        rows, cols = weight.shape
        if cols % 2 != 0:
            weight = torch.nn.functional.pad(weight, (0, 1), value=0)
            cols += 1
        packed = (weight[:, 0::2] & 0xF) | ((weight[:, 1::2] & 0xF) << 4)
        return packed.contiguous()

    def get_weight(self, packed: torch.Tensor) -> torch.Tensor:
        """
        [rows, ceil(cols/2)] uint8 -> [rows, cols] uint8 in [0, 15]
        """
        rows, packed_cols = packed.shape
        qweight = torch.empty((rows, packed_cols * 2), dtype=torch.uint8, device=packed.device)
        qweight[:, 0::2] = packed & 0xF
        qweight[:, 1::2] = (packed >> 4) & 0xF
        return qweight[:, :self.in_features].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1, self.in_features)
        # Compute quantized matmul
        output = triton_int4_gemm(
            x_flat.float(),
            self.packed_weight,
            self.scales,
            self.zeros,
            group_size=self.groupsize,
        )

        if self.bias is not None:
            output += self.bias

        return output.view(*x.shape[:-1], self.out_features)


def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() // (1024 ** 2)
    return 0


def test_gptqlinear_vs_nnlinear(
    in_features=2048,
    out_features=4096,
    groupsize=64,
    wbits=4,
    device="cuda"
):
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Create single input vector ----
    x = torch.randn(in_features, device=device, dtype=torch.float16)
    linear = nn.Linear(in_features, out_features, bias=True, device=device, dtype=torch.float16).eval()

    weight = linear.weight.detach().to(device).float()
    bias = linear.bias.detach().to(device).float() if linear.bias is not None else None

    # --- Quantize using GPTQ ---
    gptq = GPTQ(wbits=wbits, groupsize=groupsize, device=device)
    qweight, qzeros, qscales = gptq.quantize(weight)
    packed_weight = GPTQLinear.pack_weight(qweight)

    gptqlinear = GPTQLinear(in_features, out_features, bias=True, groupsize=groupsize, device=device).to(device)
    gptqlinear.packed_weight = packed_weight
    gptqlinear.scales = qscales
    gptqlinear.zeros = qzeros
    gptqlinear.bias = bias if bias is not None else None
    gptqlinear.eval()

    print("\n== Latency ==")
    time_fp = triton.testing.do_bench(lambda: linear(x))
    time_q = triton.testing.do_bench(lambda: gptqlinear(x))
    print(f"nn.Linear (fp16): {time_fp:.3f} ms")
    print(f"GPTQLinear:       {time_q:.3f} ms")

    # print(torch.allclose(linear(x), gptqlinear(x), atol=1e-3))  # True / False
    a = linear(x)
    b = gptqlinear(x)
    abs_error = torch.abs(a - b)
    rel_error = abs_error / (torch.abs(b) + 1e-8)
    print("Mean abs error:", abs_error.mean().item())
    print("Max abs error:", abs_error.max().item())
    print("Mean rel error:", rel_error.mean().item())
    print("Max rel error:", rel_error.max().item())

if __name__ == "__main__":
    test_gptqlinear_vs_nnlinear()