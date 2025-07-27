import triton
import triton.language as tl
import torch
import torch.nn as nn
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lite_llama.quantization.utils import pack_weight


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
        k_offset = k + offs_k
        group_idx = k_offset // GROUP_SIZE

        # Load scale and zero for each [N, G]
        scale = tl.load(bscales_ptr + stride_bn * offs_bn[None, :] + group_idx[:, None]).to(tl.float16)
        zero = tl.load(bzeros_ptr + stride_bn * offs_bn[None, :] + group_idx[:, None]).to(tl.float16)

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
    weight = weight.t().contiguous()
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
    4-bit quantized linear layer using Triton kernels (修复版本)
    """

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16, bits=4, groupsize=64, device="cuda",
                 tile_cols=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.device = device
        self.dtype = dtype
        self.bits = bits
        self.tile_cols = groupsize
        self.original_out_features = out_features

        # 计算量化参数的形状
        self.num_groups = (in_features + groupsize - 1) // groupsize
        packed_width = (in_features + 1) // 2  # 2个int4打包成1个uint8

        # 注册量化参数缓冲区 - 修复：确保所有缓冲区都被正确初始化
        self.register_buffer("packed_weight", torch.zeros(out_features, packed_width, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(out_features, self.num_groups, dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros(out_features, self.num_groups, dtype=torch.float16))

        # Bias参数
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_features, dtype=torch.float16)))
        else:
            self.register_buffer("bias", None)


    def set_quantized_params(self, packed_weight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
        """设置量化参数 - 新增方法"""
        with torch.no_grad():
            if packed_weight is not None:
                self.packed_weight.copy_(packed_weight)
            if scales is not None:
                self.scales.copy_(scales)
            if zeros is not None:
                self.zeros.copy_(zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1, self.in_features)

        # 确保所有参数都已正确设置
        if self.packed_weight is None or self.scales is None or self.zeros is None:
            raise RuntimeError("Quantized parameters not properly initialized. Call set_quantized_params() first.")

        # 使用Triton优化的int4 GEMM
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

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, groupsize: int = 128, bits: int = 4):
        """从标准线性层创建GPTQ层"""
        gptq_layer = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
            groupsize=groupsize,
            bits=bits,
            device=linear_layer.weight.device
        )

        # 复制bias
        if linear_layer.bias is not None:
            gptq_layer.bias.data.copy_(linear_layer.bias.data)

        return gptq_layer

    def __repr__(self):
        return f"GPTQLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, bits={self.bits}, groupsize={self.groupsize})"


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

    # --- 创建GPTQ层并模拟量化参数 ---
    gptqlinear = GPTQLinear(in_features, out_features, bias=True, groupsize=groupsize, device=device).to(device)

    # 模拟量化参数（实际使用中这些来自GPTQ量化算法）
    num_groups = (in_features + groupsize - 1) // groupsize
    packed_width = (in_features + 1) // 2

    # 创建模拟的量化参数
    mock_packed_weight = torch.randint(0, 255, (out_features, packed_width), dtype=torch.uint8, device=device)
    mock_scales = torch.randn(out_features, num_groups, dtype=torch.float16, device=device).abs() + 0.1
    mock_zeros = torch.randint(0, 15, (out_features, num_groups), dtype=torch.float16, device=device)

    # 设置量化参数
    gptqlinear.set_quantized_params(mock_packed_weight, mock_scales, mock_zeros)

    if bias is not None:
        gptqlinear.bias.data.copy_(bias)

    gptqlinear.eval()

    print("\n== Latency ==")
    time_fp = triton.testing.do_bench(lambda: linear(x))
    time_q = triton.testing.do_bench(lambda: gptqlinear(x))
    print(f"nn.Linear (fp16): {time_fp:.3f} ms")
    print(f"GPTQLinear:       {time_q:.3f} ms")

    # 测试输出形状
    a = linear(x)
    b = gptqlinear(x)
    print(f"Output shapes - Linear: {a.shape}, GPTQ: {b.shape}")
    print("GPTQ layer test completed successfully!")


if __name__ == "__main__":
    test_gptqlinear_vs_nnlinear()