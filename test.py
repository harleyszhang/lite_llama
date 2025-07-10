import torch
import triton
import triton.language as tl
import torch.nn as nn
from typing import Optional


@triton.jit
def _int4_linear_kernel(
    input_ptr,        # [M, K]
    qweight_ptr,      # [N, K//2]
    scales_ptr,       # [N, K//groupsize]
    zeros_ptr,        # [N, K//groupsize]
    output_ptr,       # [M, N]
    bias_ptr,         # [N] or dummy
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    groupsize: tl.constexpr,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
    stride_zn, stride_zg,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Input block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    input_block = tl.load(input_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    # ---- Load and unpack packed int4 weights ----
    packed_k = offs_k // 2  # [BLOCK_SIZE_K] → K//2 indices
    is_high = offs_k % 2    # [BLOCK_SIZE_K]

    weight_ptrs = qweight_ptr + offs_n[:, None] * stride_wn + packed_k[None, :] * stride_wk
    packed_vals = tl.load(weight_ptrs, mask=mask_n[:, None] & (packed_k[None, :] < (K // 2)), other=0)

    low = packed_vals & 0xF
    high = (packed_vals >> 4) & 0xF
    unpacked = tl.where(is_high[None, :] == 1, high, low).to(tl.float32)  # [N, K]

    # ---- Dequantization ----
    group_id = (offs_k // groupsize)[None, :]  # [1, K]
    scale_ptrs = scales_ptr + offs_n[:, None] * stride_sn + group_id * stride_sg
    zero_ptrs = zeros_ptr + offs_n[:, None] * stride_zn + group_id * stride_zg

    scale_vals = tl.load(scale_ptrs, mask=mask_n[:, None], other=1.0)
    zero_vals = tl.load(zero_ptrs, mask=mask_n[:, None], other=0.0)

    dequant = (unpacked - zero_vals) * scale_vals  # [BLOCK_SIZE_N, BLOCK_SIZE_K]

    # ---- GEMM ----
    acc = tl.dot(input_block, tl.trans(dequant))  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # ---- Add bias if present ----
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias_vals = tl.load(bias_ptrs, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # ---- Write output ----
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])



class Int4Linear(nn.Module):
    """
    A linear layer that uses int4 quantized weights with Triton kernel.
    """

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                 bias: Optional[torch.Tensor] = None, groupsize: int = 128):
        super().__init__()

        # Validate inputs
        assert qweight.dtype == torch.uint8, "qweight must be uint8"
        assert scales.dtype in [torch.float16, torch.float32], "scales must be float16 or float32"
        assert zeros.dtype in [torch.float16, torch.float32], "zeros must be float16 or float32"

        self.out_features, packed_in_features = qweight.shape
        self.in_features = packed_in_features * 2  # Each uint8 contains 2 int4 values
        self.groupsize = groupsize

        # Register quantized parameters
        self.register_buffer('qweight', qweight)
        self.register_buffer('scales', scales.to(torch.float32))  # Always use fp32 for scales
        self.register_buffer('zeros', zeros.to(torch.float32))  # Always use fp32 for zeros

        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using int4 Triton kernel.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        # Reshape input to 2D
        input_shape = x.shape
        x_2d = x.view(-1, self.in_features)
        M, K = x_2d.shape
        N = self.out_features

        # Allocate output
        output = torch.empty((M, N), dtype=x.dtype, device=x.device)

        # Calculate grid dimensions
        BLOCK_SIZE_M = min(64, triton.next_power_of_2(M))
        BLOCK_SIZE_N = min(64, triton.next_power_of_2(N))
        BLOCK_SIZE_K = min(64, triton.next_power_of_2(K))

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N'])
        )

        # Launch kernel
        _int4_linear_kernel[grid](
            x_2d, self.qweight, self.scales, self.zeros, output,
            self.bias if self.bias is not None else x_2d,  # Dummy pointer if no bias
            M, N, K, self.groupsize,
            x_2d.stride(0), x_2d.stride(1),
            self.qweight.stride(0), self.qweight.stride(1),
            self.scales.stride(0), self.scales.stride(1),
            self.zeros.stride(0), self.zeros.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            HAS_BIAS=self.bias is not None,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output back to original shape
        return output.view(*input_shape[:-1], self.out_features)


def pack_int4_weights(qweight: torch.Tensor) -> torch.Tensor:
    """
    Pack int4 weights from [N, K] uint8 to [N, K//2] uint8.
    Each output uint8 contains two int4 values.
    """
    N, K = qweight.shape
    assert K % 2 == 0, "K must be even for packing"

    # Pack two int4 values into one uint8
    packed = torch.zeros((N, K // 2), dtype=torch.uint8, device=qweight.device)
    packed = (qweight[:, 0::2] & 0xF) | ((qweight[:, 1::2] & 0xF) << 4)

    return packed


def create_int4_linear_from_quantized(qweight: torch.Tensor, scales: torch.Tensor,
                                      zeros: torch.Tensor, bias: Optional[torch.Tensor] = None,
                                      groupsize: int = 128) -> Int4Linear:
    """
    Create an Int4Linear layer from quantized parameters.

    Args:
        qweight: Quantized weights [out_features, in_features] as uint8
        scales: Dequantization scales [out_features, num_groups]
        zeros: Dequantization zeros [out_features, num_groups]
        bias: Optional bias term [out_features]
        groupsize: Group size for quantization

    Returns:
        Int4Linear layer ready for inference
    """
    # Pack int4 weights if needed
    if qweight.shape[1] % 2 == 0:
        # Assume weights are not packed yet
        packed_qweight = pack_int4_weights(qweight)
    else:
        # Assume weights are already packed
        packed_qweight = qweight

    return Int4Linear(packed_qweight, scales, zeros, bias, groupsize)


# Example usage and testing
if __name__ == "__main__":
    def test_int4_linear():
        # Test parameters
        batch_size = 32
        in_features = 512
        out_features = 256
        groupsize = 128

        # Create random quantized weights
        qweight = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8, device='cuda')
        scales = torch.randn(out_features, in_features // groupsize, dtype=torch.float16, device='cuda')
        zeros = torch.randint(0, 16, (out_features, in_features // groupsize), dtype=torch.float16, device='cuda')
        bias = torch.randn(out_features, dtype=torch.float16, device='cuda')

        # Create Int4Linear layer
        int4_layer = create_int4_linear_from_quantized(qweight, scales, zeros, bias, groupsize)

        # Test forward pass
        x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')

        # Warm up
        for _ in range(10):
            _ = int4_layer(x)

        torch.cuda.synchronize()

        # Benchmark
        import time
        start_time = time.time()
        for _ in range(100):
            output = int4_layer(x)
        torch.cuda.synchronize()
        end_time = time.time()

        print(f"Int4Linear forward time: {(end_time - start_time) * 10:.2f} ms per call")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")

        # Compare with standard linear (using dequantized weights)
        from lite_llama.quantization.gptq import GPTQ
        gptq = GPTQ(wbits=4, groupsize=groupsize)
        dequant_weight = gptq.dequantize(qweight, zeros, scales)

        std_layer = nn.Linear(in_features, out_features, bias=True, device='cuda', dtype=torch.float16)
        std_layer.weight.data = dequant_weight.T.to(torch.float16)
        std_layer.bias.data = bias

        start_time = time.time()
        for _ in range(100):
            std_output = std_layer(x)
        torch.cuda.synchronize()
        end_time = time.time()

        print(f"Standard Linear forward time: {(end_time - start_time) * 10:.2f} ms per call")

        # Check numerical accuracy (should be close due to quantization)
        diff = torch.abs(output - std_output).max()
        print(f"Max difference between Int4Linear and Standard Linear: {diff:.6f}")


    test_int4_linear()