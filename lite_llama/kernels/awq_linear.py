import torch
import triton
import triton.language as tl
from typing import Optional
import psutil, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lite_llama.quantization.utils import pack_weight


@triton.jit
def awq_linear_kernel(
        input_ptr, qweight_ptr, qscales_ptr, qzeros_ptr, output_ptr, bias_ptr,
        M, N, K, group_size,
        stride_input_m, stride_input_k,
        stride_qweight_n, stride_qweight_k,
        stride_qscales_n, stride_qscales_g,
        stride_qzeros_n, stride_qzeros_g,
        stride_output_m, stride_output_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    """Ultra-simplified AWQ linear kernel"""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Only process one element per thread for maximum compatibility
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_idx < M
    n_mask = n_idx < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Simple loop over all K without chunking
    for k in range(K):
        # Load input values [BLOCK_M]
        input_ptrs = input_ptr + m_idx * stride_input_m + k * stride_input_k
        input_vals = tl.load(input_ptrs, mask=m_mask, other=0.0)

        # Load and process weights [BLOCK_N]
        packed_k = k // 2
        is_high = k % 2

        qweight_ptrs = qweight_ptr + n_idx * stride_qweight_n + packed_k * stride_qweight_k
        packed_weights = tl.load(qweight_ptrs, mask=n_mask, other=0)

        # Unpack 4-bit weights
        if is_high == 1:
            weights_int4 = (packed_weights >> 4) & 0xF
        else:
            weights_int4 = packed_weights & 0xF

        # Get quantization parameters
        group_idx = k // GROUP_SIZE

        qscales_ptrs = qscales_ptr + n_idx * stride_qscales_n + group_idx * stride_qscales_g
        qzeros_ptrs = qzeros_ptr + n_idx * stride_qzeros_n + group_idx * stride_qzeros_g

        scales = tl.load(qscales_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(qzeros_ptrs, mask=n_mask, other=0.0)

        # Dequantize
        weights_fp = (weights_int4.to(tl.float32) - zeros) * scales

        # Accumulate outer product: input[m] * weight[n] -> acc[m, n]
        acc += input_vals[:, None] * weights_fp[None, :]

    # Add bias
    if HAS_BIAS:
        bias_ptrs = bias_ptr + n_idx
        bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    # Store result
    output_ptrs = output_ptr + m_idx[:, None] * stride_output_m + n_idx[None, :] * stride_output_n
    tl.store(output_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def awq_linear_triton(
        input: torch.Tensor,
        qweight: torch.Tensor,
        qscales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128
) -> torch.Tensor:
    """
    AWQ quantized linear layer using Triton

    Args:
        input: Input tensor [*, in_features] in fp16
        qweight: Packed quantized weights [out_features, in_features//2] in int8
        qscales: Quantization scales [out_features, in_features//group_size]
        qzeros: Quantization zeros [out_features, in_features//group_size]
        bias: Optional bias [out_features]
        group_size: Group size for quantization

    Returns:
        Output tensor [*, out_features] in fp16
    """

    # Reshape input to 2D
    input_shape = input.shape
    input_2d = input.view(-1, input.shape[-1])
    M, K = input_2d.shape
    N = qweight.shape[0]

    # Ensure input is fp16
    if input_2d.dtype != torch.float16:
        input_2d = input_2d.to(torch.float16)

    # Create output tensor
    output = torch.empty((M, N), dtype=torch.float16, device=input.device)

    # Block sizes - smaller for better compatibility
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Grid configuration
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    # Launch kernel
    awq_linear_kernel[grid](
        input_2d, qweight, qscales, qzeros, output, bias,
        M, N, K, group_size,
        # Input strides
        input_2d.stride(0), input_2d.stride(1),
        # QWeight strides
        qweight.stride(0), qweight.stride(1),
        # QScales strides
        qscales.stride(0), qscales.stride(1),
        # QZeros strides
        qzeros.stride(0), qzeros.stride(1),
        # Output strides
        output.stride(0), output.stride(1),
        # Block sizes
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE=group_size,
        HAS_BIAS=bias is not None,
    )

    # Reshape output back to original shape
    output_shape = input_shape[:-1] + (N,)
    return output.view(output_shape)


class AWQLinear(torch.nn.Module):
    """
    AWQ Quantized Linear Layer using Triton
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            group_size: int = 128,
            wbits: int = 4,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.wbits = wbits

        # Calculate number of groups
        self.num_groups = (in_features + group_size - 1) // group_size

        # Register quantized weight parameters
        # Packed weights: 2 int4 values per int8
        packed_width = (in_features + 1) // 2
        self.register_buffer('qweight', torch.zeros((out_features, packed_width), dtype=torch.uint8))
        self.register_buffer('qscales', torch.zeros((out_features, self.num_groups), dtype=torch.float16))
        self.register_buffer('qzeros', torch.zeros((out_features, self.num_groups), dtype=torch.float16))

        if bias:
            self.register_parameter('bias', torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16)))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Triton kernel"""
        return awq_linear_triton(
            input=x,
            qweight=self.qweight,
            qscales=self.qscales,
            qzeros=self.qzeros,
            bias=self.bias,
            group_size=self.group_size
        )

    @classmethod
    def from_float(
            cls,
            linear: torch.nn.Linear,
            qweight: torch.Tensor,
            qscales: torch.Tensor,
            qzeros: torch.Tensor,
            group_size: int = 128,
    ):
        """
        Create AWQLinear from a regular Linear layer and quantization parameters

        Args:
            linear: Original torch.nn.Linear layer
            qweight: Packed quantized weights
            qscales: Quantization scales
            qzeros: Quantization zeros
            group_size: Group size used for quantization
        """

        awq_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
        )

        # Copy quantized parameters
        with torch.no_grad():
            awq_linear.qweight.copy_(qweight)
            awq_linear.qscales.copy_(qscales)
            awq_linear.qzeros.copy_(qzeros)

            if linear.bias is not None:
                awq_linear.bias.copy_(linear.bias.to(torch.float16))

        return awq_linear


def demo_awq_triton():
    """Demo function for AWQ Triton linear layer"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not available, demo will run on CPU (Triton requires CUDA)")
        return

    # Create test data
    batch_size, seq_len = 2, 128
    in_features, out_features = 768, 768
    group_size = 128

    # Create original linear layer
    linear = torch.nn.Linear(in_features, out_features, bias=True).to(device)


    # Mock quantized weights (in practice, these come from AWQ.quantize())
    weight_int4 = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8, device=device)
    qweight = pack_weight(weight_int4)

    num_groups = (in_features + group_size - 1) // group_size
    qscales = torch.randn((out_features, num_groups), dtype=torch.float16, device=device).abs() + 0.1
    qzeros = torch.randint(0, 16, (out_features, num_groups), dtype=torch.float16, device=device)

    # Create AWQ linear layer
    awq_linear = AWQLinear.from_float(linear, qweight, qscales, qzeros, group_size)
    awq_linear = awq_linear.to(device)

    # Test input
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.float16, device=device)

    print(f"Input shape: {x.shape}")
    print(f"QWeight shape: {qweight.shape}")
    print(f"QScales shape: {qscales.shape}")
    print(f"QZeros shape: {qzeros.shape}")

    # Forward pass
    with torch.no_grad():
        output = awq_linear(x)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print("AWQ Triton linear demo completed successfully!")


if __name__ == "__main__":
    demo_awq_triton()