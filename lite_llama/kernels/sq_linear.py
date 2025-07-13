import torch
import triton
import triton.language as tl
from typing import Optional
import math


def pack_weight(weight):
    """Pack two INT4 values into one UINT8 value"""
    rows, cols = weight.shape
    if cols % 2 != 0:
        weight = torch.nn.functional.pad(weight, (0, 1), value=0)
        cols += 1

    # Ensure weights are in INT4 range [-8, 7]
    weight = torch.clamp(weight, min=-8, max=7)

    # Convert to unsigned representation [0, 15]
    weight_unsigned = weight + 8

    # Pack two INT4 values into one UINT8
    packed = (weight_unsigned[:, 0::2] & 0xF) | ((weight_unsigned[:, 1::2] & 0xF) << 4)
    return packed.contiguous().to(torch.uint8)


def unpack_weight(packed_weight, original_cols):
    """Unpack UINT8 values back to two INT4 values"""
    rows, packed_cols = packed_weight.shape
    unpacked = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed_weight.device)
    unpacked[:, 0::2] = packed_weight & 0xF
    unpacked[:, 1::2] = (packed_weight >> 4) & 0xF
    # Convert back to signed INT4 range [-8, 7]
    unpacked = unpacked.to(torch.int8) - 8
    return unpacked[:, :original_cols].contiguous()


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def smoothquant_int4_kernel(
        # Pointers to matrices
        x_ptr, w_ptr, bias_ptr, output_ptr,
        # Quantization parameters
        scale_ptr, zp_ptr, smooth_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        stride_sm, stride_sn,
        stride_zpm, stride_zpn,
        # Group size for quantization
        group_size,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized SmoothQuant linear kernel with INT4 weights and FP16 activations.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Current k indices
        k_indices = k * BLOCK_SIZE_K + offs_k

        # Load input activations
        x_ptrs = x_ptr + offs_am[:, None] * stride_xm + k_indices[None, :] * stride_xk
        x_mask = (offs_am[:, None] < M) & (k_indices[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float16)

        # Apply SmoothQuant inverse smoothing
        if smooth_ptr is not None:
            smooth_ptrs = smooth_ptr + k_indices
            smooth_mask = k_indices < K
            smooth_scale = tl.load(smooth_ptrs, mask=smooth_mask, other=1.0).to(tl.float16)
            # FIX: Ensure the division result stays as fp16
            x = (x / smooth_scale[None, :]).to(tl.float16)

        # Load packed weights (each packed value contains 2 INT4 weights)
        k_pack = k_indices // 2  # Packed dimension
        w_pack_ptrs = w_ptr + offs_bn[None, :] * stride_wn + k_pack[:, None] * stride_wk
        w_packed = tl.load(w_pack_ptrs, mask=(k_pack[:, None] < tl.cdiv(K, 2)) & (offs_bn[None, :] < N), other=0)

        # Unpack INT4 weights
        # For even k_indices, use lower 4 bits; for odd k_indices, use upper 4 bits
        even_mask = (k_indices % 2 == 0)
        w_vals = tl.where(even_mask[:, None], w_packed & 0xF, (w_packed >> 4) & 0xF)
        w_vals = w_vals.to(tl.int8) - 8  # Convert to signed INT4 range [-8, 7]

        # Load quantization parameters (group-wise)
        group_idx = k_indices // group_size
        scale_ptrs = scale_ptr + group_idx[:, None] * stride_sm + offs_bn[None, :] * stride_sn
        zp_ptrs = zp_ptr + group_idx[:, None] * stride_zpm + offs_bn[None, :] * stride_zpn

        scale_mask = (group_idx[:, None] < tl.cdiv(K, group_size)) & (offs_bn[None, :] < N)
        scale = tl.load(scale_ptrs, mask=scale_mask, other=1.0).to(tl.float16)
        zp = tl.load(zp_ptrs, mask=scale_mask, other=0.0).to(tl.float16)

        # Dequantize weights: (w_vals - zero_point) * scale
        w_vals = (w_vals.to(tl.float16) - zp) * scale

        # Matrix multiplication
        # Ensure we only process valid k values
        valid_mask = k_indices[:, None] < K
        # FIX: Use fp16 zero value to prevent dtype promotion
        w_vals = tl.where(valid_mask, w_vals, tl.zeros_like(w_vals))

        accumulator += tl.dot(x, w_vals, out_dtype=tl.float16)

    # Convert to output precision
    c = accumulator.to(tl.float16)

    # Add bias if provided
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_bn
        bias_mask = offs_bn < N
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float16)
        c += bias[None, :]

    # Store output
    output_ptrs = output_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    output_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(output_ptrs, c, mask=output_mask)


class SmoothQuantLinear(torch.nn.Module):
    """
    PyTorch module wrapper for SmoothQuant INT4 linear layer.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            group_size: int = 128,
            alpha: float = 0.5
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.alpha = alpha

        # Ensure in_features is compatible with packing
        self.packed_in_features = (in_features + 1) // 2  # For packing 2 INT4 into 1 UINT8

        # Initialize quantized weight storage (packed)
        self.register_buffer('packed_weight',
                             torch.zeros(out_features, self.packed_in_features, dtype=torch.uint8))

        # Quantization parameters (group-wise)
        self.num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer('weight_scale',
                             torch.ones(self.num_groups, out_features, dtype=torch.float16))
        self.register_buffer('weight_zero',
                             torch.zeros(self.num_groups, out_features, dtype=torch.float16))

        # SmoothQuant scaling factor
        self.register_buffer('smooth_scale',
                             torch.ones(in_features, dtype=torch.float16))

        # Bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer('bias', None)

    def quantize_weight(self, weight: torch.Tensor, act_scales: Optional[torch.Tensor] = None):
        """
        Quantize FP16/FP32 weight to INT4 with SmoothQuant.
        """
        assert weight.shape == (self.out_features, self.in_features)

        # Compute smoothing scale
        if act_scales is not None:
            # SmoothQuant formula: s_j = (max|X_j|)^α / (max|W_j|)^(1-α)
            weight_scales = weight.abs().max(dim=0)[0]
            # Avoid division by zero
            weight_scales = torch.clamp(weight_scales, min=1e-5)
            act_scales = torch.clamp(act_scales, min=1e-5)

            smooth_scale = (act_scales.pow(self.alpha) /
                            weight_scales.pow(1 - self.alpha))
            smooth_scale = torch.clamp(smooth_scale, min=0.01, max=100.0)
            self.smooth_scale.copy_(smooth_scale.to(torch.float16))

            # Apply smoothing to weights
            weight = weight * self.smooth_scale.unsqueeze(0)

        # Group-wise quantization
        weight_groups = []
        scales = []
        zeros = []

        for i in range(self.num_groups):
            start_idx = i * self.group_size
            end_idx = min((i + 1) * self.group_size, self.in_features)

            # Extract group
            w_group = weight[:, start_idx:end_idx]

            # Compute scale and zero point for this group
            w_max = w_group.max(dim=1, keepdim=True)[0]
            w_min = w_group.min(dim=1, keepdim=True)[0]

            # Symmetric quantization for INT4 [-8, 7]
            scale = (w_max - w_min) / 15.0
            scale = torch.clamp(scale, min=1e-5)
            zero = torch.round((w_max + w_min) / 2.0 / scale) * scale

            # Quantize to INT4 range
            w_quant = torch.round((w_group - zero) / scale).clamp(-8, 7)

            weight_groups.append(w_quant)
            scales.append(scale.squeeze(1))  # [out_features]
            zeros.append((zero / scale).squeeze(1))  # [out_features]

        # Concatenate groups back
        weight_quantized = torch.cat(weight_groups, dim=1).to(torch.int8)

        # Store quantization parameters
        self.weight_scale.copy_(torch.stack(scales, dim=0).to(torch.float16))
        self.weight_zero.copy_(torch.stack(zeros, dim=0).to(torch.float16))

        # Pack weights
        self.packed_weight.copy_(pack_weight(weight_quantized))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with INT4 quantized weights and FP16 activations.
        """
        assert x.shape[-1] == self.in_features
        assert x.dtype == torch.float16, "Input must be FP16"

        # Flatten input for matrix multiplication
        x_shape = x.shape
        x = x.view(-1, self.in_features)
        M, K = x.shape
        N = self.out_features

        # Allocate output
        output = torch.empty(M, N, dtype=torch.float16, device=x.device)

        # Launch kernel
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        smoothquant_int4_kernel[grid](
            x, self.packed_weight, self.bias, output,
            self.weight_scale, self.weight_zero, self.smooth_scale,
            M, N, K,
            x.stride(0), x.stride(1),
            self.packed_weight.stride(0), self.packed_weight.stride(1),
            output.stride(0), output.stride(1),
            self.weight_scale.stride(0), self.weight_scale.stride(1),
            self.weight_zero.stride(0), self.weight_zero.stride(1),
            self.group_size
        )

        return output.view(*x_shape[:-1], N)


import torch
import torch.nn as nn
import time
import gc
import numpy as np
from typing import Dict, List, Tuple


# Import the SmoothQuant implementation
# from smoothquant_int4 import SmoothQuantLinear

def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def format_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """Simple table formatter without external dependencies"""
    if not rows:
        return ""

    # Calculate column widths
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Create format string
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    separator = "-+-".join("-" * w for w in widths)

    # Build table
    result = []
    if title:
        total_width = sum(widths) + 3 * (len(widths) - 1)
        result.append(f"\n{title}")
        result.append("=" * max(len(title), total_width))

    result.append(fmt.format(*headers))
    result.append(separator)

    for row in rows:
        result.append(fmt.format(*[str(cell) for cell in row]))

    return "\n".join(result)


import torch
import torch.nn as nn
import time
import gc
import numpy as np
from typing import Dict, List, Tuple


# Import the SmoothQuant implementation
# from smoothquant_int4 import SmoothQuantLinear

def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def format_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """Simple table formatter without external dependencies"""
    if not rows:
        return ""

    # Calculate column widths
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Create format string
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    separator = "-+-".join("-" * w for w in widths)

    # Build table
    result = []
    if title:
        total_width = sum(widths) + 3 * (len(widths) - 1)
        result.append(f"\n{title}")
        result.append("=" * max(len(title), total_width))

    result.append(fmt.format(*headers))
    result.append(separator)

    for row in rows:
        result.append(fmt.format(*[str(cell) for cell in row]))

    return "\n".join(result)


