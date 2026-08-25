"""SmoothQuant W8A8 GEMM: int8 weights + dynamic per-token int8 activations.

SmoothQuant migrates the quantization difficulty from activations to weights by
applying a mathematically equivalent transformation that smooths the activation
distribution. At inference time:

* **Weights** are pre-quantised to int8 with per-channel scales (static).
* **Activations** are quantised on-the-fly to int8 with per-token scales (dynamic).

The GEMM then runs as ``int8 @ int8 -> int32 -> fp16``, which is faster than
fp16 @ fp16 on Ampere+ GPUs that have int8 tensor cores.

Unlike w8a16 (which keeps activations at fp16), SmoothQuant quantises both
operands, so the kernel must:
1. Compute per-token activation scales on the fly.
2. Quantise activations to int8.
3. Perform int8 @ int8 matmul with int32 accumulation.
4. Dequantise the result using both weight and activation scales.

Usage:
    y = smoothquant_matmul(x, qweight, weight_scales)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Activation quantisation kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _quantize_activations_kernel(
    x_ptr, q_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_qm, stride_qk,
    BLOCK_K: tl.constexpr,
):
    """Quantise ``[M, K]`` fp16 activations to int8 with per-token scales.

    Each row gets its own scale: ``scale[m] = max(|x[m]|) / 127``.
    """
    pid_m = tl.program_id(0)

    # First pass: find the max abs value across the entire row
    abs_max = 0.0
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + pid_m * stride_xm + offs_k * stride_xk
        x = tl.load(x_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)
        abs_max = tl.maximum(abs_max, tl.max(tl.abs(x), axis=0))

    scale = abs_max / 127.0
    scale = tl.where(scale > 0, scale, 1.0)

    # Second pass: quantise and store
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + pid_m * stride_xm + offs_k * stride_xk
        x = tl.load(x_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)
        q = (x / scale).to(tl.int8)
        q_ptrs = q_ptr + pid_m * stride_qm + offs_k * stride_qk
        tl.store(q_ptrs, q, mask=offs_k < K)

    tl.store(scale_ptr + pid_m, scale)


# --------------------------------------------------------------------------- #
# GEMM kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _smoothquant_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` tile of ``C = dequant(A @ B.T)``.

    A is ``[M, K]`` int8, B is ``[N, K]`` int8, C is ``[M, N]`` fp16.
    ``a_scale`` is ``[M]`` per-token, ``b_scale`` is ``[N]`` per-channel.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk

    # int32 accumulation for int8 @ int8
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Dequantise: multiply by per-token and per-channel scales
    a_scale = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=1.0)
    b_scale = tl.load(b_scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)
    result = accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]

    if HAS_BIAS:
        result += tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result.to(c_ptr.dtype.element_ty), mask=c_mask)


# --------------------------------------------------------------------------- #
# Launch configuration
# --------------------------------------------------------------------------- #
def _launch_config(num_tokens: int) -> dict:
    """Tile shape for ``num_tokens`` rows of activations."""
    if num_tokens <= 32:
        return {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 1,
                "num_warps": 8, "num_stages": 3}
    if num_tokens <= 128:
        return {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,
                "num_warps": 4, "num_stages": 3}
    return {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8,
            "num_warps": 8, "num_stages": 3}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def smoothquant_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequant(qweight).T (+ bias)`` with dynamic per-token activation quantisation.

    Args:
        x: ``[..., K]`` fp16 activations. Leading dims are flattened.
        qweight: ``[N, K]`` int8 weights (per-channel quantised).
        weight_scales: ``[N]`` or ``[N, 1]`` fp32 per-channel dequantisation scales.
        bias: Optional ``[N]`` bias, added in fp32 before the output cast.

    Returns:
        ``[..., N]`` in ``x``'s dtype.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"smoothquant activations must be fp16 or bf16, got {x.dtype}")
    if qweight.dtype != torch.int8:
        raise ValueError(f"qweight must be int8, got {qweight.dtype}")

    n, k = qweight.shape
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} cols but weight expects {k}")

    leading = x.shape[:-1]
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]

    # Quantise activations on the fly
    qactivations = torch.empty((m, k), dtype=torch.int8, device=x.device)
    activation_scales = torch.empty((m,), dtype=torch.float32, device=x.device)

    block_k = min(triton.next_power_of_2(k), 1024)
    _quantize_activations_kernel[(m,)](
        a, qactivations, activation_scales,
        m, k,
        a.stride(0), a.stride(1),
        qactivations.stride(0), qactivations.stride(1),
        BLOCK_K=block_k,
        num_warps=4,
    )

    # Flatten weight scales to [N]
    if weight_scales.dim() > 1:
        weight_scales = weight_scales.squeeze(-1)

    # Run int8 @ int8 GEMM
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    cfg = _launch_config(m)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)

    _smoothquant_matmul_kernel[grid](
        qactivations, qweight, out,
        activation_scales, weight_scales, bias,
        m, n, k,
        qactivations.stride(0), qactivations.stride(1),
        qweight.stride(0), qweight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        **cfg,
    )
    return out.reshape(*leading, n)
