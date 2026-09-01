"""FP8 W8A8 GEMM: fp8-e4m3 weights + dynamic per-token fp8-e4m3 activations.

Both operands are quantised before the launch — activations per token,
weights per block — and the kernel multiplies in fp8 with fp32
accumulation, applying scales in the epilogue.

Usage:
    y = fp8_matmul(qx, x_scale, qweight, weight_scale_inv)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .w8a16 import FP8_E4M3_BIT_TRICK_SCALE, dequant_fp8e4m3

#: Exponent correction when *both* operands went through the e4m3 -> fp16 bit
#: trick: each is short a factor of 256, so the product is short 256**2.
_FP8_BIT_TRICK_SCALE_SQ = FP8_E4M3_BIT_TRICK_SCALE * FP8_E4M3_BIT_TRICK_SCALE


# --------------------------------------------------------------------------- #
# GEMM kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _fp8_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bsn,
    stride_bsk,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DEQUANT_SCALE: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` tile of ``C = (A @ B.T) * a_scale * b_scale``.

    A is ``[M, K]`` and B is ``[N, K]``, both ``uint8`` e4m3 bit patterns;
    C is ``[M, N]``. ``a_scale`` is ``[M]`` (per token), ``b_scale`` covers a
    ``GROUP_N x GROUP_K`` weight block. ``BLOCK_K`` divides ``GROUP_K`` so a
    k-tile never straddles two b-scale blocks.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Grouped pid ordering: consecutive programs walk down a column strip so
    # the A tiles they share stay resident in L2.
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
    # B tiles load in their natural [N, K] row-major layout and transpose
    # in-register: the fp8 MMA wants its second operand K-major.
    b_ptrs = b_ptr + offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk
    b_scale_ptrs = b_scale_ptr + (offs_bn // GROUP_N) * stride_bsn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        # NATIVE_FP8 is constexpr: Triton emits two specialised kernels, so
        # the fp16 widening below never reaches an sm89+ binary (and the fp8
        # bitcast never reaches an sm86 one).
        if NATIVE_FP8:
            a = a.to(tl.float8e4nv, bitcast=True)
            b = b.to(tl.float8e4nv, bitcast=True)
        else:
            a = dequant_fp8e4m3(a)
            b = dequant_fp8e4m3(b)
        b_scale = tl.load(b_scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_bsk)
        accumulator += tl.dot(a, tl.trans(b)) * b_scale[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_scale = tl.load(a_scale_ptr + offs_am)
    result = accumulator * (DEQUANT_SCALE * a_scale[:, None])
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
    """Tile shape for ``num_tokens`` rows of activations.

    Same shapes as w8a16: one byte per operand element, k-contiguous, so a
    128-wide k-tile is exactly one memory transaction per output channel.
    """
    if num_tokens <= 32:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 1,
            "num_warps": 8,
            "num_stages": 3,
        }
    if num_tokens <= 128:
        return {
            "BLOCK_M": 32,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 256,
        "BLOCK_K": 128,
        "GROUP_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fp8_matmul(
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    group_n: int,
    group_k: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``dequant(qx) @ dequant(qweight).T (+ bias)`` with both operands fp8.

    Args:
        qx: ``[..., K]`` ``uint8`` e4m3 bit patterns — activations already
            quantised per token (see ``params.quantize_fp8_per_token``).
            Leading dims are flattened.
        x_scale: ``[..., 1]`` fp32 per-token activation scales.
        qweight: ``[N, K]`` ``uint8`` e4m3 bit patterns. Last dim contiguous.
        weight_scale_inv: ``[ceil(N/group_n), ceil(K/group_k)]`` fp32 weight
            scales; entry ``[i, j]`` covers ``(i*group_n, j*group_k)``.
        group_n: Rows of one weight-scale block. ``1`` means per-output-channel.
        group_k: Columns of one weight-scale block. ``>= K`` means one per row.
        bias: Optional ``[N]`` bias, added in fp32 before the output cast.
        out_dtype: Output dtype; the rest of the network runs bf16 (or fp16
            for a checkpoint that declares it).

    Returns:
        ``[..., N]`` in ``out_dtype``.
    """
    if qx.dtype != torch.uint8 or qweight.dtype != torch.uint8:
        raise ValueError(
            f"fp8 operands must be uint8 e4m3 bit patterns, got {qx.dtype} / {qweight.dtype}"
        )
    if qweight.stride(-1) != 1:
        raise ValueError("qweight last dimension must be contiguous")

    n, k = qweight.shape
    if qx.shape[-1] != k:
        raise ValueError(f"qx has {qx.shape[-1]} cols but weight expects {k}")
    if group_k % 128 != 0 and group_k < k:
        raise ValueError(f"group_k ({group_k}) must be a multiple of 128 unless it covers K")

    leading = qx.shape[:-1]
    a = qx.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    a_scale = x_scale.reshape(-1).contiguous()

    out = torch.empty((m, n), dtype=out_dtype, device=qx.device)
    native = torch.cuda.get_device_capability(qx.device) >= (8, 9)

    cfg = _launch_config(m)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _fp8_matmul_kernel[grid](
        a,
        qweight,
        out,
        a_scale,
        weight_scale_inv,
        bias,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        out.stride(0),
        out.stride(1),
        weight_scale_inv.stride(0),
        weight_scale_inv.stride(1),
        GROUP_N=group_n,
        GROUP_K=min(group_k, k),
        NATIVE_FP8=native,
        HAS_BIAS=bias is not None,
        DEQUANT_SCALE=1.0 if native else _FP8_BIT_TRICK_SCALE_SQ,
        **cfg,
    )
    return out.reshape(*leading, n)
