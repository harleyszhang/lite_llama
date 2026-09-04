"""SmoothQuant W8A8 GEMM: int8 weights + dynamic per-token int8 activations.

``int8_quantize_per_token`` quantises the activations per token, then the GEMM
multiplies int8 x int8 with int32 accumulation and applies both scales in the
epilogue.

Usage:
    y = smoothquant_matmul(x, qweight, weight_scales)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..tile_policy import TileTier, resolve_tiles, tile_tier

# --------------------------------------------------------------------------- #
# Per-token activation quantisation
# --------------------------------------------------------------------------- #

#: Largest magnitude symmetric int8 stores.
_INT8_MAX = 127.0


@triton.jit
def _quantize_int8_per_token_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    K,
    stride_xm,
    stride_qm,
    QMAX: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Quantise one row of ``x`` to int8 with its own scale.

    Two passes over the row, both inside one program: the amax must be known
    before any element can be scaled, and keeping both passes here makes the
    scale a register rather than a round trip through HBM.
    """
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm

    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))

    # An all-zero row would divide by zero; 1.0 leaves it exactly zero instead.
    scale = tl.where(amax > 0.0, amax / QMAX, 1.0)
    tl.store(s_ptr + row, scale)

    q_row = q_ptr + row * stride_qm
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        # rint, not a plain .to(int8): torch's .round() — and the fused MoE
        # kernel's inline A-quantiser — round to nearest even, while .to
        # truncates toward zero, a different byte wherever the quotient's
        # fraction exceeds one half. The clamp is a no-op by construction; it
        # guards a non-finite input instead of producing an int8 overflow.
        r = tl.extra.cuda.libdevice.rint(x / scale)
        r = tl.minimum(tl.maximum(r, -QMAX), QMAX)
        tl.store(q_row + offs, r.to(tl.int8), mask=mask)


def int8_quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise activations to int8, one scale per row, in one launch.

    Serves the dense SmoothQuant path and the W8A8-int8 MoE path (whose grouped
    GEMM quantises inline only on the launch-bound small shapes), mirroring
    :func:`~rapid_llm.kernels.ops.quantization.fp8.fp8_quantize_per_token`. No
    host synchronisation, so a layer holding it on its critical path stays
    CUDA-graph capturable.

    Args:
        x: ``[..., K]`` fp16/bf16 activations. Leading dims are flattened to rows.

    Returns:
        ``(qx, scales)`` with ``qx`` int8 shaped like ``x`` and ``scales``
        ``[..., 1]`` fp32.
    """
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    m = flat.shape[0]

    qx = torch.empty_like(flat, dtype=torch.int8)
    scale = torch.empty(m, dtype=torch.float32, device=x.device)
    _quantize_int8_per_token_kernel[(m,)](
        flat,
        qx,
        scale,
        k,
        flat.stride(0),
        qx.stride(0),
        QMAX=_INT8_MAX,
        BLOCK_K=min(triton.next_power_of_2(k), 1024),
    )
    return qx.reshape(x.shape), scale.reshape(*x.shape[:-1], 1)


# --------------------------------------------------------------------------- #
# GEMM kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _smoothquant_matmul_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` tile of ``C = dequant(A @ B.T)``.

    A is ``[M, K]`` int8, B is ``[N, K]`` int8, C is ``[M, N]`` in ``a``'s
    dtype. ``a_scale`` is ``[M]`` per-token, ``b_scale`` is ``[N]`` per-channel.
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

    # int32 accumulation: int8 x int8 products fit exactly, and the scales
    # apply once in the epilogue instead of every k step.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Both scales are k-invariant, so the dequantisation rides out here.
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
def _launch_config(num_tokens: int, device_index: int | None) -> dict:
    """Tile shape for ``num_tokens`` rows on this device.

    Swept over the five dense projections of both test models on an H100
    (the ``bench_quant_gemm.py`` shape set): the old ``BLOCK_N`` 128/256
    starves the grid at every weight shape (24 column blocks at ``N=6144``
    against 132 SMs), so narrow N tiles carry every band, with wider tiles
    only where M itself supplies the parallelism. Pre-Hopper keeps the
    A10-era table the sm90 sweep replaced; ``n`` is not an input, because
    narrow/wide projection groups pick the same geomean-best tile per band.
    """
    if tile_tier(device_index) is TileTier.PRE_HOPPER:
        # A10-era table (sm86, measured): fatter tiles, fewer of them.
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
    if num_tokens <= 32:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 8,
            "num_stages": 4,
        }
    if num_tokens <= 128:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


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
        x: ``[..., K]`` fp16/bf16 activations. Leading dims are flattened.
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

    qactivations, activation_scales = int8_quantize_per_token(a)
    qactivations = qactivations.reshape(-1, k)
    activation_scales = activation_scales.reshape(-1)

    if weight_scales.dim() > 1:
        weight_scales = weight_scales.squeeze(-1)

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    # Autotune lookup (per-GPU entry) or the device-tiered heuristic.
    cfg = resolve_tiles(
        "smoothquant_matmul",
        m=m,
        n=n,
        k=k,
        dtype_label="int8",
        heuristic=lambda dev: _launch_config(m, dev),
        device_index=x.device.index,
    )
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)

    _smoothquant_matmul_kernel[grid](
        qactivations,
        qweight,
        out,
        activation_scales,
        weight_scales,
        bias,
        m,
        n,
        k,
        qactivations.stride(0),
        qactivations.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        **cfg,
    )
    return out.reshape(*leading, n)
