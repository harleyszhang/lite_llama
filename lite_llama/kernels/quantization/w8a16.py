"""W8A16 GEMM: 8-bit weights (fp8-e4m3 or int8), fp16 activations, fp32 accumulation.

The weight stays 8-bit into the multiply-accumulate loop and is widened to fp16
one tile at a time. Two formats share one kernel via the IS_FP8 constexpr:
fp8-e4m3 (Qwen/DeepSeek checkpoints, bit-surgery dequant) and symmetric int8
(per-channel or group-wise, produced at load time).

Usage:
    y = w8a16_matmul(x, qweight, scales, group_n=128, group_k=128)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Exponent correction of the e4m3 -> fp16 bit trick, see :func:`dequant_fp8e4m3`.
FP8_E4M3_BIT_TRICK_SCALE = 256.0

#: Block size of the fine-grained FP8 format used by Qwen FP8 checkpoints.
FP8_BLOCK = 128


# --------------------------------------------------------------------------- #
# fp8 e4m3 -> fp16 bit trick
# --------------------------------------------------------------------------- #
@triton.jit
def dequant_fp8e4m3(q):
    """Widen e4m3 bytes to fp16, up to a constant factor of 256.

    e4m3 is ``s.eeee.mmm`` (bias 7), fp16 is ``s.eeeee.mmmmmmmmmm`` (bias 15).
    Dropping the byte's fields into fp16 unchanged encodes a value 2**(15-7)=256
    times too small. The 256 is *not* applied here — callers fold it into the
    dequantisation scale they were going to multiply by anyway.

    Every one of the 254 finite e4m3 values round-trips exactly. The two NaN
    encodings come out as ``±480``; a NaN weight is a broken checkpoint.
    """
    bits = q.to(tl.uint16)
    bits = ((bits & 0x80) << 8) | ((bits & 0x7F) << 7)
    return bits.to(tl.float16, bitcast=True)


# --------------------------------------------------------------------------- #
# GEMM kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _w8a16_matmul_kernel(
    a_ptr, b_ptr, c_ptr, scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk,
    stride_cm, stride_cn, stride_sn, stride_sk,
    GROUP_N: tl.constexpr, GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP8: tl.constexpr, HAS_BIAS: tl.constexpr,
    DEQUANT_SCALE: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` tile of ``C = A @ dequant(B).T``.

    A is ``[M, K]`` fp16, B is ``[N, K]`` 8-bit, C is ``[M, N]``. ``BLOCK_K``
    divides ``GROUP_K`` so a k-tile never straddles two scale blocks.
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
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk
    scale_ptrs = scale_ptr + (offs_bn // GROUP_N) * stride_sn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        # IS_FP8 is constexpr: Triton emits two specialised kernels, no branch.
        if IS_FP8:
            b = dequant_fp8e4m3(b)
        else:
            b = b.to(tl.float16)
        # tl.dot needs both operands in the activation's dtype. Widening the
        # widened-from-fp16 weight to bf16 rounds at 2^-8 — an order below the
        # 2^-4 the 8-bit weight itself carries, so nothing measurable is lost.
        b = b.to(a.dtype)
        scale = tl.load(scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_sk)
        accumulator += tl.dot(a, b) * scale[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    accumulator *= DEQUANT_SCALE
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


# --------------------------------------------------------------------------- #
# Launch configuration
# --------------------------------------------------------------------------- #
def _launch_config(num_tokens: int) -> dict:
    """Tile shape for ``num_tokens`` rows of activations, measured on an A10.

    ``BLOCK_K`` is always 128: one byte per weight element, k-contiguous, so a
    128-wide k-tile is exactly one 128-byte memory transaction per output
    channel. Decode wants the narrowest ``BLOCK_M`` ``tl.dot`` accepts; prefill
    grows it towards a compute-shaped block.
    """
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
def w8a16_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_n: int,
    group_k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequant(qweight).T (+ bias)`` with the weight widened in-kernel.

    Args:
        x: ``[..., K]`` fp16 activations. Leading dims are flattened.
        qweight: ``[N, K]`` weights, ``uint8`` for fp8-e4m3 or ``int8`` for
            symmetric integer quantisation. Last dim must be contiguous.
        scales: ``[ceil(N/group_n), ceil(K/group_k)]`` fp32 dequantisation
            scales; ``scales[i, j]`` covers ``(i*group_n, j*group_k)``.
        group_n: Rows of one scale block. ``1`` means per-output-channel.
        group_k: Columns of one scale block. ``>= K`` means one scale per row.
        bias: Optional ``[N]`` bias, added in fp32 before the output cast.

    Returns:
        ``[..., N]`` in ``x``'s dtype.
    """
    is_fp8 = qweight.dtype == torch.uint8
    if not is_fp8 and qweight.dtype != torch.int8:
        raise ValueError(f"qweight must be uint8 (fp8) or int8, got {qweight.dtype}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"w8a16 activations must be fp16 or bf16, got {x.dtype}")
    if qweight.stride(-1) != 1:
        raise ValueError("qweight last dimension must be contiguous")

    n, k = qweight.shape
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} cols but weight expects {k}")
    if group_k % 128 != 0 and group_k < k:
        raise ValueError(f"group_k ({group_k}) must be a multiple of 128 unless it covers K")

    leading = x.shape[:-1]
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    cfg = _launch_config(m)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _w8a16_matmul_kernel[grid](
        a, qweight, out, scales, bias,
        m, n, k,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_N=group_n,
        GROUP_K=min(group_k, k),
        IS_FP8=is_fp8,
        HAS_BIAS=bias is not None,
        DEQUANT_SCALE=FP8_E4M3_BIT_TRICK_SCALE if is_fp8 else 1.0,
        **cfg,
    )
    return out.reshape(*leading, n)
