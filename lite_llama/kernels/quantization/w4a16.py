"""W4A16 GEMM: 4-bit weights (AWQ/GPTQ), fp16 activations, fp32 accumulation.

AWQ and GPTQ checkpoints pack 8 int4 values into each int32 word along the K
dimension, with one fp32 scale (and zero point) per group of ``group_size``
input channels. The kernel unpacks the int4 nibbles, applies the group-wise
dequantisation, and multiplies by the fp16 activation — all inside the GEMM
loop, so the weight never exists at fp16 in HBM.

v0.5 rewrite: uses ``tl.dot`` for tensor-core acceleration (SM80+ fp16 HMMA).
Each k-iteration processes one full group (group_size elements = BLOCK_K):
  1. Load [BLOCK_N, group_size//8] packed int32 words
  2. Unpack to [group_size, BLOCK_N] fp16 via bit shift
  3. Apply dequant: (nibble - zero) * scale
  4. ``tl.dot(a_tile, b_tile)`` accumulates on tensor cores

Packing order (AWQ/GPTQ standard):
    int32 word w contains values for K indices [8*i, 8*i+7]:
        nibble_j = (w >> (4*j)) & 0xF,  j = 0..7
    The dequantised value is: (nibble - zero) * scale.

Usage:
    y = w4a16_matmul(x, qweight, scales, zeros, group_size=128)
"""

from __future__ import annotations

import torch

from .._compat import tl, triton

_PACK_FACTOR = 8


# --------------------------------------------------------------------------- #
# GEMM kernel — per-group tl.dot (tensor core accelerated)
# --------------------------------------------------------------------------- #
@triton.jit
def _w4a16_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_ptr,
    zero_ptr,
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
    stride_sn,
    stride_sk,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One [BLOCK_M, BLOCK_N] tile of C = A @ dequant(B).T.

    Iterates K in steps of GROUP_SIZE (one quant group per iteration).
    Each step: unpack [BLOCK_N, GROUP_SIZE//8] int32 -> [GROUP_SIZE, BLOCK_N] fp16,
    dequant, then tl.dot accumulate.
    """
    WORDS_PER_GROUP: tl.constexpr = GROUP_SIZE // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # A base pointer for this tile's rows
    a_base = a_ptr + offs_m[:, None] * stride_am
    # B base pointer for this tile's columns
    b_base = b_ptr + offs_n[:, None] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Shift constants for unpacking 8 nibbles from one int32
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)  # [8]

    num_groups = K // GROUP_SIZE
    for g_idx in range(num_groups):
        k_start = g_idx * GROUP_SIZE

        # Load A slice: [BLOCK_M, GROUP_SIZE]
        offs_k = k_start + tl.arange(0, GROUP_SIZE)
        a_tile = tl.load(
            a_base + offs_k[None, :] * stride_ak,
            mask=offs_k[None, :] < K,
            other=0.0,
        )

        # Load B packed: [BLOCK_N, WORDS_PER_GROUP] int32
        offs_bk = (k_start // 8) + tl.arange(0, WORDS_PER_GROUP)
        b_packed = tl.load(
            b_base + offs_bk[None, :] * stride_bk,
            mask=offs_n[:, None] < N,
            other=0,
        )  # [BLOCK_N, WORDS_PER_GROUP]

        # Unpack: [BLOCK_N, WORDS_PER_GROUP, 1] >> [1, 1, 8] -> [BLOCK_N, WORDS_PER_GROUP, 8]
        b_expanded = (b_packed[:, :, None] >> shifts[None, None, :]) & 0xF
        # Reshape to [BLOCK_N, GROUP_SIZE] then cast to float
        b_flat = tl.reshape(b_expanded, (BLOCK_N, GROUP_SIZE)).to(tl.float32)

        # Load scale and zero for this group: [BLOCK_N]
        scale = tl.load(
            scale_ptr + offs_n * stride_sn + g_idx * stride_sk,
            mask=offs_n < N,
            other=1.0,
        )
        zero = tl.load(
            zero_ptr + offs_n * stride_sn + g_idx * stride_sk,
            mask=offs_n < N,
            other=0.0,
        )

        # Dequant: [BLOCK_N, GROUP_SIZE]
        b_dequant = (b_flat - zero[:, None]) * scale[:, None]

        # Transpose to [GROUP_SIZE, BLOCK_N] for tl.dot
        b_tile = tl.trans(b_dequant).to(tl.float16)  # [GROUP_SIZE, BLOCK_N]

        # Accumulate: [BLOCK_M, GROUP_SIZE] @ [GROUP_SIZE, BLOCK_N]
        accumulator += tl.dot(a_tile, b_tile)

    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]

    # Store output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def w4a16_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequant(qweight).T (+ bias)`` with int4 weights unpacked in-kernel.

    Uses tl.dot for tensor-core acceleration. Each GEMM iteration processes
    one full quantisation group (group_size elements).

    Args:
        x: ``[..., K]`` fp16 activations. Leading dims are flattened.
        qweight: ``[N, K//8]`` packed int32 weights (8 int4 values per word).
        scales: ``[N, K//group_size]`` fp32 dequantisation scales.
        zeros: ``[N, K//group_size]`` fp32 zero points.
        group_size: Number of input channels per quantisation group.
        bias: Optional ``[N]`` bias, added in fp32 before the output cast.

    Returns:
        ``[..., N]`` in ``x``'s dtype.
    """
    if x.dtype != torch.float16:
        raise ValueError(f"w4a16 activations must be fp16, got {x.dtype}")
    if qweight.dtype != torch.int32:
        raise ValueError(f"qweight must be int32 (packed int4), got {qweight.dtype}")

    n, k_packed = qweight.shape
    k = k_packed * _PACK_FACTOR
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} cols but weight expects {k}")
    if k % group_size != 0:
        raise ValueError(f"K ({k}) must be a multiple of group_size ({group_size})")

    leading = x.shape[:-1]
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    # Autotune lookup or heuristic fallback
    from lite_llama.kernels.autotune import get_best_config

    config = get_best_config("w4a16_matmul", m=m, n=n, k=k, dtype="int4")
    if config is None:
        if m <= 32:
            config = {"BLOCK_M": 16, "BLOCK_N": 64, "GROUP_M": 1, "num_warps": 4, "num_stages": 2}
        elif m <= 128:
            config = {"BLOCK_M": 32, "BLOCK_N": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 2}
        else:
            config = {"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 2}

    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    _w4a16_matmul_kernel[grid](
        a,
        qweight,
        out,
        scales,
        zeros,
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
        scales.stride(0),
        scales.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=config["GROUP_M"],
        HAS_BIAS=bias is not None,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out.reshape(*leading, n)
