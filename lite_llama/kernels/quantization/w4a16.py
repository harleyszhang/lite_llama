"""W4A16 GEMM: 4-bit weights (AWQ/GPTQ), fp16 activations, fp32 accumulation.

AWQ and GPTQ checkpoints pack 8 int4 values into each int32 word along the K
dimension, with one fp32 scale (and zero point) per group of ``group_size``
input channels. The kernel unpacks the int4 nibbles, applies the group-wise
dequantisation, and multiplies by the fp16 activation — all inside the GEMM
loop, so the weight never exists at fp16 in HBM.

Packing order (AWQ/GPTQ standard):
    int32 word w contains values for K indices [8*i, 8*i+7]:
        nibble_j = (w >> (4*j)) & 0xF,  j = 0..7
    The dequantised value is: (nibble - zero) * scale.

Usage:
    y = w4a16_matmul(x, qweight, scales, zeros, group_size=128)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# GEMM kernel - tiled version
# --------------------------------------------------------------------------- #
@triton.jit
def _w4a16_matmul_kernel(
    a_ptr, b_ptr, c_ptr, scale_ptr, zero_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sn, stride_sk,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` tile of ``C = A @ dequant(B).T``.

    BLOCK_K must be a multiple of 8 (packing factor) and GROUP_SIZE.
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

    # A pointers: [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B pointers: [BLOCK_N, BLOCK_K // 8] packed int32
    offs_bk = tl.arange(0, BLOCK_K // 8)
    b_ptrs = b_ptr + offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of K-groups per BLOCK_K tile
    groups_per_block = BLOCK_K // GROUP_SIZE

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K
        k_rem = K - k_start

        # Load A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)

        # Load B tile: [BLOCK_N, BLOCK_K // 8] packed int32
        b_packed = tl.load(b_ptrs, mask=offs_bk[None, :] < (k_rem + 7) // 8, other=0)

        # Process each group within this block
        for g in range(groups_per_block):
            k_group_start = k_start + g * GROUP_SIZE
            k_group_idx = k_group_start // GROUP_SIZE

            # Load scale and zero for this group: [BLOCK_N]
            scale = tl.load(scale_ptr + offs_bn * stride_sn + k_group_idx * stride_sk,
                           mask=offs_bn < N, other=1.0)
            zero = tl.load(zero_ptr + offs_bn * stride_sn + k_group_idx * stride_sk,
                          mask=offs_bn < N, other=0.0)

            # Unpack and dequantize GROUP_SIZE int4 values
            # Each group spans GROUP_SIZE // 8 packed words
            words_per_group = GROUP_SIZE // 8
            for w in range(words_per_group):
                # Load one packed word per N: [BLOCK_N]
                word_idx = g * words_per_group + w
                packed_word = tl.load(b_ptr + offs_bn * stride_bn + (k_block * (BLOCK_K // 8) + word_idx) * stride_bk,
                                     mask=offs_bn < N, other=0)

                # Unpack 8 int4 values
                for nibble in tl.static_range(8):
                    k_offset = w * 8 + nibble
                    k_idx = g * GROUP_SIZE + k_offset

                    # Extract nibble
                    int4_val = (packed_word >> (4 * nibble)) & 0xF

                    # Dequantize
                    dequant = (int4_val.to(tl.float32) - zero) * scale

                    # Load A column for this K index
                    a_col = tl.load(a_ptr + offs_am * stride_am + (k_start + k_idx) * stride_ak,
                                   mask=offs_am < M, other=0.0)

                    # Accumulate outer product
                    accumulator += a_col[:, None] * dequant[None, :]

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

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
def _launch_config(num_tokens: int, group_size: int) -> dict:
    """Tile shape for ``num_tokens`` rows of activations."""
    # BLOCK_K must be a multiple of both 8 and group_size
    block_k = max(8, group_size)
    if num_tokens <= 32:
        return {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": block_k, "GROUP_M": 1,
                "num_warps": 4, "num_stages": 2}
    if num_tokens <= 128:
        return {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": block_k, "GROUP_M": 8,
                "num_warps": 4, "num_stages": 2}
    return {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": block_k, "GROUP_M": 8,
            "num_warps": 4, "num_stages": 2}


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
    k = k_packed * 8  # 8 int4 values per int32 word
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

    cfg = _launch_config(m, group_size)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)

    _w4a16_matmul_kernel[grid](
        a, qweight, out, scales, zeros, bias,
        m, n, k,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size,
        HAS_BIAS=bias is not None,
        **cfg,
    )
    return out.reshape(*leading, n)
