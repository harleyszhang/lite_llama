"""W4A16 GEMM: 4-bit weights (AWQ/GPTQ), fp16 activations, fp32 accumulation.

AWQ and GPTQ checkpoints pack 8 int4 values into each int32 word along the K
dimension, with one fp32 scale (and zero point) per group of ``group_size``
input channels. The kernel unpacks the int4 nibbles, applies the group-wise
dequantisation, and multiplies by the fp16 activation — all inside the GEMM
loop, so the weight never exists at fp16 in HBM.

v0.6 rewrite, measured on an H100 (N=K=4096, fp16): ``BLOCK_K`` is decoupled
from ``group_size`` and the unpack follows the :mod:`.nvfp4` idiom — a
coalesced [BLOCK_N, BLOCK_K//8] word load, a 3-D shift/reshape to nibbles in
registers, fp32 dequant, ``tl.trans`` into ``tl.dot``. The previous per-group
loop (``BLOCK_K == group_size == 128``) fetched 64 bytes per output channel
per iteration — half a 128-byte transaction; ``BLOCK_K = 256`` fills it and
is worth 1.4-1.6x at decode widths (m=1: 33.9 -> 23.2 us, on par with cuBLAS
fp16's 23.1; m=64: 49.9 -> 31.1), parity at prefill. ``BLOCK_K = 512`` loses
to register pressure, so the tune space stops at 256. A fp16 magic-number
dequant variant measured 9% faster at m=64 but 12% slower at m=1, so there is
one code path.

Packing order (AWQ/GPTQ standard):
    int32 word w contains values for K indices [8*i, 8*i+7]:
        nibble_j = (w >> (4*j)) & 0xF,  j = 0..7
    The dequantised value is: (nibble - zero) * scale.

Usage:
    y = w4a16_matmul(x, qweight, scales, zeros)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_PACK_FACTOR = 8


# --------------------------------------------------------------------------- #
# GEMM kernel — coalesced word loads, in-register unpack, tl.dot
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
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One [BLOCK_M, BLOCK_N] tile of C = A @ dequant(B).T.

    ``BLOCK_K`` is a multiple of ``GROUP_SIZE`` (clamped by the launcher), so
    one iteration covers ``BLOCK_K // GROUP_SIZE`` quantisation groups: the
    packed tile loads as [BLOCK_N, BLOCK_K//8] int32 words — a coalesced
    row-major read — and the scales as [BLOCK_N, BLOCK_K//GROUP_SIZE]. Both
    are indexed [n, k], so the operand is transposed in registers for the dot.
    K divides ``BLOCK_K`` evenly (the launcher guarantees it), so the loop
    needs no k mask.
    """
    WORDS: tl.constexpr = BLOCK_K // 8
    SCALES: tl.constexpr = BLOCK_K // GROUP_SIZE

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
    offs_k = tl.arange(0, BLOCK_K)
    offs_word = tl.arange(0, WORDS)
    offs_scale = tl.arange(0, SCALES)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_word[None, :] * stride_bk
    s_ptrs = scale_ptr + offs_n[:, None] * stride_sn + offs_scale[None, :] * stride_sk
    z_ptrs = zero_ptr + offs_n[:, None] * stride_sn + offs_scale[None, :] * stride_sk

    # Shift constants for unpacking 8 nibbles from one int32
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)  # [8]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        a_tile = tl.load(a_ptrs)
        b_packed = tl.load(b_ptrs)  # [BLOCK_N, WORDS] int32
        scale = tl.load(s_ptrs)  # [BLOCK_N, SCALES] fp32
        zero = tl.load(z_ptrs)  # [BLOCK_N, SCALES] fp32

        # [BLOCK_N, WORDS, 1] >> [1, 1, 8] -> [BLOCK_N, WORDS, 8]; the reshape
        # lands in k order because nibble j of word w covers k = w*8 + j.
        b_expanded = (b_packed[:, :, None] >> shifts[None, None, :]) & 0xF
        b_flat = tl.reshape(b_expanded, (BLOCK_N, BLOCK_K)).to(tl.float32)
        scale_b = tl.reshape(
            tl.broadcast_to(scale[:, :, None], (BLOCK_N, SCALES, GROUP_SIZE)),
            (BLOCK_N, BLOCK_K),
        )
        zero_b = tl.reshape(
            tl.broadcast_to(zero[:, :, None], (BLOCK_N, SCALES, GROUP_SIZE)),
            (BLOCK_N, BLOCK_K),
        )

        # Dequant in fp32, narrowed to the activation's dtype for the dot. The
        # (nibble - zero) factor is exact — both are integers in [0, 15] — so
        # only the scale's low mantissa bits round, as they did before.
        b_dequant = (b_flat - zero_b) * scale_b
        b_tile = tl.trans(b_dequant).to(a_tile.dtype)  # [BLOCK_K, BLOCK_N]

        accumulator += tl.dot(a_tile, b_tile)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += WORDS * stride_bk
        s_ptrs += SCALES * stride_sk
        z_ptrs += SCALES * stride_sk

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
def launch_config(m: int) -> dict[str, int]:
    """Tile config for ``m`` rows when the autotune store has no entry.

    Measured on an H100. ``BLOCK_N=32`` is what the unpacked ``[BLOCK_N, BLOCK_K]``
    operand's register budget wants, and ``BLOCK_K=256`` fills a 128-byte
    transaction per output channel where ``group_size=128`` fills half of one.
    The switches at 32 and 128 rows are both ``bucket_m`` boundaries, so one store
    entry never stands in for two heuristic choices.

    Public because ``benchmarks/kernels/bench_quant_gemm.py --tune`` needs the
    config it is trying to beat. It used to keep its own copy and guard it by
    timing both, which cannot work: at one row the same config measures 22-30 us
    run to run, so the guard reported noise as a divergence.
    """
    if m <= 32:
        return {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 8, "num_warps": 4, "num_stages": 4}
    if m <= 128:
        return {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 8, "num_warps": 4, "num_stages": 4}
    return {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 8, "num_warps": 4, "num_stages": 3}


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

    Uses tl.dot for tensor-core acceleration. The k-tile (``BLOCK_K``) is a
    multiple of ``group_size`` chosen per M bucket — from the autotune store
    when an entry exists, else the measured heuristic below.

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
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"w4a16 activations must be fp16 or bf16, got {x.dtype}")
    if qweight.dtype != torch.int32:
        raise ValueError(f"qweight must be int32 (packed int4), got {qweight.dtype}")

    n, k_packed = qweight.shape
    k = k_packed * _PACK_FACTOR
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} cols but weight expects {k}")
    if k % group_size != 0:
        raise ValueError(f"K ({k}) must be a multiple of group_size ({group_size})")
    if group_size & (group_size - 1) != 0 or group_size < 16:
        raise ValueError(
            f"group_size must be a power of two >= 16 (tl.arange / tl.dot), got {group_size}"
        )

    leading = x.shape[:-1]
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    # Autotune lookup or heuristic fallback
    from lite_llama.kernels.dispatcher.autotune import get_best_config

    config = get_best_config("w4a16_matmul", m=m, n=n, k=k, dtype="int4")
    if config is None:
        config = launch_config(m)

    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    # Store entries written before BLOCK_K existed carry no key; fall back to
    # the measured 256. Whatever the source, the k-tile must cover whole
    # quantisation groups and divide K evenly — halve until it does. The
    # group_size fallback always fits because k % group_size == 0 above.
    block_k = config.get("BLOCK_K", 256)
    while block_k > group_size and (block_k % group_size != 0 or k % block_k != 0):
        block_k //= 2
    if block_k % group_size != 0 or k % block_k != 0:
        block_k = group_size
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
        BLOCK_K=block_k,
        GROUP_M=config["GROUP_M"],
        HAS_BIAS=bias is not None,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out.reshape(*leading, n)
