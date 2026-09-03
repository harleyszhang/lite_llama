"""W8A16 GEMM: 8-bit weights (fp8-e4m3 or int8), fp16 activations.

The kernel dequantises each weight tile on the fly (fp8 via one hardware
``cvt`` on sm89+, falling back to a bit-trick below that; int8 by scale
multiply) and accumulates in fp32, so activations never leave fp16.

Usage:
    y = w8a16_matmul(x, qweight, scales)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..tile_policy import TileTier, has_native_fp8, resolve_tiles, tile_tier

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
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP8: tl.constexpr,
    HAS_ZEROS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DEQUANT_SCALE: tl.constexpr,
    FP8_CVT: tl.constexpr,
    SINGLE_SCALE: tl.constexpr,
    EPILOGUE_SCALE: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` tile of ``C = A @ dequant(B).T``.

    A is ``[M, K]`` fp16, B is ``[N, K]`` 8-bit, C is ``[M, N]``. ``BLOCK_K``
    divides ``GROUP_K`` so a k-tile never straddles two scale blocks. With
    ``HAS_ZEROS`` (asymmetric int8, GPTQ ``bits=8``) the group zero point is
    subtracted in fp32 before the dot — the operand stays an exact integer,
    exactly as the fused MoE kernel's int4 and int8-asym branches do it.
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
    # Zeros share the scales' shape and strides; the pointer may be null —
    # the constexpr guard keeps the address arithmetic off the symmetric
    # kernel's trace entirely, so ``None`` never reaches ``+``.
    if HAS_ZEROS:
        zero_ptrs = zero_ptr + (offs_bn // GROUP_N) * stride_sn
        if SINGLE_SCALE:
            zero = tl.load(zero_ptrs)
    # One scale per weight row (per-channel / per-row quantisation): its k
    # address never moves, so load it before the loop and keep the loop body
    # down to weight tiles and dots -- a per-iteration [BLOCK_N] scale load
    # and its address arithmetic otherwise ride every k step.
    if SINGLE_SCALE:
        scale = tl.load(scale_ptrs)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        # IS_FP8 is constexpr, so Triton emits two specialised kernels with no
        # branch. Either way the tile lands in the activation's dtype: e4m3 is
        # exact in fp16 (and the fp16 -> bf16 hop rounds at 2^-8, an order below
        # the 8-bit weight's own 2^-4), int8 is exact in both. sm89+ widens
        # e4m3 with one hardware cvt per element; the bit trick needs five
        # integer instructions plus the 256x correction folded into
        # DEQUANT_SCALE by the launcher.
        if IS_FP8:
            if FP8_CVT:
                b = b.to(tl.float8e4nv, bitcast=True).to(a.dtype)
            else:
                b = dequant_fp8e4m3(b).to(a.dtype)
        elif HAS_ZEROS:
            # Differences of int8 integers are integers within [-255, 255],
            # exact in fp16 and bf16 alike, so the widened tile loses nothing.
            if not SINGLE_SCALE:
                zero = tl.load(zero_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_sk)
            b = (b.to(tl.float32) - zero[None, :]).to(a.dtype)
        else:
            b = b.to(a.dtype)
        if SINGLE_SCALE:
            if EPILOGUE_SCALE:
                # In-place accumulation (``acc=``): the per-row scale distributes
                # over the sum, so one epilogue multiply is the same arithmetic
                # and unlocks the 128x128 tile, where the per-step multiply
                # otherwise wrecks the pipeline (2-3x). Decode keeps the in-loop
                # form instead -- ``acc=`` costs int8 weights 3-11% there.
                accumulator = tl.dot(a, b, acc=accumulator)
            else:
                accumulator += tl.dot(a, b) * scale[None, :]
        else:
            scale = tl.load(scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_sk)
            accumulator += tl.dot(a, b) * scale[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    accumulator *= DEQUANT_SCALE
    if SINGLE_SCALE and EPILOGUE_SCALE:
        accumulator *= scale[None, :]
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
def _launch_config(
    num_tokens: int, single_scale: bool, is_fp8: bool, device_index: int | None
) -> dict:
    """Tile shape for ``num_tokens`` rows on this device and weight dtype.

    Each input besides the row count is forked on measured evidence, and the
    one dimension the data rejects is noted too:

    - **Device**: the sm90 tables were swept on an H100 (132 SMs, where
      narrow N tiles fill the grid); pre-Hopper keeps the A10-era table the
      sm90 sweep replaced, measured on that hardware. ``sm_version`` gates.
    - **Weight dtype**: int8 with per-row scales (``single_scale`` + int8,
      the runtime ``int8`` scheme) takes its own table -- the fp8 numbers do
      *not* carry over. Re-swept on the same five projections with the same
      EPILOGUE_SCALE bands, the fp8 tile costs int8 up to 57% at decode (geo
      1.30): its ``BLOCK_N=128``, kept for the N=19456 gate_up projection, is
      the wrong width for int8, which wants 64 everywhere and ``BLOCK_M=256``
      at m=2048 (the 128-row tile is up to 31% slower there). int8
      *block-scale* (GPTQ ``bits=8``) keeps the fp8 table: it shares the
      in-loop path and has no sweep of its own.
    - **Shape**: no fork. Split narrow/wide and re-analysed per band, the two
      groups pick the same geomean-best tile everywhere, so ``n`` is
      deliberately not an input.

    The sm90 fp8 bands: decode keeps ``BLOCK_N=128`` -- the N=19456 gate_up
    projection regresses 38% on 64 -- and gains from deeper pipelining
    (``s=5``). Prefill forks on ``single_scale``: the epilogue-scale kernel
    takes the compute-shaped 128x128 tile (the per-k multiply used to make it
    2-3x slower), while the block-scale path still pays that multiply every k
    step and must stay at 64 rows of accumulator or spill registers.
    """
    if tile_tier(device_index) is TileTier.PRE_HOPPER:
        # A10-era table (sm86, measured, dtype-agnostic): the H100 SM count
        # rewards wider grids, not the fatter tiles this table uses.
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
    if single_scale and not is_fp8:
        # int8 per-channel, swept with the EPILOGUE_SCALE bands the launcher
        # compiles (in-loop multiply to 128 rows, acc= accumulation above).
        if num_tokens <= 32:
            return {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 1,
                "num_warps": 8,
                "num_stages": 5,
            }
        if num_tokens <= 128:
            return {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
                "num_warps": 4,
                "num_stages": 3,
            }
        if num_tokens <= 512:
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
                "num_warps": 8,
                "num_stages": 3,
            }
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        }
    if num_tokens <= 32:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 5,
        }
    if num_tokens <= 128:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }
    return {
        "BLOCK_M": 128 if single_scale else 64,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


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
    zeros: torch.Tensor | None = None,
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
        zeros: Optional asymmetric int8 zero points (GPTQ ``bits=8``), same
            shape and layout as ``scales``; ``None`` keeps the symmetric path.
            fp8 weights reject them — e4m3 has no zero point.

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
    if zeros is not None:
        if is_fp8:
            raise ValueError("zero points are asymmetric-int8 (GPTQ bits=8) only; fp8 is symmetric")
        if zeros.shape != scales.shape:
            raise ValueError(
                f"zeros {tuple(zeros.shape)} must share the scales' shape {tuple(scales.shape)}"
            )

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

    # sm89+ widens fp8 with the hardware cvt (one instruction, no 256x
    # correction); older devices keep the bit trick.
    fp8_cvt = is_fp8 and has_native_fp8(x.device.index)
    # per-row scales (group_k >= k): the scale's k address never moves, so the
    # kernel hoists its load out of the k loop.
    single_scale = group_k >= k
    # Autotune lookup (per-GPU entry) or the device-tiered heuristic, both
    # behind ``resolve_tiles``. The label names both forks the heuristic
    # tiers on — weight format and scale layout — so a tuned entry only ever
    # replays the path it measured.
    cfg = resolve_tiles(
        "w8a16_matmul",
        m=m,
        n=n,
        k=k,
        dtype_label=f"{'fp8' if is_fp8 else 'int8'}_{'single' if single_scale else 'block'}",
        heuristic=lambda dev: _launch_config(m, single_scale, is_fp8, dev),
        device_index=x.device.index,
    )
    # epilogue-scale accumulation only where the sm90 table is in play: the
    # in-loop multiply is faster at decode for int8 weights (3-11% measured),
    # and pre-Hopper keeps the in-loop form its A10 table was measured with.
    epilogue_scale = (
        single_scale and m > 128 and tile_tier(x.device.index) is TileTier.HOPPER_UP
    )
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _w8a16_matmul_kernel[grid](
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
        GROUP_N=group_n,
        GROUP_K=min(group_k, k),
        IS_FP8=is_fp8,
        HAS_ZEROS=zeros is not None,
        HAS_BIAS=bias is not None,
        DEQUANT_SCALE=1.0 if fp8_cvt else (FP8_E4M3_BIT_TRICK_SCALE if is_fp8 else 1.0),
        FP8_CVT=fp8_cvt,
        SINGLE_SCALE=single_scale,
        EPILOGUE_SCALE=epilogue_scale,
        **cfg,
    )
    return out.reshape(*leading, n)
