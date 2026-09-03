"""FP8 W8A8 GEMM: fp8-e4m3 weights + dynamic per-token fp8-e4m3 activations.

Both operands are quantised before the launch — activations per token,
weights per block — and the kernel multiplies in fp8 with fp32
accumulation, applying scales in the epilogue.

Usage:
    qx, x_scale = fp8_quantize_per_token(x)
    y = fp8_matmul(qx, x_scale, qweight, weight_scale_inv, group_n=1, group_k=K)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..tile_policy import TileTier, resolve_tiles, tile_tier
from ..tile_policy import (
    has_native_fp8 as has_native_fp8,  # re-exported: activation.py imports it from here
)
from .w8a16 import FP8_E4M3_BIT_TRICK_SCALE, dequant_fp8e4m3

#: Exponent correction when *both* operands went through the e4m3 -> bf16 bit
#: trick: each is short a factor of 256, so the product is short 256**2.
_FP8_BIT_TRICK_SCALE_SQ = FP8_E4M3_BIT_TRICK_SCALE * FP8_E4M3_BIT_TRICK_SCALE

#: Largest finite magnitude of e4m3. Mirrors
#: ``lite_llama.modules.quantization.utils.FP8_E4M3_MAX``; the two must agree,
#: which ``tests/kernels/test_quantization.py`` checks by comparing the two
#: quantisers' scales, which are exactly ``amax / FP8_E4M3_MAX``.
FP8_E4M3_MAX = 448.0

#: Elements one program of :func:`_quantize_fp8_per_token_kernel` handles per
#: pass. A row is walked in tiles rather than loaded whole so that a 9728-wide
#: FFN row does not need 38 KB of registers per program.
_QUANT_BLOCK_K = 1024


# --------------------------------------------------------------------------- #
# Per-token activation quantisation
# --------------------------------------------------------------------------- #
@triton.jit
def _quantize_fp8_per_token_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    K,
    stride_xm,
    stride_qm,
    FP8_MAX: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Quantise one row of ``x`` to e4m3 bytes with its own scale.

    Two passes over the row, both inside one program: the amax must be known
    before any element can be scaled, and keeping both passes here is what makes
    the scale a register rather than a round trip through HBM. The row is
    re-read from L2 on the second pass, which is why the traffic is counted as
    one read even though the kernel issues two.
    """
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm

    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))

    # An all-zero row would divide by zero; 1.0 leaves it exactly zero instead.
    scale = tl.where(amax > 0.0, amax / FP8_MAX, 1.0)
    tl.store(s_ptr + row, scale)

    q_row = q_ptr + row * stride_qm
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        q = x / scale
        # Bytes match the torch quantiser except on an exact tie (a quotient
        # of 84.0, halfway between the e4m3 codes 80 and 88): the hardware cvt
        # takes it the other way from torch's software cast -- one code, about
        # one element in 30k, and ``ieee_rounding=True`` changed zero bytes, so
        # the tie is genuine rather than slack in ``/``.
        # ``test_fp8_quantize_per_token_matches_torch_helper`` gates that bound.
        # The clamp is a no-op by construction (amax/FP8_MAX) and stays as the
        # guard against a non-finite input turning into a NaN byte pattern.
        q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
        tl.store(q_row + offs, q.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=mask)


def fp8_quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise activations to e4m3 bytes, one scale per row, in one launch.

    The torch spelling (``modules.quantization.utils.quantize_fp8_per_token``)
    is a chain of about eight elementwise ops whose launch overhead *is* the
    cost at decode shapes: 45-55 us against a 20-32 us fp8 GEMM, which made
    ``w8a8_fp8`` slower than bf16 cuBLAS at ``m=1`` for reasons that had
    nothing to do with fp8. This kernel removes it (the decomposition came
    from ``bench_quant_gemm.py``'s ``ablation: fp8_matmul only`` row).

    Contract-identical to the torch helper — bytes agree except on exact e4m3
    ties, see the kernel — and no host synchronisation, so a MoE layer holding
    it on its critical path stays CUDA-graph capturable.

    Args:
        x: ``[..., K]`` float activations. Leading dims are flattened to rows.

    Returns:
        ``(qx, scales)`` with ``qx`` the ``uint8`` e4m3 bit pattern shaped like
        ``x`` and ``scales`` ``[..., 1]`` fp32.
    """
    if not has_native_fp8(x.device.index):
        # Triton cannot emit an e4m3 cast below sm89, and there is no cheap bit
        # trick in this direction (rounding and subnormals both need handling),
        # so pre-Hopper keeps the torch path. It is the slow one, but fp8 W8A8
        # has no native MMA there either — the format is not the fast choice on
        # that hardware to begin with.
        from ....modules.quantization.utils import quantize_fp8_per_token

        return quantize_fp8_per_token(x)

    k = x.shape[-1]
    flat = x.reshape(-1, k)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    m = flat.shape[0]

    qx = torch.empty_like(flat, dtype=torch.uint8)
    scale = torch.empty(m, dtype=torch.float32, device=x.device)
    _quantize_fp8_per_token_kernel[(m,)](
        flat,
        qx,
        scale,
        k,
        flat.stride(0),
        qx.stride(0),
        # Passed in rather than read from the module: a @triton.jit body can only
        # close over constexpr globals, and one definition of the range limit is
        # worth more here than saving an argument.
        FP8_MAX=FP8_E4M3_MAX,
        BLOCK_K=min(triton.next_power_of_2(k), _QUANT_BLOCK_K),
    )
    return qx.reshape(x.shape), scale.reshape(*x.shape[:-1], 1)


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
    SINGLE_SCALE: tl.constexpr,
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
    # One scale per weight row (per-channel quantisation, the scheme's
    # default): its k address never moves, so load it before the loop and keep
    # the loop body down to fp8 tiles and dots.
    if SINGLE_SCALE:
        b_scale = tl.load(b_scale_ptrs)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        # NATIVE_FP8 is constexpr: Triton emits two specialised kernels, so
        # the bf16 widening below never reaches an sm89+ binary (and the fp8
        # bitcast never reaches an sm86 one).
        if NATIVE_FP8:
            a = a.to(tl.float8e4nv, bitcast=True)
            b = b.to(tl.float8e4nv, bitcast=True)
        else:
            a = dequant_fp8e4m3(a)
            b = dequant_fp8e4m3(b)
        if SINGLE_SCALE:
            # ``D = A*B + D`` keeps the wgmma chain asynchronous end to end;
            # multiplying the fp32 product by the scale every k step serialises
            # it (3-4x at prefill shapes, measured). The per-row scale
            # distributes over the sum, so one multiply in the epilogue is the
            # same arithmetic up to fp32 rounding order, which the unit tests'
            # tolerances cover.
            accumulator = tl.dot(a, tl.trans(b), acc=accumulator)
        else:
            b_scale = tl.load(b_scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_bsk)
            accumulator += tl.dot(a, tl.trans(b)) * b_scale[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_scale = tl.load(a_scale_ptr + offs_am)
    result = accumulator * (DEQUANT_SCALE * a_scale[:, None])
    if SINGLE_SCALE:
        result *= b_scale[None, :]
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
def _launch_config(num_tokens: int, single_scale: bool, device_index: int | None) -> dict:
    """Tile shape for ``num_tokens`` rows on this device.

    Swept over the five dense projections of both test models on an H100
    (the ``bench_quant_gemm.py`` shape set); each band's entry is the
    geomean-best config over those five shapes. Decode stays ``BLOCK_N=64``
    because it fills the grid when M supplies no parallelism, and the bands
    with real arithmetic fork on ``single_scale``: the epilogue-scale kernel
    takes the wider tile now that ``acc=`` keeps the wgmma chain asynchronous,
    while the block-scale path still pays the per-k multiply and keeps the
    leaner 64-row accumulator. Pre-Hopper keeps the A10-era table the sm90
    sweep replaced; ``n`` is not an input, because narrow/wide projection
    groups pick the same geomean-best tile.
    """
    if tile_tier(device_index) is TileTier.PRE_HOPPER:
        # A10-era table (sm86, measured): one tile set, no single_scale
        # fork -- the block-scale/per-row paths had not diverged yet.
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
            "num_stages": 3,
        }
    if num_tokens <= 128:
        return {
            "BLOCK_M": 32,
            "BLOCK_N": 128 if single_scale else 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 4 if single_scale else 3,
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
        out_dtype: Output dtype; the rest of the network runs bf16 (or bf16
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
    native = has_native_fp8(qx.device.index)
    # per-row weight scales (group_k >= k): the scale's k address never moves,
    # so the kernel can hoist its load out of the k loop.
    single_scale = group_k >= k

    # Autotune lookup (per-GPU entry) or the device-tiered heuristic, both
    # behind ``resolve_tiles``. The dtype label carries the scale layout
    # because the two paths fork on tile shape in the heuristic and compile
    # different epilogues — one entry must never stand in for the other.
    cfg = resolve_tiles(
        "fp8_matmul",
        m=m,
        n=n,
        k=k,
        dtype_label="fp8_single" if single_scale else "fp8_block",
        heuristic=lambda dev: _launch_config(m, single_scale, dev),
        device_index=qx.device.index,
    )
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
        SINGLE_SCALE=single_scale,
        **cfg,
    )
    return out.reshape(*leading, n)
