"""Fused MoE: top-k routed experts as a Triton grouped GEMM.

``moe_align_block_size`` sorts tokens by expert into padded blocks; the
fused kernel then runs both expert GEMMs and the activation for every
block, with quantisation folded in per weight format.

The activation may be fp16 or bf16 and every quantised weight tile is widened to
whichever it is: ``tl.dot`` needs both operands in one type, so the compute dtype
is the activation's to choose, not the weight format's. The one exception is
:func:`fused_moe_w8a8_fp8`, where the activation is quantised to e4m3 as well and
both operands stay 8-bit through the dot. Whether that dot *lands* on the fp8
tensor cores is Triton's call, not ours: it emits ``wgmma`` only from ``BLOCK_M >=
64``, and widens both e4m3 operands to an fp16 ``mma.sync`` below that. Either way
the inner loop skips the bit-trick dequantisation the weight-only modes pay, which
is where the mode's speed actually comes from -- see ``_FP8_A8_PROMOTE_EVERY``.

Usage:
    y = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)
    y = fused_moe(x, qw1, qw2, tw, ids, w1_scale=s1, w2_scale=s2,
                  group_n=128, group_k=128)
    y = fused_moe_w8a8_fp8(x, qw1, qw2, tw, ids, w1_scale=s1, w2_scale=s2,
                           group_n=1, group_k=hidden)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..activation.activations import silu
from ..quantization.fp8 import fp8_quantize_per_token
from ..quantization.w8a16 import FP8_E4M3_BIT_TRICK_SCALE, dequant_fp8e4m3
from ..utils import torch_to_triton_dtype

#: ``QUANT_MODE`` values shared by the kernel and its launcher. Modes 1-3 are
#: weight-only — the activation stays fp16/bf16 — while mode 4 is true W8A8:
#: both operands enter the dot as e4m3 bytes. Mode 4 cannot be inferred from
#: ``w1.dtype`` the way the others are, because weight-only fp8 and W8A8 fp8
#: store the same ``uint8`` experts; the caller selects it.
_QUANT_NONE = 0
_QUANT_FP8 = 1
_QUANT_INT8 = 2
_QUANT_INT4 = 3
_QUANT_FP8_A8 = 4

#: Exponent correction for the pre-sm89 path of mode 4, where *both* operands
#: are widened by the e4m3 -> fp16 bit trick and each is short a factor of 256.
#: Derived from the one imported constant rather than restated, so it cannot
#: drift from ``quantization.fp8``'s copy of the same reasoning.
_FP8_BIT_TRICK_SCALE_SQ = FP8_E4M3_BIT_TRICK_SCALE * FP8_E4M3_BIT_TRICK_SCALE

#: k-tile of the quantised path: one byte per weight element, k-contiguous, so a
#: 128-wide tile is one full memory transaction per output channel.
_QUANT_BLOCK_K = 128

#: How many k elements mode 4 may accumulate inside Hopper's fp8 ``wgmma`` before
#: the partial sum is promoted into a real fp32 accumulator. The instruction's
#: internal accumulation is *not* full fp32, and Triton's sm90 default
#: (``max_num_imprecise_acc_default = 2**30``) never promotes at all.
#:
#: Measured on H100 (sm90), K=512 dot with both operands filling the e4m3 range,
#: against an fp32 reference over the same bytes -- p99.9 relative error:
#: unbounded 7.0e-2, promote every 32 2.5e-2, promote every 0 4.5e-6. The last is
#: not a more accurate fp8 dot: at 0 Triton drops ``wgmma`` and widens both
#: operands to fp16 ``mma.sync`` instead, so there is no precise fp8 MMA to pick.
#:
#: At the Qwen3-30B-A3B MoE geometry all three are indistinguishable (RMS relative
#: error 6.5e-3 / 6.6e-3 / 6.8e-3): the two e4m3 roundings swamp the accumulator,
#: and the grouped GEMM is bound by re-reading expert weights once per row-block,
#: not by the MMA, so the three also land within 5% of each other on time
#: (tokens=16384: 4637 / 4695 / 4457 us). 32 is the middle: it keeps the dot at 8
#: bits, where the unbounded default's worst case does not apply.
_FP8_A8_PROMOTE_EVERY = 32

#: Number of int4 values packed per int32 word.
_INT4_PACK_FACTOR = 8


# --------------------------------------------------------------------------- #
# Token alignment (torch-native; every output shape is static -> no host sync)
# --------------------------------------------------------------------------- #
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort token slots by expert and pad each expert's run to ``block_size``.

    Args:
        topk_ids: ``[num_tokens, top_k]`` expert indices.
        block_size: GEMM row-block the kernel iterates over.
        num_experts: Total expert count ``E``.

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``; see the
        module docstring for the exact protocol.
    """
    device = topk_ids.device
    flat_experts = topk_ids.reshape(-1).to(torch.int64)
    num_slots = flat_experts.numel()

    # Stable argsort groups slots by expert while keeping token order inside
    # each expert; sort_index doubles as the slot ids in sorted order.
    sort_index = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[sort_index]

    # scatter_add rather than bincount: bincount reads the largest element back
    # to the host to size its output, which both stalls the launch pipeline
    # (~0.4 ms, more than the two GEMMs cost at decode batch sizes) and makes the
    # whole MoE layer uncapturable as a CUDA graph.
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts))
    padded_counts = (
        torch.div(counts + (block_size - 1), block_size, rounding_mode="floor") * block_size
    )
    padded_starts = torch.cumsum(padded_counts, 0) - padded_counts
    real_starts = torch.cumsum(counts, 0) - counts
    num_tokens_post_padded = padded_counts.sum().to(torch.int32).reshape(1)

    # Static upper bound: every expert may waste at most block_size - 1 slots.
    max_padded = num_slots + num_experts * (block_size - 1)
    sorted_token_ids = torch.full((max_padded,), num_slots, dtype=torch.int32, device=device)
    rows = torch.arange(num_slots, device=device, dtype=torch.int64)
    dest = padded_starts[sorted_experts] + (rows - real_starts[sorted_experts])
    sorted_token_ids[dest] = sort_index.to(torch.int32)

    # expert_ids[b] = expert owning row-block b. searchsorted over the block
    # boundaries keeps the output shape static (out-of-range blocks are never
    # read: the kernel returns early on num_tokens_post_padded).
    max_num_blocks = (max_padded + block_size - 1) // block_size
    block_ends = torch.cumsum(padded_counts // block_size, 0)
    block_ids = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    expert_ids = (
        torch.searchsorted(block_ends, block_ids, right=True)
        .clamp_(max=num_experts - 1)
        .to(torch.int32)
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# --------------------------------------------------------------------------- #
# Grouped GEMM
# --------------------------------------------------------------------------- #
@triton.jit
def _fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    b_zeros_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_slots,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    stride_bsk,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    QUANT_MODE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
    K_PROMOTE: tl.constexpr,
    DEQUANT_SCALE: tl.constexpr,
    HAS_ZEROS: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One C row-block of ``A @ B[expert]`` where rows of A are gathered tokens.

    A: ``[num_tokens, K]`` activations. B: ``[E, N, K]`` stacked expert weights,
    fp16 or 8-bit or int4 packed. C: ``[num_tokens * top_k, N]`` (each token's per-slot output row).
    When ``QUANT_MODE`` is non-zero, ``b_scale_ptr`` holds dequantisation scales.
    When ``QUANT_MODE == 3`` (INT4), B is ``[E, N, K//8]`` int32 packed (8 nibbles per word),
    ``b_scale_ptr`` is ``[E, N, K//group_k]``, and optionally ``b_zeros_ptr`` holds zero points.
    When ``QUANT_MODE == 4`` (fp8 W8A8), A is ``uint8`` e4m3 too and
    ``a_scale_ptr`` holds one fp32 scale per A row; ``NATIVE_FP8`` then picks
    between keeping both operands 8-bit for the sm89+ fp8 MMA and the pre-sm89
    widening. Both are read only in that mode, and unused elsewhere.
    """
    # Grouped pid ordering for L2 reuse (same scheme as vLLM/triton matmul).
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_slots
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak

    if QUANT_MODE == 3:
        # INT4: B is [E, N, K//8] int32, packed along K dim. stride_bk is per-word.
        # For each k-tile of BLOCK_K logical elements, we load BLOCK_K//8 int32 words.
        offs_k_words = tl.arange(0, BLOCK_K // 8)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[None, :] * stride_bn
            + offs_k_words[:, None] * stride_bk
        )
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
    if QUANT_MODE != 0:
        # Scale row is fixed for this tile; only the k-block index advances.
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + (offs_bn // GROUP_N) * stride_bsn

    # fp32 accumulation keeps the K-loop noise below the fp16 storage floor.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
            # An int literal, not 0.0: in mode 4 this pointer is ``uint8``, and
            # the e4m3 bit pattern 0 is +0.0 anyway, so one spelling serves both.
            other=0,
        )
        if QUANT_MODE == 3:
            # INT4 path: load int32 words [BLOCK_K//8, BLOCK_N], unpack to [BLOCK_K, BLOCK_N]
            b_packed = tl.load(
                b_ptrs,
                mask=offs_k_words[:, None] < (K // 8) - k * (BLOCK_K // 8),
                other=0,
            )  # [BLOCK_K//8, BLOCK_N]
            # Unpack 8 int4 nibbles per word: shift by [0,4,8,...,28] and mask 0xF
            shifts = (tl.arange(0, 8) * 4).to(tl.int32)  # [8]
            # Reshape for broadcast: [BLOCK_K//8, 1, BLOCK_N] x [1, 8, 1] -> [BLOCK_K//8, 8, BLOCK_N]
            b_expanded = (b_packed[:, None, :] >> shifts[None, :, None]) & 0xF
            # Flatten to [BLOCK_K, BLOCK_N]
            b = tl.reshape(b_expanded, (BLOCK_K, BLOCK_N)).to(tl.float32)
            # Load scale and optionally zero point
            b_scale = tl.load(b_scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_bsk)
            if HAS_ZEROS:
                b_zero = tl.load(
                    b_zeros_ptr
                    + off_experts * stride_bse
                    + (offs_bn // GROUP_N) * stride_bsn
                    + ((k * BLOCK_K) // GROUP_K) * stride_bsk
                )
                b = b - b_zero[None, :]
            # Only the zero point enters the operand; the fp32 scale stays
            # outside the dot, as in the 8-bit branches below. Both the nibble
            # and the zero point are integers in [0, 15], so ``b`` is an exact
            # small integer in ``compute_type`` and folding the scale in here
            # would cost precision as well as the operand's dtype.
            accumulator += tl.dot(a, b.to(compute_type)) * b_scale[None, :]
            b_ptrs += (BLOCK_K // 8) * stride_bk
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
            if QUANT_MODE == 0:
                accumulator = tl.dot(a, b, acc=accumulator)
            else:
                # Widened to the activation's dtype, not to a fixed fp16:
                # ``tl.dot`` rejects mixed operand types, and a bf16 model would
                # otherwise fail to compile every quantised expert GEMM. The
                # e4m3 bit trick lands in fp16 by construction and its values
                # carry 4 significant bits, so the extra hop to bf16 is exact.
                if QUANT_MODE == 4:
                    # W8A8: nothing is widened to compute_type at all. Both
                    # operands stay 8-bit into the dot, and Triton decides from
                    # BLOCK_M whether that becomes an fp8 wgmma or an fp16
                    # mma.sync. B is already loaded k-contiguous ([E, N, K] means
                    # stride_bk == 1), the layout the fp8 MMA wants, so no
                    # transpose is needed here -- unlike the dense
                    # ``fp8_matmul``, whose B arrives as [N, K].
                    if NATIVE_FP8:
                        a = a.to(tl.float8e4nv, bitcast=True)
                        b = b.to(tl.float8e4nv, bitcast=True)
                    else:
                        a = dequant_fp8e4m3(a)
                        b = dequant_fp8e4m3(b)
                elif QUANT_MODE == 1:
                    b = dequant_fp8e4m3(b).to(compute_type)
                else:
                    b = b.to(compute_type)
                # Hoisted out of the dot, so the tensor cores see two 16-bit
                # operands instead of the fp32 scale's type.
                b_scale = tl.load(b_scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_bsk)
                if QUANT_MODE == 4:
                    # Hopper's fp8 wgmma accumulates at reduced precision inside
                    # the instruction, and Triton's sm90 default lets that run
                    # unbounded. ``K_PROMOTE`` caps how many k elements may
                    # accumulate before the result is promoted into a real fp32
                    # accumulator; 0 makes Triton drop wgmma entirely and widen
                    # both operands to fp16 instead. See ``_FP8_A8_PROMOTE_EVERY``.
                    accumulator += tl.dot(a, b, max_num_imprecise_acc=K_PROMOTE) * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * b_scale[None, :]
            b_ptrs += BLOCK_K * stride_bk
        a_ptrs += BLOCK_K * stride_ak

    accumulator *= DEQUANT_SCALE
    if QUANT_MODE == 4:
        # One scale per activation row, so it leaves the k-loop untouched. The
        # mask matters: a padded slot's ``offs_token`` is the sentinel
        # ``num_valid_slots``, one past the last real row of a_scale.
        a_scale = tl.load(a_scale_ptr + offs_token // top_k, mask=token_mask, other=0.0)
        accumulator *= a_scale[:, None]
    # Router weights multiply in fp32 before the final downcast (gemm2 only).
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _launch_config(num_tokens: int, quant_mode: int) -> dict:
    """Heuristic tile config, used whenever the autotune store has no entry.

    ``BLOCK_M`` must be identical for both GEMMs because they share one alignment.

    ``BLOCK_K`` is 128 for every mode, including fp16. It used to be 32 there, on
    the reasoning that an fp16 tile already fills a memory transaction at 32 while
    a byte tile needs 128 — true about the transaction, wrong about the layer. A
    tile sweep over the Qwen3-30B-A3B expert geometry
    (``benchmarks/kernels/bench_fused_moe.py --tune``) found no winning fp16 config
    with ``BLOCK_K`` below 64 at any token count, and the narrow tile was costing
    26% at 64 tokens and 32% at 512-4096 by running four times the k-iterations.
    The quantised paths were never affected, which is why the defect survived: it
    made every quantisation benchmark on this kernel look better than it was.

    ``quant_mode`` is still in the signature because callers pass it and a future
    per-mode divergence belongs here rather than at the call sites.
    """
    if num_tokens <= 16:
        block_m = 16
    elif num_tokens <= 64:
        block_m = 32
    else:
        block_m = 64
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": 64,
        "BLOCK_K": _QUANT_BLOCK_K,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


def _invoke_moe_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scale: torch.Tensor | None,
    b_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool,
    quant_mode: int,
    group_n: int,
    group_k: int,
    config: dict,
) -> None:
    """Launch one grouped GEMM: ``C[slot] = A[slot // top_k] @ B[expert].T``."""
    assert a.stride(-1) == 1 and b.stride(-1) == 1, "last dims must be contiguous"
    em = sorted_token_ids.numel()
    num_slots = c.shape[0]
    n, k = b.shape[1], b.shape[2]
    # For INT4 mode, K in the tensor is K_logical // 8 (packed), but we pass K_logical.
    k_logical = k * _INT4_PACK_FACTOR if quant_mode == _QUANT_INT4 else k
    # Only mode 4 puts A through the tensor cores as fp8, so only it cares which
    # of the two widenings the kernel compiles; the weight-only modes always take
    # the bit trick and always pay its single correction factor.
    native_fp8 = quant_mode == _QUANT_FP8_A8 and torch.cuda.get_device_capability(a.device) >= (
        8,
        9,
    )
    if quant_mode == _QUANT_FP8_A8:
        dequant_scale = 1.0 if native_fp8 else _FP8_BIT_TRICK_SCALE_SQ
    else:
        dequant_scale = FP8_E4M3_BIT_TRICK_SCALE if quant_mode == _QUANT_FP8 else 1.0
    grid = (triton.cdiv(em, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]),)
    _fused_moe_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        b_zeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k_logical,
        em,
        num_slots,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        *(b_scale.stride() if b_scale is not None else (0, 0, 0)),
        GROUP_N=group_n or 1,
        GROUP_K=min(group_k, k_logical) if group_k else 1,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        GROUP_M=config["GROUP_M"],
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        QUANT_MODE=quant_mode,
        NATIVE_FP8=native_fp8,
        K_PROMOTE=min(_FP8_A8_PROMOTE_EVERY, config["BLOCK_K"]),
        DEQUANT_SCALE=dequant_scale,
        HAS_ZEROS=b_zeros is not None,
        compute_type=torch_to_triton_dtype[c.dtype],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


# --------------------------------------------------------------------------- #
# Activation and top-k reduction
# --------------------------------------------------------------------------- #
@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    N,
    BLOCK_N: tl.constexpr,
):
    """``out = silu(x[:, :N]) * x[:, N:]`` on a contiguous ``[tokens, 2N]`` input."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    # silu evaluates its sigmoid in fp32, matching the dense swiglu kernel.
    gate = tl.load(x_ptr + pid_m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + pid_m * stride_xm + N + offs, mask=mask, other=0.0).to(tl.float32)
    out = silu(gate) * up
    tl.store(out_ptr + pid_m * N + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    N,
    top_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``output[m] = sum_k input[m * top_k + k]`` over the expanded slot dim."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    base = input_ptr + pid_m * top_k * N
    for i in tl.static_range(top_k):
        acc += tl.load(base + i * N + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + pid_m * N + offs, acc.to(output_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
def _quant_mode(weight: torch.Tensor, scale: torch.Tensor | None) -> int:
    """Classify an expert weight tensor into one of the ``QUANT_MODE`` values.

    Never returns :data:`_QUANT_FP8_A8`: the dtype cannot tell weight-only fp8
    from W8A8 fp8, since both store ``uint8`` e4m3 experts. Only the entry point
    the caller chose says which, so :func:`fused_moe_w8a8_fp8` promotes the mode
    after this classification.
    """
    if scale is None:
        return _QUANT_NONE
    if weight.dtype == torch.uint8:
        return _QUANT_FP8
    if weight.dtype == torch.int8:
        return _QUANT_INT8
    if weight.dtype == torch.int32:
        return _QUANT_INT4
    raise ValueError(f"quantised expert weights must be uint8, int8 or int32, got {weight.dtype}")


def _fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
    act_fp8: bool = False,
) -> torch.Tensor:
    """Body of both public entry points; ``act_fp8`` is their only difference.

    Private, and the two thin wrappers below are what callers and the registry
    see, because ``tests/ops/test_native_specs.py`` pins every registered
    target's parameter *names* to the :class:`MoeOp` ABC. A public ``act_fp8``
    flag would either break that pin or have to be added to the ABC, where it
    would become a promise the DeepGEMM row cannot keep. Two names for two rows
    costs less than one signature that lies.
    """
    num_tokens, hidden = hidden_states.shape
    num_experts, two_inter, _ = w1.shape
    intermediate = two_inter // 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    quant_mode = _quant_mode(w1, w1_scale)
    if quant_mode != _quant_mode(w2, w2_scale):
        raise ValueError("w1 and w2 must use the same quantisation format")
    if quant_mode and group_k % 128 != 0 and group_k < min(hidden, intermediate):
        raise ValueError(f"group_k ({group_k}) must be a multiple of 128 unless it covers K")
    if act_fp8:
        if quant_mode != _QUANT_FP8:
            raise ValueError(
                "fp8 W8A8 experts must be uint8 e4m3 bytes with scales, got "
                f"{w1.dtype} (mode {quant_mode})"
            )
        quant_mode = _QUANT_FP8_A8

    topk_ids = topk_ids.to(torch.int32)
    flat_weights = topk_weights.reshape(-1).to(dtype).contiguous()

    # Autotune lookup: use persisted best config if available, else heuristic.
    from ...dispatcher.autotune import get_best_config

    dtype_label = {
        _QUANT_NONE: "fp16",
        _QUANT_FP8: "fp8",
        _QUANT_INT8: "int8",
        _QUANT_INT4: "int4",
        # Its own key, not "fp8": the two modes compile different inner loops and
        # want different tiles, so sharing a TuneKey would let one mode's search
        # install a config the other never measured.
        _QUANT_FP8_A8: "fp8_a8",
    }.get(quant_mode, "fp16")
    config = get_best_config("fused_moe", m=num_tokens, n=two_inter, k=hidden, dtype=dtype_label)
    if config is None:
        config = _launch_config(num_tokens, quant_mode)
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, config["BLOCK_M"], num_experts
    )

    # GEMM1: [M, hidden] x [E, 2I, hidden] -> [M * top_k, 2I]
    # In W8A8 the activation is quantised once per token here, before the gather:
    # a slot reads a row, so quantising the [M, hidden] input costs top_k times
    # less than quantising the expanded slots would, and every slot of a token
    # then shares one scale. No host synchronisation in the quantiser, so the
    # layer stays capturable into a CUDA graph.
    a1, a1_scale = hidden_states, None
    if quant_mode == _QUANT_FP8_A8:
        a1, a1_scale = fp8_quantize_per_token(hidden_states)
        a1_scale = a1_scale.reshape(-1)
    gate_up = torch.empty((num_tokens * top_k, two_inter), device=device, dtype=dtype)
    _invoke_moe_gemm(
        a1,
        w1,
        gate_up,
        a1_scale,
        w1_scale,
        w1_zeros,
        flat_weights,
        sorted_ids,
        expert_ids,
        num_post,
        top_k,
        mul_routed_weight=False,
        quant_mode=quant_mode,
        group_n=group_n,
        group_k=group_k,
        config=config,
    )

    # silu(gate) * up -> [M * top_k, I]
    act = torch.empty((num_tokens * top_k, intermediate), device=device, dtype=dtype)
    block_n = min(triton.next_power_of_2(intermediate), 1024)
    _silu_and_mul_kernel[(num_tokens * top_k, triton.cdiv(intermediate, block_n))](
        gate_up, act, gate_up.stride(0), intermediate, BLOCK_N=block_n, num_warps=4
    )

    # GEMM2 with the routing weight folded in: [M * top_k, I] x [E, hidden, I].
    # ``act`` is already expanded per slot, so the kernel's A-row gather must use
    # top_k=1 (vLLM does the same: the second invocation passes ``1``), turning
    # ``offs_token // top_k`` into the identity on slot indices -- which is also
    # what makes one a_scale row per slot the right shape here.
    a2, a2_scale = act, None
    if quant_mode == _QUANT_FP8_A8:
        a2, a2_scale = fp8_quantize_per_token(act)
        a2_scale = a2_scale.reshape(-1)
    expanded = torch.empty((num_tokens * top_k, hidden), device=device, dtype=dtype)
    _invoke_moe_gemm(
        a2,
        w2,
        expanded,
        a2_scale,
        w2_scale,
        w2_zeros,
        flat_weights,
        sorted_ids,
        expert_ids,
        num_post,
        1,
        mul_routed_weight=True,
        quant_mode=quant_mode,
        group_n=group_n,
        group_k=group_k,
        config=config,
    )

    # Reduce over the top_k slot dim -> [M, hidden]
    out = torch.empty((num_tokens, hidden), device=device, dtype=dtype)
    block_n = min(triton.next_power_of_2(hidden), 1024)
    _moe_sum_kernel[(num_tokens, triton.cdiv(hidden, block_n))](
        expanded, out, hidden, top_k=top_k, BLOCK_N=block_n, num_warps=4
    )
    return out


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """Run the routed-expert FFN: ``sum_k w_k * (silu(x @ W1g.T) * (x @ W1u.T)) @ W2.T``.

    Weight-only for every quantised format: the expert tiles are widened to the
    activation's dtype inside the loop and the activation itself is never
    touched. :func:`fused_moe_w8a8_fp8` is the one that quantises it.

    Args:
        hidden_states: ``[num_tokens, hidden]`` activations (fp16 or bf16,
            contiguous rows). Every quantised weight tile is widened to this
            dtype before the dot.
        w1: ``[E, 2 * moe_intermediate, hidden]`` fused gate/up projections, fp16
            or 8-bit (``uint8`` fp8-e4m3 / ``int8``) or int4 packed (``int32``).
        w2: ``[E, hidden, moe_intermediate]`` down projections, same dtype as ``w1``.
        topk_weights: ``[num_tokens, top_k]`` routing weights.
        topk_ids: ``[num_tokens, top_k]`` expert indices.
        w1_scale: Dequantisation scales; ``None`` selects the fp16 path.
        w2_scale: Scales for ``w2``.
        w1_zeros: Zero points for int4 (AWQ/GPTQ); ``None`` for symmetric.
        w2_zeros: Zero points for ``w2``; ``None`` for symmetric.
        group_n: Rows of one scale block (``1`` = per output channel).
        group_k: Columns of one scale block.

    Returns:
        ``[num_tokens, hidden]`` combined expert output, in ``hidden_states``' dtype.
    """
    return _fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        group_n=group_n,
        group_k=group_k,
        act_fp8=False,
    )


def fused_moe_w8a8_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """The same FFN with the activations in fp8 as well: true W8A8 experts.

    Both grouped GEMMs run on the fp8 tensor cores (sm89+) instead of widening
    the expert tile to bf16 first, which is the difference between quantising for
    memory and quantising for arithmetic. The activation is quantised per token
    before GEMM1 and per slot-row before GEMM2, in one kernel each
    (:func:`~lite_llama.kernels.ops.quantization.fp8.fp8_quantize_per_token`),
    with no host synchronisation — the layer must stay CUDA-graph capturable,
    which is also why ``moe_align_block_size`` avoids ``bincount``.

    ``w1_zeros``/``w2_zeros`` exist only to keep the :class:`MoeOp` contract; fp8
    is symmetric, so a non-``None`` value is a caller error.

    Args:
        See :func:`fused_moe`. ``w1``/``w2`` must be ``uint8`` e4m3 bytes with
        scales — anything else raises, rather than silently degrading to the
        weight-only path.

    Returns:
        ``[num_tokens, hidden]`` combined expert output, in ``hidden_states``' dtype.
    """
    if w1_zeros is not None or w2_zeros is not None:
        raise ValueError("fp8 is symmetric; zero points belong to the int4 path")
    return _fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        group_n=group_n,
        group_k=group_k,
        act_fp8=True,
    )
