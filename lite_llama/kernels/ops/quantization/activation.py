"""Standalone dynamic activation quantisation (per-token-group).

The W8A8 GEMM files each embed their own quantiser — ``w8a8.py`` and
``fp8.py`` quantise activations per *token*, because a per-token scale is what
their GEMM epilogues consume — so until now the package had no separate
activation-quantisation op of its own. This module adds the one the
block-wise schemes need, mirroring sglang's ``per_token_group_quant``
(``kernels/ops/quantization/per_token_group_quant.py``): the activation-side
companion of a ``block_shape=[128, 128]`` W8A8 checkpoint, where every
``group_size``-element slice of every token carries its own scale.

One launch, one pass. A per-token scale needs the whole row's amax before any
element of the row can be scaled, which is why the per-token quantisers walk
their row twice. A *group* fits in one tile, so the amax reduction and the
scaling happen on data that is already in registers: the row is read once,
and each program carries ``_QUANT_TILE`` elements' worth of groups (eight at
``group_size=128``) where sglang's flat kernel carries one.

Usage:
    qx, scales = per_token_group_quant(x, 128, out_dtype=torch.uint8)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..activation.activations import silu
from .fp8 import FP8_E4M3_MAX, has_native_fp8

#: Largest magnitude symmetric int8 stores. A local restatement of
#: ``w8a8._INT8_MAX`` — kept private there, so this module states its own.
_INT8_MAX = 127.0

#: Elements one program of :func:`_per_token_group_quant_kernel` covers. Eight
#: 128-wide groups: the same 1024-element footprint the per-token quantisers
#: walk per pass, held as ``[groups, group_size]`` so one program amortises
#: its launch over several scales.
_QUANT_TILE = 1024


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _per_token_group_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_qm,
    stride_sm,
    stride_sk,
    G,
    H,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
    QMAX: tl.constexpr,
    OUT_FP8: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    EPS: tl.constexpr,
):
    """Quantise ``[GROUPS_PER_PROG, GROUP_SIZE]`` groups of one token row.

    A group's amax and the elements it scales live in the same tile, so the
    reduction (``axis=1``) and the scaling are register work — no second pass
    through HBM, which is the tax the per-token quantisers must pay because a
    whole row does not fit alongside its own amax.
    """
    row = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1)

    g = pid_g * GROUPS_PER_PROG + tl.arange(0, GROUPS_PER_PROG)
    k = tl.arange(0, GROUP_SIZE)
    offs = g[:, None] * GROUP_SIZE + k[None, :]
    mask = offs < H

    gate = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    if FUSE_SILU:
        # Gate/up halves share the row's base pointer; ``offs`` stays in
        # ``[0, H)`` so the up load at ``H + offs`` cannot leave the row.
        up = tl.load(x_ptr + row * stride_xm + H + offs, mask=mask, other=0.0).to(tl.float32)
        val = silu(gate) * up
    else:
        val = gate

    amax = tl.max(tl.abs(val), axis=1)
    # sglang's eps floor, not the per-token quantisers' 1.0-on-zero: a scale of
    # eps/QMAX keeps an all-zero group's bytes exactly zero without a branch,
    # and matches the reference the block-wise schemes were calibrated on.
    scale = tl.maximum(amax, EPS) / QMAX
    q = val / scale[:, None]
    # amax/QMAX makes the clamp a no-op by construction; it stays as the guard
    # against a non-finite input turning into a NaN byte pattern.
    q = tl.minimum(tl.maximum(q, -QMAX), QMAX)

    tl.store(s_ptr + row * stride_sm + g * stride_sk, scale, mask=g < G)
    if OUT_FP8:
        # Same cast story as fp8.py's per-token kernel: the hardware cvt is
        # round-to-nearest-even except on an exact e4m3 tie (a quotient landing
        # halfway between two codes), where it may pick the other neighbour
        # from torch's software cast — one code, about one element in 30k. The
        # byte-difference tests below gate that bound.
        q = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        # rint, not a plain .to(int8): round-to-nearest-even matches torch's
        # .round() (and the per-token int8 quantiser), where .to truncates
        # toward zero — a different byte wherever the quotient's fraction
        # exceeds one half. Like the e4m3 cast above, it agrees with the torch
        # chain everywhere except on a quotient landing exactly on a .5
        # boundary, where a 1 ULP difference between this kernel's and torch's
        # fp32 division flips the tie (~4e-4 of elements on a 512x7168 row);
        # the byte-difference tests gate that bound.
        q = tl.extra.cuda.libdevice.rint(q).to(tl.int8)
    tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)


def _per_token_group_quant_torch(
    flat: torch.Tensor,
    h: int,
    group_size: int,
    fuse_silu_and_mul: bool,
    qmax: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager spelling of the kernel, for devices without the fp8 cast.

    Same role as ``fp8_quantize_per_token``'s torch fallback: pre-sm89 Triton
    cannot emit the e4m3 conversion, and block-wise W8A8 has no native MMA
    there either. int8 never routes here — its conversion is plain arithmetic.
    """
    val = flat.float()
    if fuse_silu_and_mul:
        gate, up = val[:, :h], val[:, h:]
        val = gate * torch.sigmoid(gate) * up
    t, g = flat.shape[0], h // group_size
    grouped = val.reshape(t, g, group_size)
    amax = grouped.abs().amax(dim=-1)
    scale = amax.clamp_min(eps) / qmax
    q = (grouped / scale[:, :, None]).clamp(-qmax, qmax).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).reshape(t, h), scale


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def per_token_group_quant(
    x: torch.Tensor,
    group_size: int = 128,
    *,
    out_dtype: torch.dtype = torch.int8,
    column_major_scales: bool = False,
    fuse_silu_and_mul: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise activations with one scale per token per ``group_size`` elements.

    The activation-side companion of a ``block_shape=[128, 128]`` W8A8 scheme
    (sglang's ``per_token_group_quant``): where the per-token quantisers give
    every row one scale, this gives every ``group_size``-wide slice of every
    row its own — the granularity a block-quantised *weight* can exploit.
    No host synchronisation, so a layer holding this on its critical path
    stays CUDA-graph capturable.

    Args:
        x: ``[..., K]`` float activations. Leading dims are flattened to rows.
            With ``fuse_silu_and_mul`` the row holds ``[gate | up]`` halves
            (``K`` even) and the quantised value is ``silu(gate) * up``.
        group_size: Elements per scale — a power of two that divides the
            (post-halving) row width; 128 matches the blockwise checkpoints.
        out_dtype: ``torch.int8``, or ``torch.uint8`` for fp8-e4m3 bytes (the
            bit-pattern convention every fp8 op in this package uses).
        column_major_scales: Store the ``[T, G]`` scale grid with token stride
            1 — the layout a TMA-based consumer reads it in. 2D inputs only.
        fuse_silu_and_mul: Quantise ``silu(x[:, :K//2]) * x[:, K//2:]`` straight
            from the gate/up buffer, saving the fp16 round trip through HBM a
            separate activation kernel would take before the quantiser.
        eps: Amax floor, so an all-zero group divides by ``eps/QMAX`` rather
            than zero (sglang bakes in ``1e-10``).

    Returns:
        ``(q, scales)``: ``q`` shaped like ``x`` (or ``[..., K/2]`` when
        fused) in ``out_dtype``, ``scales`` fp32 shaped ``[..., K/group_size]``
        — a transposed-stride view of it when ``column_major_scales``.
    """
    if out_dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"out_dtype must be int8 or uint8 (e4m3 bytes), got {out_dtype}")
    if group_size <= 0 or group_size & (group_size - 1):
        raise ValueError(f"group_size must be a power of two, got {group_size}")
    if not x.is_floating_point():
        raise ValueError(f"x must be a float tensor, got {x.dtype}")

    k = x.shape[-1]
    if fuse_silu_and_mul:
        if k % 2:
            raise ValueError(f"fuse_silu_and_mul needs an even row width, got {k}")
        h = k // 2
    else:
        h = k
    if h % group_size:
        raise ValueError(f"row width {h} is not a multiple of group_size {group_size}")
    if column_major_scales and x.dim() != 2:
        raise ValueError("column_major_scales supports 2D inputs only")

    qmax = FP8_E4M3_MAX if out_dtype is torch.uint8 else _INT8_MAX
    flat = x.reshape(-1, k)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    t = flat.shape[0]
    g = h // group_size

    q = torch.empty((t, h), device=x.device, dtype=out_dtype)
    if column_major_scales:
        # Allocated [G, T] and viewed transposed: the buffer stays what the
        # kernel wrote, the *shape* the caller sees is [T, G] with token
        # stride 1.
        scales = torch.empty((g, t), device=x.device, dtype=torch.float32).transpose(0, 1)
        out_scales = scales
    else:
        scales = torch.empty((t, g), device=x.device, dtype=torch.float32)
        out_scales = scales.reshape(*x.shape[:-1], g)

    if t == 0 or g == 0:
        return q.reshape(*x.shape[:-1], h), out_scales

    if out_dtype is torch.uint8 and not has_native_fp8(x.device.index):
        q, scales = _per_token_group_quant_torch(flat, h, group_size, fuse_silu_and_mul, qmax, eps)
        if column_major_scales:
            scales = scales.t().contiguous().t()
            return q.reshape(*x.shape[:-1], h), scales
        return q.reshape(*x.shape[:-1], h), scales.reshape(*x.shape[:-1], g)

    # Tile size: _QUANT_TILE elements split as [groups, group_size], capped by
    # however few groups a narrow row actually has. num_warps follows the
    # element count — 256 elements per warp is the same ratio the per-token
    # quantisers' 1024/4 uses.
    groups_per_prog = max(1, min(g, _QUANT_TILE // group_size))
    _per_token_group_quant_kernel[(t, triton.cdiv(g, groups_per_prog))](
        flat,
        q,
        scales,
        flat.stride(0),
        q.stride(0),
        scales.stride(0),
        scales.stride(1),
        g,
        h,
        GROUP_SIZE=group_size,
        GROUPS_PER_PROG=groups_per_prog,
        QMAX=qmax,
        OUT_FP8=out_dtype is torch.uint8,
        FUSE_SILU=fuse_silu_and_mul,
        EPS=eps,
        num_warps=max(1, (groups_per_prog * group_size) // 256),
    )
    return q.reshape(*x.shape[:-1], h), out_scales
