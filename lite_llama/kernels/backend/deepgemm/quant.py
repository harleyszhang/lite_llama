"""Per-token-group fp8 quantisation for the DeepGEMM GEMMs.

:func:`per_token_group_quant_fp8` scales activations group-wise into
e4m3; the block/NT helpers convert checkpoint weights to the layout
DeepGEMM expects, and back again for reference checks.

Usage:
    qx, scale = per_token_group_quant_fp8(x)
"""

from __future__ import annotations

import torch

#: Largest finite e4m3 value; scales normalise by amax / FP8_E4M3_MAX.
FP8_E4M3_MAX = 448.0

#: DeepGEMM's group size — 128 on both the activation and the weight side.
GROUP_SIZE = 128


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[m, k]`` activations per token per ``group_size`` group.

    Returns:
        ``(x_fp8, scales)`` with ``x_fp8`` ``[m, k]`` e4m3 and ``scales``
        ``[m, k // group_size]`` fp32 — DeepGEMM's activation operand pair.
        All-zero rows quantise to zero with a clamped scale, never NaN.
    """
    if x.dim() != 2:
        raise ValueError(f"per_token_group_quant_fp8 wants [m, k], got {tuple(x.shape)}")
    m, k = x.shape
    if k % group_size:
        raise ValueError(f"DeepGEMM needs k % {group_size} == 0, got k={k}")
    groups = x.reshape(m, k // group_size, group_size)
    amax = groups.abs().amax(dim=-1)
    scales = (amax / FP8_E4M3_MAX).clamp(min=1e-12).float()
    q = (groups / scales.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return q.reshape(m, k).to(torch.float8_e4m3fn), scales


def block_quant_fp8_nt(
    weight: torch.Tensor,
    group_n: int = GROUP_SIZE,
    group_k: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise a bf16/fp16 ``[n, k]`` weight into DeepGEMM's NT form.

    Returns:
        ``(w_fp8, scales)`` with ``w_fp8`` ``[k, n]`` row-major e4m3 — the
        ``[n, k]`` column-major NT operand — and ``scales``
        ``[n // group_n, k // group_k]`` fp32, entry ``[i, j]`` covering rows
        ``i*group_n .. i*group_n+group_n`` by columns ``j*group_k ..``.
    """
    n, k = weight.shape
    if n % group_n or k % group_k:
        raise ValueError(
            f"DeepGEMM needs weight dims multiples of ({group_n}, {group_k}), got {(n, k)}"
        )
    w = weight.t().contiguous().float()  # [k, n]
    # (k // gk, gk, n // gn, gn) — one 128x128 block per (i, j) cell.
    blocks = w.reshape(k // group_k, group_k, n // group_n, group_n)
    amax = blocks.abs().amax(dim=(1, 3))  # [kG, nG]
    scales = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
    q = (blocks / scales[:, None, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = q.reshape(k, n).to(torch.float8_e4m3fn)
    # DeepGEMM indexes weight scales [n // gn, k // gk] — the transpose of ours.
    return w_fp8, scales.t().contiguous()


def dequant_fp8_blocks(
    qweight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    group_n: int,
    group_k: int,
) -> torch.Tensor:
    """Dequantise a native fp8 checkpoint weight to bf16.

    ``qweight`` is ``[*, n, k]`` uint8 e4m3 bit patterns;
    ``weight_scale`` is ``[*, ceil(n / group_n), ceil(k / group_k)]`` fp32 in
    the native convention (``LinearOp.weight_scale``), where ``group_n == 1``
    means per-output-channel and ``group_k >= k`` one scale per row. The
    broadcast below handles every granularity without padding assumptions.
    """
    n, k = qweight.shape[-2:]
    w = qweight.view(torch.float8_e4m3fn).to(torch.bfloat16).float()
    if weight_scale is None:
        return w.to(torch.bfloat16)
    gn = group_n if group_n and group_n < n else n
    gk = group_k if group_k and group_k < k else k
    scale = weight_scale.float()
    # Repeat each block scale over the rows/columns it covers, tolerating a
    # ragged final block (ceil-shaped scales).
    scale = torch.repeat_interleave(scale, gn, dim=-2)[..., :n, :]
    scale = torch.repeat_interleave(scale, gk, dim=-1)[..., :, :k]
    return (w * scale).to(torch.bfloat16)


def nt_block_fp8_from_checkpoint(
    qweight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    group_n: int,
    group_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a native fp8 checkpoint weight into DeepGEMM's NT operands.

    ``qweight`` is ``[*, n, k]`` uint8 e4m3 bit patterns with the native block
    scales. Two outcomes, depending on the checkpoint's granularity:

    * exactly 128x128 blockwise — the bits and scales are already what
      DeepGEMM wants, so the weight is only transposed into NT layout and
      the scales pass through untouched (no requantisation error);
    * anything finer or coarser — the weight is dequantised to bf16 and
      requantised 128x128, trading one extra rounding for the granularity
      the kernel requires.

    Returns:
        ``(w_fp8, scales)`` — ``[*, k, n]`` e4m3 (NT) and
        ``[*, n // 128, k // 128]`` fp32, matching the per-expert stacked
        operands of the grouped GEMM when ``qweight`` is 3-D.
    """
    lead_shape = qweight.shape[:-2]
    n, k = qweight.shape[-2:]
    if (
        weight_scale is not None
        and group_n == GROUP_SIZE
        and group_k == GROUP_SIZE
        and n % GROUP_SIZE == 0
        and k % GROUP_SIZE == 0
        and weight_scale.shape == (*lead_shape, n // GROUP_SIZE, k // GROUP_SIZE)
    ):
        w_fp8 = qweight.view(torch.float8_e4m3fn).transpose(-1, -2).contiguous()
        return w_fp8, weight_scale.float()

    # Any other granularity: dequantise, then requantise blockwise per expert.
    bf16 = dequant_fp8_blocks(qweight, weight_scale, group_n, group_k)
    flat = bf16.reshape(-1, n, k)
    w_list, s_list = [], []
    for expert in range(flat.shape[0]):
        w_nt, s = block_quant_fp8_nt(flat[expert])
        w_list.append(w_nt)
        s_list.append(s)
    w_fp8 = w_list[0].unsqueeze(0) if not lead_shape else torch.stack(w_list)
    scales = s_list[0].unsqueeze(0) if not lead_shape else torch.stack(s_list)
    return w_fp8, scales
