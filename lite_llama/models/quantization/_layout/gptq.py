"""GPTQ checkpoint layout -> the canonical w4a16 layout.

An HF AutoGPTQ checkpoint (``desc_act=False``) already packs in sequential
bit order, so the conversion is cheaper than AWQ's:

* ``qweight``: ``[K//8, N]`` int32 — 8 *input* channels per word along dim 0,
  exactly the canonical packing; only the word matrix needs transposing.
* ``qzeros``: ``[G, N//8]`` int32 — sequential packing over outputs, and each
  stored value is the zero point *minus one* (GPTQ's convention, so the real
  zero point of a symmetric group encodes as ``8 - 1 = 7``).
* ``scales``: ``[G, N]`` fp16.
* ``g_idx``: per-column group indices. Dropped: with ``desc_act=False`` the
  groups are in input order, so the index carries no information.

Canonical (what ``w4a16_matmul`` consumes): ``qweight`` ``[N, K//8]`` int32,
plus fp32 ``zeros`` and ``scales`` of shape ``[N, G]``.
"""

from __future__ import annotations

import torch


def _unpack(words: torch.Tensor) -> torch.Tensor:
    """``[..., W]`` int32 words -> ``[..., W, 8]`` nibbles, sequential bit order."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=words.device)
    return (words.unsqueeze(-1) >> shifts) & 0xF


def qweight_to_canonical(qweight: torch.Tensor) -> torch.Tensor:
    """``[K//8, N]`` GPTQ-packed -> ``[N, K//8]`` canonical-packed int32.

    Both layouts pack 8 consecutive input channels per word in sequential bit
    order, so the words themselves are unchanged — only transposed.
    """
    return qweight.t().contiguous()


def qzeros_to_canonical(qzeros: torch.Tensor) -> torch.Tensor:
    """``[G, N//8]`` GPTQ-packed zero points -> ``[N, G]`` fp32 (bias undone)."""
    z = _unpack(qzeros).reshape(qzeros.shape[0], -1)  # [G, N], still biased
    return (z + 1).t().float().contiguous()


def scales_to_canonical(scales: torch.Tensor) -> torch.Tensor:
    """``[G, N]`` group scales -> ``[N, G]`` fp32."""
    return scales.t().float().contiguous()


def adapt_key(key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    """Rename + relayout one checkpoint tensor to the canonical w4a16 names.

    ``g_idx`` keys are dropped (``None``): ``desc_act=False`` checkpoints
    store groups in input order, making the index redundant. Tensors that do
    not belong to a quantised projection pass through unchanged.
    """
    if key.endswith(".g_idx"):
        return None
    if key.endswith(".qweight"):
        return key.removesuffix(".qweight") + ".weight", qweight_to_canonical(tensor)
    if key.endswith(".qzeros"):
        return key.removesuffix(".qzeros") + ".weight_zeros", qzeros_to_canonical(tensor)
    if key.endswith(".scales"):
        return key.removesuffix(".scales") + ".weight_scale", scales_to_canonical(tensor)
    return key, tensor
