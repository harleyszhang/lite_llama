"""AWQ checkpoint layout -> the canonical w4a16 layout.

An HF AutoAWQ checkpoint packs differently from what the w4a16 kernel
expects:

* ``qweight``: ``[K, N//8]`` int32 — 8 *output* channels per word along dim 1,
  in the interleaved bit order ``[0, 4, 1, 5, 2, 6, 3, 7]``.
* ``qzeros``: ``[G, N//8]`` int32 — same interleaved packing over outputs.
* ``scales``: ``[G, N]`` fp16.

Canonical (what ``w4a16_matmul`` consumes): ``qweight`` ``[N, K//8]`` int32
packed along the *input* dim in sequential bit order, plus fp32 ``zeros`` and
``scales`` of shape ``[N, G]``. All conversions are exact bit rearrangements
— no requantisation, so no accuracy is lost on the way in.

AWQ dequantises as ``w = (q - zero) * scale`` with the zero point stored
directly (no bias, unlike GPTQ's ``zero - 1`` convention).
"""

from __future__ import annotations

import torch

#: AWQ's non-standard bit order inside an int32 word: the value at bit
#: position ``4*i`` is not channel ``i``. Indexing with this permutation puts
#: the nibbles back in channel order.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _unpack(words: torch.Tensor) -> torch.Tensor:
    """``[..., W]`` int32 words -> ``[..., W, 8]`` nibbles, sequential bit order."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=words.device)
    return (words.unsqueeze(-1) >> shifts) & 0xF


def qweight_to_canonical(qweight: torch.Tensor) -> torch.Tensor:
    """``[K, N//8]`` AWQ-packed -> ``[N, K//8]`` canonical-packed int32."""
    k, n_packed = qweight.shape
    n = n_packed * 8
    order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=qweight.device)
    q = _unpack(qweight)[..., order].reshape(k, n)  # [K, N] channel values
    q = q.t().reshape(n, k // 8, 8)  # [N, K//8, 8]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    return (q << shifts).sum(dim=-1, dtype=torch.int32)


def qzeros_to_canonical(qzeros: torch.Tensor) -> torch.Tensor:
    """``[G, N//8]`` AWQ-packed zero points -> ``[N, G]`` fp32."""
    g, n_packed = qzeros.shape
    order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=qzeros.device)
    z = _unpack(qzeros)[..., order].reshape(g, n_packed * 8)  # [G, N]
    return z.t().float().contiguous()


def scales_to_canonical(scales: torch.Tensor) -> torch.Tensor:
    """``[G, N]`` group scales -> ``[N, G]`` fp32."""
    return scales.t().float().contiguous()


def adapt_key(key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    """Rename + relayout one checkpoint tensor to the canonical w4a16 names.

    Tensors that do not belong to a quantised projection (fp16 ``weight`` of
    an ignored module, norms, biases) pass through unchanged.
    """
    if key.endswith(".qweight"):
        return key.removesuffix(".qweight") + ".weight", qweight_to_canonical(tensor)
    if key.endswith(".qzeros"):
        return key.removesuffix(".qzeros") + ".weight_zeros", qzeros_to_canonical(tensor)
    if key.endswith(".scales"):
        return key.removesuffix(".scales") + ".weight_scale", scales_to_canonical(tensor)
    return key, tensor
