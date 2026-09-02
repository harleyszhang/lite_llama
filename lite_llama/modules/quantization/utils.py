"""Quantisation utilities: quantize helpers and checkpoint layout adapters.

The ``quantize_*`` functions produce each kernel's expected payload
(per-channel, per-token, group-wise int4/int8, fp8 blocks); the ``awq_*``
and ``adapt_*`` helpers convert published checkpoint layouts.

Usage:
    q = quantize_fp8_per_token(x)
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .base_config import QuantizationConfig

# =========================================================================== #
# FP8 quantisation helpers
# =========================================================================== #

#: Largest finite magnitude of the e4m3 format.
FP8_E4M3_MAX = 448.0


def _quantize_fp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale ``values`` into the e4m3 range and return the ``uint8`` bit pattern."""
    q = (values / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8)


def quantize_fp8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to fp8-e4m3 with per-output-channel scales.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]`` for stacked experts) float weights.

    Returns:
        ``(qweight, scales)`` where ``qweight`` is ``uint8`` holding the e4m3
        bit pattern and ``scales`` is ``[..., N, 1]`` fp32.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True).float() / FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return _quantize_fp8(weight.float(), scale), scale


def quantize_fp8_per_tensor(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantise ``x`` to fp8-e4m3 with one caller-supplied scalar scale.

    Args:
        x: fp16 values of any shape.
        scale: ``x / scale`` is what actually lands in e4m3.

    Returns:
        ``uint8`` tensor holding the e4m3 bit pattern, shaped like ``x``.
    """
    return _quantize_fp8(x.float(), torch.full((), scale, dtype=torch.float32, device=x.device))


def quantize_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise activations to fp8-e4m3, one scale per token (row).

    Args:
        x: ``[..., K]`` fp16/bf16 activations.

    Returns:
        ``(qx, scales)`` with ``qx`` the ``uint8`` e4m3 bit pattern shaped like
        ``x`` and ``scales`` ``[..., 1]`` fp32.
    """
    flat = x.reshape(-1, x.shape[-1]).float()
    scale = flat.abs().amax(dim=-1, keepdim=True) / FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qx = _quantize_fp8(flat, scale).reshape(x.shape)
    return qx, scale.reshape(*x.shape[:-1], 1)


# =========================================================================== #
# INT8 quantisation helpers
# =========================================================================== #


def quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to symmetric per-channel int8.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]`` for stacked experts) float weights.

    Returns:
        ``(qweight, scales)`` with ``scales`` shaped ``[..., N, 1]``.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True).float() / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qweight = (weight.float() / scale).round().clamp_(-127, 127).to(torch.int8)
    return qweight, scale


def quantize_int8_groupwise(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[..., K]`` fp16 weights to symmetric int8, one scale per group.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]``) float weights. K must be a
            multiple of ``group_size``.
        group_size: Input channels covered by one scale.

    Returns:
        ``(qweight, scales)`` with ``scales`` shaped ``[..., N, K//group_size]``.
    """
    k = weight.shape[-1]
    if k % group_size != 0:
        raise ValueError(f"in_features {k} must be a multiple of group_size {group_size}")
    w = weight.float().unflatten(-1, (k // group_size, group_size))
    scale = w.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qweight = (w / scale.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
    return qweight.flatten(-2), scale


def quantize_int8_groupwise_asym(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to asymmetric int8 (GPTQ ``bits=8``).

    Each group of ``group_size`` input channels gets its own fp32 scale and
    integer zero point, and the packed output stores 4 int8 values per int32
    word — the same container AutoGPTQ 8-bit checkpoints ship, so a runtime
    quantised weight and a loaded one are indistinguishable downstream of
    ``process_weights_after_loading``.

    Args:
        weight: ``[N, K]`` float weights. K must be a multiple of ``group_size``.
        group_size: Number of input channels per quantisation group.

    Returns:
        ``(qweight, scales, zeros)`` where ``qweight`` is ``[N, K//4]`` int32,
        and ``scales``/``zeros`` are ``[N, K//group_size]`` fp32 (``zeros``
        holding integer-valued zero points, typically in ``[-128, 127]`` but
        unclamped — see below).
    """
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"in_features {k} must be a multiple of group_size {group_size}")

    w = weight.float().reshape(n, k // group_size, group_size)
    w_min = w.amin(dim=-1)
    w_max = w.amax(dim=-1)

    # The full [-128, 127] range: asymmetric quantisation has 255 code levels
    # between min and max, where the symmetric form budgets 127 per side. The
    # zero point starts in the uint8 domain AutoGPTQ uses (``-min/scale`` lives
    # in [0, 255] for mixed-sign groups) and shifts by 128 into the int8
    # domain the kernels subtract in — ``(q_u - 128) - (z_u - 128) == q_u -
    # z_u``, so the shift only picks the container and leaves the dequant
    # arithmetic untouched.
    #
    # No clamp on the zero point, unlike the int4 form above: a group that is
    # entirely negative (or positive) has ``-w_min/scale > 255`` — the uint8
    # span covers ``w_max - w_min``, not ``|w_min|`` — so clamping would
    # corrupt the reconstruction. The fp32 ``weight_zeros`` container carries
    # the out-of-range point fine, and ``q = w/scale + z`` then lands back in
    # ``[-128, 127]`` anyway, which is the one range that *is* a container
    # constraint (the packed byte).
    qspan = 255.0
    scale = (w_max - w_min).clamp(min=1e-5) / qspan
    zero = torch.round(-w_min / scale) - 128
    q = (w / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int32)

    # Pack 4 int8 values per int32 word along the K dimension, byte j at bits
    # 8j — the little-endian order the int8 unpack kernel reverses. The
    # ``& 0xFF`` is load-bearing, unlike in the int4 pack above: a negative
    # byte's two's-complement sign bits survive a left shift (``-9 << 16``
    # has 0xFF in bits 24-31), so summing the raw words borrows a carry into
    # the next byte up. Masking first makes every addend a plain non-negative
    # bit pattern, and the modular wrap-around in the int32 sum then packs
    # each byte into its own lane.
    q = q.reshape(n, -1, 4)
    shifts = torch.arange(4, device=q.device, dtype=torch.int32) * 8
    packed = ((q & 0xFF) << shifts[None, None, :]).sum(dim=-1)  # [N, K//4]

    return packed.to(torch.int32), scale.float(), zero.float()


# =========================================================================== #
# INT4 quantisation helpers
# =========================================================================== #


def quantize_int4_groupwise(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to group-wise int4 (AWQ/GPTQ format).

    Each group of ``group_size`` input channels gets its own fp32 scale and zero
    point. The packed output stores 8 int4 values per int32 word.

    Args:
        weight: ``[N, K]`` float weights. K must be a multiple of ``group_size``.
        group_size: Number of input channels per quantisation group.

    Returns:
        ``(qweight, scales, zeros)`` where ``qweight`` is ``[N, K//8]`` int32,
        ``scales`` is ``[N, K//group_size]`` fp32, and ``zeros`` is the same
        shape as ``scales``.
    """
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"in_features {k} must be a multiple of group_size {group_size}")

    w = weight.float().reshape(n, k // group_size, group_size)
    w_min = w.amin(dim=-1)
    w_max = w.amax(dim=-1)

    qmax = 7.0  # int4 range: [-8, 7]
    scale = (w_max - w_min).clamp(min=1e-5) / (2 * qmax)
    zero = (-w_min / scale).round().clamp(0, 15)
    q = (w / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 15).to(torch.int32)

    # Pack 8 int4 values per int32 word along the K dimension.
    q = q.reshape(n, -1, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int32) * 4
    packed = (q << shifts[None, None, :]).sum(dim=-1)  # [N, K//8]

    return packed.to(torch.int32), scale.float(), zero.float()


# =========================================================================== #
# AWQ checkpoint layout adapter
# =========================================================================== #

#: AWQ's non-standard bit order inside an int32 word.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _awq_unpack(words: torch.Tensor) -> torch.Tensor:
    """``[..., W]`` int32 words -> ``[..., W, 8]`` nibbles, sequential bit order."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=words.device)
    return (words.unsqueeze(-1) >> shifts) & 0xF


def awq_qweight_to_canonical(qweight: torch.Tensor) -> torch.Tensor:
    """``[K, N//8]`` AWQ-packed -> ``[N, K//8]`` canonical-packed int32."""
    k, n_packed = qweight.shape
    n = n_packed * 8
    order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=qweight.device)
    q = _awq_unpack(qweight)[..., order].reshape(k, n)  # [K, N] channel values
    q = q.t().reshape(n, k // 8, 8)  # [N, K//8, 8]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    return (q << shifts).sum(dim=-1, dtype=torch.int32)


def awq_qzeros_to_canonical(qzeros: torch.Tensor) -> torch.Tensor:
    """``[G, N//8]`` AWQ-packed zero points -> ``[N, G]`` fp32."""
    g, n_packed = qzeros.shape
    order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=qzeros.device)
    z = _awq_unpack(qzeros)[..., order].reshape(g, n_packed * 8)  # [G, N]
    return z.t().float().contiguous()


def awq_scales_to_canonical(scales: torch.Tensor) -> torch.Tensor:
    """``[G, N]`` group scales -> ``[N, G]`` fp32."""
    return scales.t().float().contiguous()


def awq_adapt_key(key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    """Rename + relayout one AWQ checkpoint tensor to the canonical w4a16 names."""
    if key.endswith(".qweight"):
        return key.removesuffix(".qweight") + ".weight", awq_qweight_to_canonical(tensor)
    if key.endswith(".qzeros"):
        return key.removesuffix(".qzeros") + ".weight_zeros", awq_qzeros_to_canonical(tensor)
    if key.endswith(".scales"):
        return key.removesuffix(".scales") + ".weight_scale", awq_scales_to_canonical(tensor)
    return key, tensor


# =========================================================================== #
# GPTQ checkpoint layout adapter
# =========================================================================== #


def _gptq_unpack(words: torch.Tensor) -> torch.Tensor:
    """``[..., W]`` int32 words -> ``[..., W, 8]`` nibbles, sequential bit order."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=words.device)
    return (words.unsqueeze(-1) >> shifts) & 0xF


def _gptq_unpack_int8(words: torch.Tensor) -> torch.Tensor:
    """``[..., W]`` int32 words -> ``[..., W, 4]`` bytes, sequential bit order.

    The ``bits=8`` sibling of :func:`_gptq_unpack`: byte ``j`` at bits ``8j``.
    The mask matters — the top byte of a negative word sign-extends under
    torch's arithmetic ``>>``.
    """
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=words.device)
    return (words.unsqueeze(-1) >> shifts) & 0xFF


def gptq_qweight_to_canonical(qweight: torch.Tensor) -> torch.Tensor:
    """``[K//pack, N]`` GPTQ-packed -> ``[N, K//pack]`` canonical-packed int32.

    The transpose is bits-agnostic — eight nibbles (``bits=4``) or four bytes
    (``bits=8``) per word ride along the same layout flip untouched.
    """
    return qweight.t().contiguous()


def gptq_qzeros_to_canonical(qzeros: torch.Tensor) -> torch.Tensor:
    """``[G, N//8]`` GPTQ-packed zero points -> ``[N, G]`` fp32 (bias undone)."""
    z = _gptq_unpack(qzeros).reshape(qzeros.shape[0], -1)  # [G, N], still biased
    return (z + 1).t().float().contiguous()


def gptq_int8_qzeros_to_canonical(qzeros: torch.Tensor) -> torch.Tensor:
    """``[G, N//4]`` GPTQ-packed int8 zero points -> ``[N, G]`` fp32.

    The ``bits=8`` sibling of :func:`gptq_qzeros_to_canonical`: four biased
    bytes per word, same ``+1`` undo of AutoGPTQ's packing offset. The extra
    ``-128`` then shifts the zero point from the uint8 domain AutoGPTQ (and
    vLLM, whose containers are ``uint8``) quantises in into the int8 domain
    our kernels subtract in — the unpacked weight bytes made the same shift
    implicitly, and ``q - z`` is invariant to a shift both operands take, so
    the dequant arithmetic matches vLLM's exactly.
    """
    z = _gptq_unpack_int8(qzeros).reshape(qzeros.shape[0], -1)  # [G, N], still biased
    return (z + 1 - 128).t().float().contiguous()


def gptq_scales_to_canonical(scales: torch.Tensor) -> torch.Tensor:
    """``[G, N]`` group scales -> ``[N, G]`` fp32."""
    return scales.t().float().contiguous()


def gptq_adapt_key(
    key: str, tensor: torch.Tensor, bits: int = 4
) -> tuple[str, torch.Tensor] | None:
    """Rename + relayout one GPTQ checkpoint tensor to the canonical names.

    ``g_idx`` keys are dropped: ``desc_act=False`` checkpoints store groups in
    input order, making the index redundant. ``bits`` selects the zero-point
    unpack width — 4 nibbles-in-a-byte (``bits=4``) or whole bytes
    (``bits=8``); the qweight transpose and scale relayout are width-agnostic.
    """
    if key.endswith(".g_idx"):
        return None
    if key.endswith(".qweight"):
        return key.removesuffix(".qweight") + ".weight", gptq_qweight_to_canonical(tensor)
    if key.endswith(".qzeros"):
        zeros = (
            gptq_qzeros_to_canonical(tensor)
            if bits == 4
            else gptq_int8_qzeros_to_canonical(tensor)
        )
        return key.removesuffix(".qzeros") + ".weight_zeros", zeros
    if key.endswith(".scales"):
        return key.removesuffix(".scales") + ".weight_scale", gptq_scales_to_canonical(tensor)
    return key, tensor


# =========================================================================== #
# Checkpoint stream adapter (dispatches by quant_method)
# =========================================================================== #

#: Checkpoint ``quant_method`` -> its backend's key adapter.
_ADAPTERS = {"awq": awq_adapt_key, "gptq": gptq_adapt_key}


def _adapter_for(quant: QuantizationConfig):
    """The key adapter for ``quant``, with its config resolved into the call.

    GPTQ is the one adapter that needs more than the tensors: its zero-point
    unpack width is the checkpoint's ``bits``, so the ``gptq`` entry closes
    over it here rather than making every adapter take a config.
    """
    if quant.method == "gptq":
        bits = getattr(quant, "bits", 4)
        return lambda key, tensor: gptq_adapt_key(key, tensor, bits=bits)
    try:
        return _ADAPTERS[quant.method]
    except KeyError:
        raise ValueError(
            f"no checkpoint layout adapter for quant_method {quant.method!r}; "
            f"supported: {sorted(_ADAPTERS)}"
        ) from None


def adapt_packed_checkpoint(
    checkpoint: Iterable[tuple[str, torch.Tensor]], quant: QuantizationConfig
) -> Iterable[tuple[str, torch.Tensor]]:
    """Rewrite a packed (int4/int8) checkpoint stream to the canonical layout.

    The int4 layouts are the original reason this exists; GPTQ ``bits=8``
    checkpoints pack the same way (four bytes per word instead of eight
    nibbles) and cross the same bridge, which is why the trigger is the
    config's ``is_packed`` rather than ``is_int4``.

    Args:
        checkpoint: ``(key, tensor)`` pairs as yielded by
            :func:`lite_llama.executor.weight_utils.hf_weights_iterator`.
        quant: The checkpoint's quantisation config; ``quant.method`` selects
            the backend adapter.

    Raises:
        ValueError: If the checkpoint's ``quant_method`` has no adapter.
    """
    adapt = _adapter_for(quant)
    for key, tensor in checkpoint:
        out = adapt(key, tensor)
        if out is not None:
            yield out


#: Backward-compat alias: int4 was the only packed format when the adapter
#: landed, and the old name still says what half the call sites mean by it.
adapt_int4_checkpoint = adapt_packed_checkpoint
