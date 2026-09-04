"""One-time int8 weight preprocessing: int32 words to bytes.

GPTQ ``bits=8`` checkpoints (and ``quantize_int8_groupwise_asym``) pack four
int8 values per ``int32`` word along K; every kernel here — the fused MoE
grouped GEMM's int8 mode and the dense ``w8a16_matmul`` — consumes one ``int8``
byte per element instead, so stacked expert weights cross this bridge once at
load, in
:meth:`~rapid_llm.modules.quantization.base_config.QuantizeMethodBase.process_weights_after_loading`.

Usage:
    kernel_w = unpack_int8_experts(checkpoint_w)  # [E, N, K//4] -> [E, N, K]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_unpack_kernel(
    src_ptr,
    dst_ptr,
    num_words,
    BLOCK_W: tl.constexpr,
):
    """One program unpacks ``BLOCK_W`` int32 words into ``4 * BLOCK_W`` bytes.

    Word ``i`` holds the int8 values of K indices ``[4i, 4i+3]`` (byte ``j`` at
    bits ``8j``, the little-endian packing AutoGPTQ and
    ``quantize_int8_groupwise_asym`` share). Byte ``4i + j`` takes that value's
    bit pattern unchanged, so a negative weight keeps its two's-complement byte
    and the dequantisation arithmetic — group scales, zero points — is bit-exact
    against the same values unpacked by torch.

    ``& 0xFF`` before the store is load-bearing: torch-style arithmetic ``>>``
    sign-extends, so without the mask a top byte at bits 24-31 of a negative
    word would arrive with 24 one-bits above it.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs < num_words
    w = tl.load(src_ptr + offs, mask=mask, other=0)  # [BLOCK_W] int32

    shifts = (tl.arange(0, 4) * 8).to(tl.int32)
    out = (w[:, None] >> shifts[None, :]) & 0xFF  # [BLOCK_W, 4], 0..255

    offs_byte = offs[:, None] * 4 + tl.arange(0, 4)[None, :]
    # uint8 first, then the bit-preserving hop to int8: a plain ``.to(tl.int8)``
    # on the 0..255 lane truncates identically, but spelling the bitcast keeps
    # the "value in, same bit pattern out" contract obvious.
    tl.store(dst_ptr + offs_byte, out.to(tl.uint8).to(tl.int8, bitcast=True), mask=mask[:, None])


def unpack_int8_experts(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K//4]`` int32 (4 int8 per word) to ``[..., K]`` int8.

    A pure layout change: every byte keeps its bit pattern and its K index, so
    group scales and zero points apply unchanged and kernel outputs stay
    bit-identical. Leading dims are preserved, so the same op serves stacked
    MoE experts ``[E, N, K//4]`` and a dense ``[N, K//4]`` weight.

    Args:
        packed: int32 tensor whose last dim packs 4 int8 values per word (the
            ``quantize_int8_groupwise_asym`` / GPTQ-8bit layout).

    Returns:
        int8 tensor with identical leading dims and a last dim 4x longer.
    """
    if packed.dtype != torch.int32:
        raise ValueError(f"packed int8 weights must be int32, got {packed.dtype}")
    if not packed.is_contiguous():
        packed = packed.contiguous()
    words = packed.reshape(-1)
    num_words = words.numel()
    out_shape = (*packed.shape[:-1], packed.shape[-1] * 4)
    if num_words == 0:
        return torch.empty(out_shape, dtype=torch.int8, device=packed.device)
    out = torch.empty(num_words * 4, dtype=torch.int8, device=packed.device)
    # 1024 words = 4 KB read and 4 KB written per program, matching
    # ``repack_int4_experts``: the op is one-time and bandwidth-bound, so the
    # block size is not tuned per shape.
    BLOCK_W = 1024
    grid = (triton.cdiv(num_words, BLOCK_W),)
    _int8_unpack_kernel[grid](words, out, num_words, BLOCK_W=BLOCK_W, num_warps=4)
    return out.reshape(out_shape)
