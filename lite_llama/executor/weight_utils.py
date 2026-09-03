"""Reading HuggingFace checkpoint files into a stream of ``(key, tensor)`` pairs.

:func:`hf_weights_iterator` walks safetensors / ``pytorch_model*.bin`` files
in checkpoint order, optionally dequantising block-fp8 weights on the fly,
so loaders never care which on-disk format a checkpoint uses.

Usage:
    for key, tensor in hf_weights_iterator(checkpoints_dir, device):
        sink[key] = tensor
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import torch

from ..modules.quantization import FP8_BLOCK, SCALE_SUFFIX
from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Suffix of the per-block scale table that accompanies an FP8 weight.
_SCALE_SUFFIX = "." + SCALE_SUFFIX


def hf_weight_files(checkpoints_dir: str | Path) -> list[Path]:
    """Return the weight files of a HuggingFace checkpoint directory.

    safetensors wins when both formats are present (HF repos that still carry legacy
    ``.bin`` mirrors).

    Raises:
        FileNotFoundError: If the directory holds no recognised weight file.
    """
    root = Path(checkpoints_dir)
    shards = sorted(root.glob("*.safetensors"))
    if shards:
        return shards
    shards = sorted(root.glob("*.bin"))
    if shards:
        return shards
    raise FileNotFoundError(
        f"no *.safetensors or *.bin weight file in {root}; point --model-dir at a "
        "HuggingFace checkpoint directory (the one holding config.json)"
    )


def dequant_block_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantise a block-wise FP8 (e4m3) matrix: ``W[i,j] = w8[i,j] * s[i//B, j//B]``.

    The multiply runs in fp32; casting fp8 first is exact (every e4m3 value is
    representable in fp32), so accuracy is governed solely by the final cast to *dtype*.
    """
    w = weight.to(torch.float32)
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    # The trailing block is partial when a dimension is not a multiple of 128.
    scale = scale[: w.shape[0], : w.shape[1]]
    return (w * scale).to(dtype)


def hf_weights_iterator(
    checkpoints_dir: str | Path,
    device: str | torch.device = "cpu",
    dequantize_fp8: bool = True,
    dequant_dtype: torch.dtype = torch.bfloat16,
    key_filter: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream ``(key, tensor)`` pairs from a checkpoint, tensors already on ``device``.

    Args:
        checkpoints_dir: Directory holding the ``*.safetensors`` (or ``*.bin``) shards.
        device: Where tensors land; the compute device lets the caller copy straight
            into its parameters and makes FP8 dequantisation a GPU op.
        dequantize_fp8: Widen FP8 weights to 16-bit and consume their scales, or hand
            both through untouched for a w8a16 model.
        dequant_dtype: Element type widened FP8 weights land in (matches the parameters
            the loader copies into).
        key_filter: Predicate applied to the key *before* the tensor is read. A caller
            wanting one layer of a sharded checkpoint pays for that layer, not the whole
            model (shards are mmap'd, an unread tensor never leaves the file); filtering
            afterwards would not (the read and H2D copy already happened).

    Yields:
        Pairs in shard order.
    """
    files = hf_weight_files(checkpoints_dir)
    logger.info("Loading weights from %d file(s) in %s", len(files), checkpoints_dir)
    for path in files:
        if path.suffix == ".safetensors":
            yield from _iter_safetensors(path, device, dequantize_fp8, dequant_dtype, key_filter)
        else:
            yield from _iter_torch_bin(path, device, dequantize_fp8, dequant_dtype, key_filter)


def _iter_safetensors(
    path: Path,
    device: str | torch.device,
    dequantize_fp8: bool,
    dequant_dtype: torch.dtype,
    key_filter: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as shard:
        # safetensors >= 0.6 dropped iteration on the handle; keys() works everywhere.
        keys = set(shard.keys())
        for key in sorted(keys):
            if key_filter is not None and not key_filter(key):
                continue
            if key.endswith(_SCALE_SUFFIX):
                if dequantize_fp8:
                    continue  # consumed alongside its weight below
                yield key, shard.get_tensor(key).to(device)
                continue
            tensor = shard.get_tensor(key)
            if key.removesuffix(".weight") + _SCALE_SUFFIX in keys:
                tensor = (
                    dequant_block_fp8(
                        tensor.to(device),
                        shard.get_tensor(key.removesuffix(".weight") + _SCALE_SUFFIX).to(device),
                        dequant_dtype,
                    )
                    if dequantize_fp8
                    # Reinterpret, not convert: Ampere cannot compute on fp8, and the
                    # w8a16 kernel widens the raw bytes itself.
                    else tensor.view(torch.uint8)
                )
            yield key, tensor.to(device)


def _iter_torch_bin(
    path: Path,
    device: str | torch.device,
    dequantize_fp8: bool,
    dequant_dtype: torch.dtype,
    key_filter: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    # mmap keeps the shard out of the resident set; weights_only rejects the
    # arbitrary-code-execution pickle payloads a downloaded .bin could carry.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    scales = {k: v for k, v in state.items() if k.endswith(_SCALE_SUFFIX)}
    for key, tensor in state.items():
        if key_filter is not None and not key_filter(key):
            continue
        if key.endswith(_SCALE_SUFFIX):
            if not dequantize_fp8:
                yield key, tensor.to(device)
            continue
        scale = scales.get(key.removesuffix(".weight") + _SCALE_SUFFIX)
        if scale is not None:
            tensor = (
                dequant_block_fp8(tensor.to(device), scale.to(device), dequant_dtype)
                if dequantize_fp8
                else tensor.view(torch.uint8)
            )
        yield key, tensor.to(device)
