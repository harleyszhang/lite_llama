"""Reading HuggingFace checkpoint files into a stream of ``(key, tensor)`` pairs.

safetensors shards are opened once and read lazily, so a 30B checkpoint never
exists in host RAM as a whole state dict.

Block-FP8 checkpoints (Qwen3-30B-A3B-FP8) can be read two ways, and which one the
loader picks is the difference between a model that fits and one that does not:

* ``dequantize=True`` widens each e4m3 block to fp16 here, on the target device
  where the GPU is ~30x faster than the CPU (~2 s vs ~56 s for 30B). The model
  then holds plain fp16 weights, at twice the memory.
* ``dequantize=False`` passes the raw bytes through as ``uint8`` and yields the
  ``*.weight_scale_inv`` tables alongside them, for a model whose layers are
  themselves 8-bit (:mod:`lite_llama.models.quantization`).

Usage:
    for key, tensor in hf_weights_iterator(checkpoints_dir, device="cuda"):
        ...
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

    safetensors wins when both formats are present, which is what HF repos that
    still carry legacy ``.bin`` mirrors look like.

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

    The multiply runs in fp32; casting the fp8 values first is exact (every e4m3
    value is representable in fp32), so accuracy is governed solely by the final
    cast to *dtype*.
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
        checkpoints_dir: Directory holding the ``*.safetensors`` (or ``*.bin``)
            shards.
        device: Where the tensors land. Passing the compute device lets the
            caller copy straight into its parameters and makes the FP8
            dequantisation a GPU op.
        dequantize_fp8: Whether to widen FP8 weights to 16-bit and consume their
            scales, or hand both through untouched for a w8a16 model.
        dequant_dtype: Element type widened FP8 weights land in; matches the
            parameters the loader is about to copy into.
        key_filter: Optional predicate applied to the key *before* the tensor is
            read. A caller that wants one layer out of a sharded checkpoint
            (:mod:`lite_llama.tools.harness`) then pays for that layer rather
            than for the whole model, because the shards are memory-mapped and an
            unread tensor never leaves the file. Filtering afterwards would not:
            the read and the host-to-device copy have already happened by then.

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
                    # Reinterpret rather than convert: Ampere cannot compute on
                    # fp8, and the w8a16 kernel widens the raw bytes itself.
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
    # mmap keeps the shard out of the process's resident set; weights_only rejects
    # the arbitrary-code-execution pickle payloads a downloaded .bin could carry.
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
