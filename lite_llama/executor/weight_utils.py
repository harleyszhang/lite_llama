"""Reading HuggingFace checkpoint files.

One job: turn the files in a checkpoint directory into a stream of
``(key, tensor)`` pairs on the target device. safetensors shards are opened once
and read lazily so a 30B checkpoint never has to exist in host RAM as a whole
state dict.

Block-FP8 checkpoints (Qwen3-30B-A3B-…-FP8) are dequantised here rather than by
the model, because "the weights are e4m3 plus a scale table" is a property of the
file, not of the architecture. The dequantisation runs on the target device: on
the CPU it dominated load time for a 30B checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Block size of the fine-grained FP8 format used by Qwen FP8 checkpoints:
#: ``weight`` is e4m3 and ``weight_scale_inv[i, j]`` scales the 128x128 block
#: starting at ``(i * 128, j * 128)``.
FP8_BLOCK = 128

#: Suffix of the per-block scale table that accompanies an FP8 weight.
_SCALE_SUFFIX = ".weight_scale_inv"


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


def dequant_block_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantise a block-wise FP8 (e4m3) matrix: ``W[i,j] = w8[i,j] * s[i//B, j//B]``.

    The multiply runs in fp32; casting the fp8 values first is exact (every e4m3
    value is representable in fp32), so accuracy is governed solely by the final
    cast to fp16.
    """
    w = weight.to(torch.float32)
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    # The trailing block is partial when a dimension is not a multiple of 128.
    scale = scale[: w.shape[0], : w.shape[1]]
    return (w * scale).to(torch.float16)


def hf_weights_iterator(
    checkpoints_dir: str | Path, device: str | torch.device = "cpu"
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream ``(key, tensor)`` pairs from a checkpoint, tensors already on ``device``.

    Args:
        checkpoints_dir: Directory holding the ``*.safetensors`` (or ``*.bin``)
            shards.
        device: Where the tensors land. Passing the compute device lets the
            caller copy straight into its parameters and makes the FP8
            dequantisation a GPU op.

    Yields:
        Pairs in shard order. FP8 weights are yielded dequantised to fp16 and
        their ``*.weight_scale_inv`` partners are consumed, not yielded.
    """
    files = hf_weight_files(checkpoints_dir)
    logger.info("Loading weights from %d file(s) in %s", len(files), checkpoints_dir)
    for path in files:
        if path.suffix == ".safetensors":
            yield from _iter_safetensors(path, device)
        else:
            yield from _iter_torch_bin(path, device)


def _iter_safetensors(path: Path, device: str | torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as shard:
        # safetensors >= 0.6 dropped iteration on the handle; keys() works everywhere.
        keys = set(shard.keys())
        for key in sorted(keys):
            if key.endswith(_SCALE_SUFFIX):
                continue  # consumed alongside its weight below
            tensor = shard.get_tensor(key).to(device)
            scale_key = key.removesuffix(".weight") + _SCALE_SUFFIX
            if scale_key in keys:
                tensor = dequant_block_fp8(tensor, shard.get_tensor(scale_key).to(device))
            yield key, tensor


def _iter_torch_bin(path: Path, device: str | torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    # mmap keeps the shard out of the process's resident set; weights_only rejects
    # the arbitrary-code-execution pickle payloads a downloaded .bin could carry.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    scales = {k: v for k, v in state.items() if k.endswith(_SCALE_SUFFIX)}
    for key, tensor in state.items():
        if key.endswith(_SCALE_SUFFIX):
            continue
        tensor = tensor.to(device)
        scale = scales.get(key.removesuffix(".weight") + _SCALE_SUFFIX)
        if scale is not None:
            tensor = dequant_block_fp8(tensor, scale.to(device))
        yield key, tensor
