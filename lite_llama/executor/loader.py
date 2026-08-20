"""Model loading strategies.

Mirrors vLLM's split between the executor and its ``ModelLoader``: the
executor decides *what* to build and *when*; the loader owns *how* weights
travel from a checkpoint on disk into a ready-to-run ``nn.Module``. The seam
keeps loading unit-testable without an executor and leaves room for extra
sources (direct safetensors, tensor-parallel shards) without touching the
executor.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from ..models.registry import ModelRegistry, ModelSpec
from ..utils.logger import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def _init_empty_parameters():
    """Skeleton context: parameters allocate on the meta device, buffers do not.

    Mirrors ``accelerate.init_empty_weights(include_buffers=False)``. Buffers must
    keep real storage because non-persistent buffers such as
    :attr:`~lite_llama.models.rotary_embedding.RotaryEmbedding.inv_freq` are absent
    from checkpoints and therefore cannot be materialised by ``load_state_dict``.
    """
    original = nn.Module.register_parameter

    def register_meta_parameter(module: nn.Module, name: str, param) -> None:
        original(module, name, param)
        if module._parameters.get(name) is None:
            return
        # Preserve the Parameter subclass and its attributes (e.g. `requires_grad`).
        existing = module._parameters[name]
        kwargs = existing.__dict__
        module._parameters[name] = type(existing)(existing.to(torch.device("meta")), **kwargs)

    try:
        nn.Module.register_parameter = register_meta_parameter
        yield
    finally:
        nn.Module.register_parameter = original


@runtime_checkable
class ModelLoader(Protocol):
    """Strategy seam: anything that can turn a checkpoint dir into a model."""

    def load_model(
        self, config: Any, spec: ModelSpec, checkpoints_dir: str, device: str
    ) -> nn.Module: ...


class DefaultModelLoader:
    """Build the skeleton on the meta device, then assign real fp16 weights.

    Weight loading uses ``torch.device("meta")`` for the empty skeleton (no
    ``accelerate`` dependency) and relies on ``load_state_dict(assign=True)``
    to replace the meta parameters with real tensors mmap-loaded from disk.
    """

    def load_model(
        self, config: Any, spec: ModelSpec, checkpoints_dir: str, device: str
    ) -> nn.Module:
        start = time.time()
        self._check_device(device)

        logger.info(
            "Initializing model of type '%s' and moving it to device '%s'...",
            spec.model_type,
            device,
        )
        with _init_empty_parameters():
            model = ModelRegistry.build_model(config, spec)
        logger.info("The model has been initialized and moved to the device. '%s'", device)

        state_dict = self._load_state_dict(checkpoints_dir, device)
        # Models whose submodule layout depends on the installed transformers
        # version (e.g. LLaVA's CLIP vision tower) normalise checkpoint keys here.
        remap = getattr(model, "remap_checkpoint_keys", None)
        if callable(remap):
            state_dict = remap(state_dict)
        # assign=True swaps the meta params for the loaded tensors instead of copying.
        model.load_state_dict(state_dict, strict=True, assign=True)

        model.eval().to(device)
        for name, param in model.named_parameters():
            if param.is_meta:
                raise RuntimeError(
                    f"parameter {name!r} was not materialised from the checkpoint"
                )
        logger.info("Loaded state dict in %.2fs", time.time() - start)

        # The converter stores fp16 weights; half() is a no-op that verifies it.
        model.half()
        for param in model.parameters():
            if param.dtype != torch.float16:
                raise RuntimeError(
                    f"expected fp16 parameters after half(), got {param.dtype}"
                )
        logger.info("Converted model to half precision (FP16)")
        return model

    @staticmethod
    def _check_device(device: str) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            # ``torch.load(..., map_location="cuda")`` would otherwise fail deep
            # inside pickle with a message that says nothing about drivers.
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "This usually means the installed torch build targets a newer CUDA "
                f"than the NVIDIA driver on this machine. Installed: torch=={torch.__version__} "
                f"(cuda={torch.version.cuda}). Fix by installing a torch build that matches "
                "the local driver, e.g. `uv pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124`."
            )

    @staticmethod
    def _load_state_dict(checkpoints_dir: str, device: str) -> dict[str, torch.Tensor]:
        checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                f"no *.pth checkpoint found in {checkpoints_dir}; run "
                "`lite-llama-convert` on the HuggingFace weights first"
            )
        logger.info('Loading checkpoint "%s"', checkpoints[0])
        return torch.load(checkpoints[0], mmap=True, weights_only=True, map_location=device)
