"""Model loading: a HuggingFace checkpoint in, a ready-to-run ``nn.Module`` out.

Mirrors vLLM's executor/loader split so loading is unit-testable and open to new
sources (TP shards, remote stores). It never holds a second copy of the model:
build the tree on ``meta`` (no alloc, no init), swap in real fp16 storage on the
target device, then stream the checkpoint through ``model.load_weights`` — a copy
loop (not ``load_state_dict``) because tensors land in the *middle* of fused K/V
and stacked-expert parameters. There is deliberately no trailing ``half()``.

Usage:
    model = DefaultModelLoader().load_model(config, model_cls, device)
"""

from __future__ import annotations

import contextlib
import time
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from ..models.config import ModelConfig
from ..models.quantization import RUNTIME_SCHEMES, QuantConfig, RawParameter
from ..utils.logger import get_logger
from .weight_utils import hf_weights_iterator

logger = get_logger(__name__)

#: Every lite_llama parameter is fp16: that is what the Triton kernels are
#: written for, and checkpoints are cast on copy.
PARAM_DTYPE = torch.float16


@contextlib.contextmanager
def init_empty_parameters():
    """Skeleton context: parameters allocate on the meta device, buffers do not.

    Mirrors ``accelerate.init_empty_weights(include_buffers=False)``. Buffers must
    keep real storage because non-persistent ones such as
    :attr:`~lite_llama.models.rotary_embedding.RotaryEmbedding.inv_freq` are absent
    from checkpoints and are computed, not loaded.
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


def materialise_parameters(
    model: nn.Module, device: str | torch.device, dtype: torch.dtype = PARAM_DTYPE
) -> None:
    """Give every meta parameter real, uninitialised storage on ``device``.

    Only floating-point parameters are cast to ``dtype``; integer ones (a
    handful of HF vision towers keep index tables as parameters) keep theirs, and
    so do the quantisation parameters marked
    :class:`~lite_llama.models.quantization.RawParameter` — an 8-bit weight must
    not be widened, and its fp32 scales must not be narrowed.
    The storage stays uninitialised because :func:`load_weights` overwrites all
    of it and verifies that it did.

    ``requires_grad=False`` is load-bearing, not just tidy: copying into a leaf
    tensor that requires grad is an in-place autograd error, and the copy is how
    the fused parameters get filled.
    """
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is None:
                continue
            keep_dtype = isinstance(param, RawParameter) or not param.is_floating_point()
            module._parameters[name] = type(param)(
                torch.empty(param.shape, dtype=param.dtype if keep_dtype else dtype, device=device),
                requires_grad=False,
            )


@runtime_checkable
class ModelLoader(Protocol):
    """Strategy seam: anything that can turn a checkpoint dir into a model."""

    def load_model(
        self,
        config: ModelConfig,
        model_cls: type[nn.Module],
        checkpoints_dir: str,
        device: str,
        quantization: str | None = None,
    ) -> nn.Module: ...


class DefaultModelLoader:
    """Loads HuggingFace weights directly, with no offline conversion step."""

    def load_model(
        self,
        config: ModelConfig,
        model_cls: type[nn.Module],
        checkpoints_dir: str,
        device: str,
        quantization: str | None = None,
    ) -> nn.Module:
        """Build ``model_cls`` and fill it from the checkpoint in ``checkpoints_dir``.

        Args:
            config: Parsed configuration, already carrying ``max_seq_len`` and the
                checkpoint's own weight format (``config.quant``).
            model_cls: The implementation class resolved from the registry.
            checkpoints_dir: HuggingFace checkpoint directory (``config.json`` plus
                ``*.safetensors``).
            device: Torch device string the model must end up on.
            quantization: Post-load quantisation to apply to an fp16 checkpoint
                (see :data:`~lite_llama.models.quantization.RUNTIME_SCHEMES`),
                or ``None``. Ignored for a checkpoint that is already quantised,
                which needs no conversion.
        """
        start = time.time()
        self._check_device(device)

        logger.info("Building %s skeleton on the meta device", model_cls.__name__)
        with init_empty_parameters():
            model = model_cls(config)
        materialise_parameters(model, device)

        # An already-quantised checkpoint is copied in byte for byte; only an
        # fp16 model wants the FP8 blocks widened on the way through.
        model.load_weights(
            hf_weights_iterator(checkpoints_dir, device, dequantize_fp8=config.quant is None)
        )
        if quantization and config.quant is None:
            self._quantize(model, quantization)
        # Buffers were computed on the CPU while the skeleton was on meta.
        model.to(device).eval()
        logger.info("Loaded %s onto %s in %.2fs", model_cls.__name__, device, time.time() - start)
        return model

    @staticmethod
    def _quantize(model: nn.Module, quantization: str) -> None:
        """Apply a runtime quantisation request to a freshly loaded fp16 model.

        Raises:
            ValueError: On an unrecognised scheme name.
        """
        try:
            quant = QuantConfig.for_runtime_scheme(quantization)
        except ValueError:
            raise ValueError(
                f"cannot quantise an fp16 checkpoint to {quantization!r} at load time; "
                f"supported schemes: {sorted(RUNTIME_SCHEMES)}, or point --model-dir "
                "at a checkpoint that ships pre-quantised weights"
            ) from None
        if not hasattr(model, "quantize_"):
            raise ValueError(f"{type(model).__name__} does not support runtime quantisation")
        logger.info("Quantising weights to %s (%s, %sx%s scale blocks)",
                    quantization, quant.format, quant.group_n, quant.group_k)
        model.quantize_(quant)

    @staticmethod
    def _check_device(device: str) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            # Copying into cuda parameters would otherwise fail deep inside torch
            # with a message that says nothing about drivers.
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "This usually means the installed torch build targets a newer CUDA "
                f"than the NVIDIA driver on this machine. Installed: torch=={torch.__version__} "
                f"(cuda={torch.version.cuda}). Fix by installing a torch build that matches "
                "the local driver, e.g. `uv pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124`."
            )
