"""Single source of truth mapping a HuggingFace ``model_type`` to its implementation.

Replaces a mapping once duplicated across the runner, a config table and a
forward-time ``if model_type in (...)`` branch. Each entry is two facts: the class
(``"module.path:ClassName"``, imported lazily so a missing transformers model only
fails when requested) and whether it consumes ``multi_modal_inputs`` (the executor
must know before it holds a model — it decides CUDA-graph eligibility). Adding a
model is one line in ``_SPECS`` plus the implementation module.

Usage:
    ModelRegistry maps a model_type to a ModelSpec (class path + multimodal flag).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import ClassVar

import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """How to serve one architecture.

    Attributes:
        model_type: The ``model_type`` string found in ``config.json``.
        implementation: ``"module.path:ClassName"``, imported on first use.
        is_multimodal: Whether the model accepts ``multi_modal_inputs``.
    """

    model_type: str
    implementation: str
    is_multimodal: bool = False

    def load_class(self) -> type[nn.Module]:
        """Import and return the implementation class."""
        module_name, class_name = self.implementation.split(":")
        return getattr(importlib.import_module(module_name), class_name)


class ModelRegistry:
    """Registry of supported architectures, keyed by ``model_type``."""

    _SPECS: ClassVar[tuple[ModelSpec, ...]] = (
        ModelSpec("llama", "lite_llama.models.llama:LlamaModel"),
        ModelSpec("qwen2", "lite_llama.models.qwen2:Qwen2Model"),
        ModelSpec("qwen3", "lite_llama.models.qwen3:Qwen3Model"),
        ModelSpec("qwen3_moe", "lite_llama.models.qwen3_moe:Qwen3MoeModel"),
        ModelSpec("llava", "lite_llama.models.llava:LlavaLlama", is_multimodal=True),
        ModelSpec("qwen3_vl", "lite_llama.models.qwen3_vl:Qwen3VLForCausalLM", is_multimodal=True),
    )

    _BY_TYPE: ClassVar[dict[str, ModelSpec]] = {spec.model_type: spec for spec in _SPECS}

    @classmethod
    def supported_types(cls) -> list[str]:
        return sorted(cls._BY_TYPE)

    @classmethod
    def resolve(cls, model_type: str) -> ModelSpec:
        """Look up a spec, raising a message that lists the alternatives."""
        spec = cls._BY_TYPE.get(model_type.lower())
        if spec is None:
            raise ValueError(
                f"unsupported model_type {model_type!r}; "
                f"supported: {', '.join(cls.supported_types())}"
            )
        return spec
