"""Single source of truth mapping a HuggingFace ``model_type`` to its implementation.

Before this module the mapping was duplicated in three places — a string-path table
in the model runner, a config-class table beside the attention metadata, and an
``if model_type in ("llava", "qwen3_vl")`` branch in the forward call — which meant
adding a model required edits in three files that could silently disagree.

A registry entry is now two facts, and nothing else:

* which class implements the architecture, written as ``"module.path:ClassName"``
  so the import is deferred — a transformers build without ``qwen3_vl`` only
  fails if that model is actually requested;
* whether the model consumes ``multi_modal_inputs``, which the executor has to
  know before it holds a model, because it decides CUDA-graph eligibility and the
  shape of the forward call.

Config parsing used to live here too, as one loader function per architecture.
It moved to :class:`~lite_llama.models.config.ModelConfig`, which reads every
checkpoint through ``AutoConfig``, so there is nothing left to vary per model.

Adding a model is therefore one line in :attr:`ModelRegistry._SPECS` plus the
implementation module.
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
