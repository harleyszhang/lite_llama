"""Single source of truth mapping a HuggingFace ``model_type`` to its implementation.

Before this module the mapping was duplicated in three places — a string-path table
in the executor, a config-class table in ``executor_struct``, and an
``if model_type in ("llava", "qwen3_vl")`` branch in the forward call — which meant
adding a model required edits in three files that could silently disagree.

A :class:`ModelSpec` now carries everything the rest of the codebase needs to know
about a model: how to parse its config, which class to instantiate, and whether it
consumes multimodal inputs. Implementation modules are imported lazily so that a
transformers build without, say, ``qwen3_vl`` only fails if that model is requested.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch.nn as nn

from .model_config import LlamaConfig, Qwen2Config, Qwen3Config, Qwen3MoeConfig, TextModelConfig


def _text_config_loader(
    config_cls: type[TextModelConfig],
) -> Callable[[Mapping[str, Any], int], Any]:
    """Build a loader that turns a raw HF config mapping into a dataclass config."""

    def load(params: Mapping[str, Any], max_seq_len: int) -> TextModelConfig:
        return config_cls.from_dict(params, max_seq_len=max_seq_len)

    return load


def _hf_config_loader(qualified_name: str) -> Callable[[Mapping[str, Any], int], Any]:
    """Build a loader for models whose config stays a HuggingFace config object.

    Vision-language configs nest a text config and carry vision fields that the
    lite_llama dataclasses do not model, so the HF object is kept as-is and
    ``max_seq_len`` is attached to it for the executor and the model to read.
    """

    def load(params: Mapping[str, Any], max_seq_len: int) -> Any:
        module_name, class_name = qualified_name.rsplit(".", 1)
        config_cls = getattr(importlib.import_module(module_name), class_name)
        config = config_cls.from_dict(dict(params))
        config.max_seq_len = max_seq_len
        return config

    return load


@dataclass(frozen=True)
class ModelSpec:
    """Everything the framework needs in order to serve one architecture.

    Attributes:
        model_type: The ``model_type`` string found in ``config.json``.
        config_loader: Turns a raw config mapping plus ``max_seq_len`` into a config.
        implementation: ``"module.path:ClassName"``, imported on first use.
        is_multimodal: Whether the model accepts ``multi_modal_inputs``.
    """

    model_type: str
    config_loader: Callable[[Mapping[str, Any], int], Any]
    implementation: str
    is_multimodal: bool = False

    def load_class(self) -> type[nn.Module]:
        module_name, class_name = self.implementation.split(":")
        return getattr(importlib.import_module(module_name), class_name)


class ModelRegistry:
    """Registry of supported architectures, keyed by ``model_type``."""

    _specs: ClassVar[dict[str, ModelSpec]] = {}

    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        cls._specs[spec.model_type] = spec

    @classmethod
    def supported_types(cls) -> list[str]:
        return sorted(cls._specs)

    @classmethod
    def resolve(cls, model_type: str) -> ModelSpec:
        """Look up a spec, raising a message that lists the alternatives."""
        spec = cls._specs.get(model_type.lower())
        if spec is None:
            raise ValueError(
                f"unsupported model_type {model_type!r}; "
                f"supported: {', '.join(cls.supported_types())}"
            )
        return spec

    @classmethod
    def load_config(cls, checkpoints_dir: str | Path, max_seq_len: int) -> tuple[Any, ModelSpec]:
        """Read ``config.json`` from a checkpoint directory and build its config.

        Returns:
            ``(config, spec)``.
        """
        config_path = Path(checkpoints_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found")

        params = json.loads(config_path.read_text(encoding="utf-8"))
        if "model_type" not in params:
            raise ValueError(f"{config_path} has no 'model_type' field")

        spec = cls.resolve(params["model_type"])
        return spec.config_loader(params, max_seq_len), spec

    @classmethod
    def build_model(cls, config: Any, spec: ModelSpec) -> nn.Module:
        return spec.load_class()(config)


ModelRegistry.register(
    ModelSpec("llama", _text_config_loader(LlamaConfig), "lite_llama.models.llama:LlamaModel")
)
ModelRegistry.register(
    ModelSpec("qwen2", _text_config_loader(Qwen2Config), "lite_llama.models.qwen2:Qwen2Model")
)
ModelRegistry.register(
    ModelSpec("qwen3", _text_config_loader(Qwen3Config), "lite_llama.models.qwen3:Qwen3Model")
)
ModelRegistry.register(
    ModelSpec(
        "qwen3_moe",
        _text_config_loader(Qwen3MoeConfig),
        "lite_llama.models.qwen3_moe:Qwen3MoeModel",
    )
)
ModelRegistry.register(
    ModelSpec(
        "llava",
        _hf_config_loader("transformers.LlavaConfig"),
        "lite_llama.models.llava:LlavaLlama",
        is_multimodal=True,
    )
)
ModelRegistry.register(
    ModelSpec(
        "qwen3_vl",
        _hf_config_loader("transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig"),
        "lite_llama.models.qwen3_vl:Qwen3VLForCausalLM",
        is_multimodal=True,
    )
)
