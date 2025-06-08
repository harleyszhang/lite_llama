"""
Refactored model configuration dataclasses plus unit‑tests.

Usage
-----
pytest -q
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Type, TypeVar, Optional
import json
import os

T = TypeVar("T", bound="BaseConfig")


# ----------------------------------------------------------------------------- #
#                               Utility helpers                                 #
# ----------------------------------------------------------------------------- #

def _apply_aliases(
    raw: Mapping[str, Any], aliases: Mapping[str, str]
) -> dict[str, Any]:
    """Return a shallow‑copied dict with alias keys renamed."""
    out: dict[str, Any] = dict(raw)
    for old, new in aliases.items():
        if old in out:
            out[new] = out.pop(old)
    return out


def _filter_fields(data: Mapping[str, Any], cls: Type) -> dict[str, Any]:
    """Drop keys that are not declared in *cls* dataclass fields."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


# ----------------------------------------------------------------------------- #
#                            BaseConfig definition                              #
# ----------------------------------------------------------------------------- #
@dataclass
class BaseConfig:
    """Minimal base class providing *from_dict* utility."""

    # subclasses may override
    _ALIASES: Mapping[str, str] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
        # apply alias mapping declared on the concrete subclass
        aliased = _apply_aliases(data, getattr(cls, "_ALIASES", {}))
        # only keep valid fields
        return cls(**_filter_fields(aliased, cls))  # type: ignore[arg-type]

    # pretty repr truncated for long sequences
    def __repr__(self) -> str:  # pragma: no cover
        cls = self.__class__.__name__
        parts = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{cls}({parts})"


# ----------------------------------------------------------------------------- #
#                               Model configs                                   #
# ----------------------------------------------------------------------------- #
@dataclass
class LlamaConfig(BaseConfig):
    # architecture‑specific
    architectures: list[str] = field(default_factory=lambda: ["LlamaForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    hidden_size: int = 2048
    intermediate_size: Optional[int] = None  # default handled in post‑init
    max_position_embeddings: Optional[int] = None
    mlp_bias: bool = False
    model_type: str = "llama"
    num_heads: int = 32
    num_layers: int = 32
    num_kv_heads: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[dict[str, Any]] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: Optional[str] = None
    use_cache: bool = True
    vocab_size: int = 32064
    _name_or_path: Optional[str] = None
    max_batch_size: int = 64
    max_seq_len: int = 2048
    device: str = "cuda"

    # alias mapping used by `from_dict`
    _ALIASES = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_kv_heads",
        "max_length": "max_seq_len",
    }

    # ----------------------------- validation -------------------------------- #
    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads

        assert (
            self.hidden_size == self.head_dim * self.num_heads
        ), "hidden_size must equal num_heads × head_dim"


# ---------------------------------------------------------------------------- #
@dataclass
class Qwen2Config(BaseConfig):
    max_batch_size: int = 4
    max_seq_len: int = 2048
    architectures: Optional[list[str]] = None
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151645
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    hidden_size: int = 1536
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 32768
    mlp_bias: bool = False
    model_type: str = "qwen2"
    num_heads: int = 12
    num_layers: int = 28
    num_kv_heads: Optional[int] = 2
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict[str, Any]] = None
    rope_theta: float = 1_000_000.0
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.43.1"
    use_cache: bool = True
    vocab_size: int = 151_936
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 21
    device: str = "cuda"
    head_dim: Optional[int] = None

    _ALIASES = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_kv_heads",
        "max_length": "max_seq_len",
    }

    def __post_init__(self) -> None:
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size == self.head_dim * self.num_heads


# ---------------------------------------------------------------------------- #
@dataclass
class Qwen3Config(BaseConfig):
    vocab_size: int = 151_936
    hidden_size: int = 4096
    intermediate_size: Optional[int] = None
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = 32
    head_dim: Optional[int] = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32_768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict[str, Any]] = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    device: str = "cuda"
    model_type: str = "qwen3"
    max_seq_len: int = 2048

    _ALIASES = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_kv_heads",
        "max_length": "max_seq_len",
    }

    def __post_init__(self) -> None:
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size == self.head_dim * self.num_heads


# ---------------------------------------------------------------------------- #
@dataclass
class VisionConfig(BaseConfig):
    hidden_size: int = 768
    image_size: int = 224
    intermediate_size: int = 3072
    model_type: str = "clip_vision_model"
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    patch_size: int = 16
    projection_dim: int = 768
    vocab_size: int = 1000


# ---------------------------------------------------------------------------- #
@dataclass
class LlavaConfig(BaseConfig):
    architectures: list[str]
    ignore_index: int
    image_token_index: int
    model_type: str
    pad_token_id: int
    projector_hidden_act: str
    text_config: LlamaConfig
    tie_word_embeddings: bool
    torch_dtype: str
    vision_config: VisionConfig
    vision_feature_layer: int
    vision_feature_select_strategy: str
    vocab_size: int
    image_seq_length: int = 576
    max_batch_size: int = 64
    max_seq_len: int = 2048
    device: str = "cuda"

    # no aliases ‑ relies on static *from_dict* below.

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LlavaConfig":
        text_cfg = LlamaConfig.from_dict(data.get("text_config", {}))
        vision_cfg = VisionConfig.from_dict(data.get("vision_config", {}))

        # retain only valid primitive fields *excluding* the nested configs
        kwargs = _filter_fields(data, LlavaConfig)
        kwargs.pop("text_config", None)
        kwargs.pop("vision_config", None)

        # supply defaults for mandatory fields if absent
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("torch_dtype", "float16")

        return LlavaConfig(text_config=text_cfg, vision_config=vision_cfg, **kwargs)

    @classmethod
    def from_json(cls, json_path: os.PathLike | str) -> "LlavaConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ----------------------------------------------------------------------------- #
#                                    Tests                                      #
# ----------------------------------------------------------------------------- #

def _make_fake_json(tmp_path) -> str:
    path = tmp_path / "llava.json"
    sample = {
        "architectures": ["LlavaForConditionalGeneration"],
        "ignore_index": -1,
        "image_token_index": 32000,
        "model_type": "llava",
        "pad_token_id": 32001,
        "projector_hidden_act": "gelu",
        "text_config": {"hidden_size": 1024, "num_attention_heads": 8},
        "vision_config": {"hidden_size": 384},
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "vocab_size": 32064,
    }
    path.write_text(json.dumps(sample))
    return str(path)


def test_llama_default():
    cfg = LlamaConfig()
    assert cfg.head_dim == cfg.hidden_size // cfg.num_heads
    assert cfg.intermediate_size == cfg.hidden_size * 4


def test_llama_from_alias():
    cfg = LlamaConfig.from_dict({"num_attention_heads": 16, "hidden_size": 1024})
    assert cfg.num_heads == 16
    assert cfg.head_dim == 64


def test_qwen2_sliding_window_disabled():
    cfg = Qwen2Config(use_sliding_window=False)
    assert cfg.sliding_window is None


def test_qwen3_valid_head_dim():
    cfg = Qwen3Config(head_dim=None, hidden_size=2048, num_heads=16)
    assert cfg.head_dim == 128


def test_llava_roundtrip(tmp_path):
    json_path = _make_fake_json(tmp_path)
    cfg = LlavaConfig.from_json(json_path)
    assert cfg.text_config.hidden_size == 1024
    assert cfg.vision_config.hidden_size == 384
