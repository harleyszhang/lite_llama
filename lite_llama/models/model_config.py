"""Dataclass model configurations for the decoder-only models lite_llama serves.

HuggingFace ``config.json`` files use their own field names (``num_attention_heads``,
``num_hidden_layers``, ...). :meth:`BaseConfig.from_dict` renames those through a
per-class alias table and drops anything the dataclass does not declare, so a raw
HF config can be loaded without pre-processing.

Note on ``head_dim``: it is *not* always ``hidden_size // num_heads``. Qwen3-0.6B,
for example, has ``hidden_size=1024`` but ``num_heads=16`` and ``head_dim=128``, so
the attention projections are wider than the residual stream. Code must therefore
use :attr:`TextModelConfig.q_size` for the query projection width and
``hidden_size`` only for the residual stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

T = TypeVar("T", bound="BaseConfig")

# HF -> lite_llama field renames shared by every text model config.
_COMMON_ALIASES: Mapping[str, str] = {
    "num_attention_heads": "num_heads",
    "num_hidden_layers": "num_layers",
    "num_key_value_heads": "num_kv_heads",
    "max_length": "max_seq_len",
}


@dataclass
class BaseConfig:
    """Provides alias-aware, unknown-key-tolerant construction from a mapping."""

    _ALIASES: Mapping[str, str] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any], **overrides: Any) -> T:
        """Build the config from a (HF-style) mapping.

        Args:
            data: Raw config mapping; unknown keys are ignored.
            overrides: Values applied after ``data``, used for runtime knobs such
                as ``max_seq_len`` that do not come from ``config.json``.
        """
        renamed = dict(data)
        for old, new in getattr(cls, "_ALIASES", {}).items():
            if old in renamed:
                renamed[new] = renamed.pop(old)
        renamed.update(overrides)

        declared = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in renamed.items() if k in declared})

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{type(self).__name__}({body})"


@dataclass
class TextModelConfig(BaseConfig):
    """Fields common to every decoder-only text model supported here."""

    # ---- architecture ---------------------------------------------------- #
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int | None = None
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int | None = None
    head_dim: int | None = None
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5

    # ---- positional encoding --------------------------------------------- #
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    partial_rotary_factor: float = 1.0

    # ---- tokenizer / misc metadata --------------------------------------- #
    architectures: list[str] | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    model_type: str = "llama"
    torch_dtype: str = "float16"
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True

    # ---- runtime knobs (not part of HF config.json) ---------------------- #
    max_seq_len: int = 2048
    max_batch_size: int = 64
    device: str = "cuda"

    _ALIASES = _COMMON_ALIASES

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        self.validate()

    def validate(self) -> None:
        """Reject configurations the kernels cannot serve."""
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for grouped-query attention"
            )
        if self.head_dim <= 0 or self.head_dim % 8 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be a positive multiple of 8 "
                "for the Triton attention kernels"
            )
        if self.max_seq_len > self.max_position_embeddings:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) exceeds the model's "
                f"max_position_embeddings ({self.max_position_embeddings})"
            )

    # ---- derived sizes ---------------------------------------------------- #
    @property
    def q_size(self) -> int:
        """Width of the query projection, which may differ from ``hidden_size``."""
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        """Width of one of the key/value projections."""
        return self.num_kv_heads * self.head_dim


@dataclass
class LlamaConfig(TextModelConfig):
    model_type: str = "llama"
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False
    pretraining_tp: int = 1


@dataclass
class Qwen2Config(TextModelConfig):
    """Qwen2 keeps a bias on the q/k/v projections but not on o_proj."""

    model_type: str = "qwen2"
    vocab_size: int = 151_936
    hidden_size: int = 1536
    num_heads: int = 12
    num_layers: int = 28
    num_kv_heads: int | None = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32_768
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 21

    def __post_init__(self) -> None:
        if not self.use_sliding_window:
            self.sliding_window = None
        super().__post_init__()


@dataclass
class Qwen3Config(TextModelConfig):
    """Qwen3 adds per-head RMSNorm on q and k, and decouples head_dim from hidden_size."""

    model_type: str = "qwen3"
    vocab_size: int = 151_936
    hidden_size: int = 1024
    num_heads: int = 16
    num_layers: int = 28
    num_kv_heads: int | None = 8
    head_dim: int | None = 128
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40_960
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28

    def __post_init__(self) -> None:
        if not self.use_sliding_window:
            self.sliding_window = None
        super().__post_init__()
