"""Model configuration, backed by HuggingFace's ``AutoConfig``.

``read_model_type`` sniffs a checkpoint's model_type;
:class:`ModelConfig` wraps the HF config and derives the geometry
(heads, head_dim, rope settings) the runtime needs.

Usage:
    config = ModelConfig(hf_config, max_seq_len)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoConfig, PretrainedConfig

if TYPE_CHECKING:
    from ..modules.quantization import QuantizationConfig

#: KV-cache dtypes accepted by :attr:`ModelConfig.kv_cache_dtype` (vLLM spelling).
#: The e4m3 bytes travel in a ``uint8`` container; the decode kernel widens them.
#: ``"auto"`` is not in the map — it resolves to :attr:`ModelConfig.dtype` at
#: read time so the cache always matches the checkpoint's element type.
KV_CACHE_DTYPES: dict[str, torch.dtype] = {
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
}

#: Spellings ``torch_dtype`` may carry in a checkpoint's ``config.json``:
#: transformers 4.x writes a string, 5.x may already store a ``torch.dtype``.
_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def read_model_type(checkpoints_dir: str | Path) -> str:
    """Return a checkpoint's ``model_type`` without parsing its whole config.

    Callers that must know the architecture *before* building anything (the CLI's
    chat-template selection) need only this one field, and need it to fail cheaply
    on a directory that is not a checkpoint at all.

    Raises:
        FileNotFoundError: If there is no ``config.json``.
        ValueError: If ``config.json`` is malformed or declares no ``model_type``.
    """
    path = Path(checkpoints_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if "model_type" not in params:
        raise ValueError(f"{path} has no 'model_type' field")
    return str(params["model_type"]).lower()


class ModelConfig:
    """A HuggingFace config plus the runtime knobs and derived geometry.

    Attribute access falls through to the *text* config, so HF field names keep
    working (``config.rms_norm_eps``, ``config.num_experts``, ...) while the
    handful of names below are normalised. Multimodal wrappers read the full
    config through :attr:`hf_config` for their vision towers.

    Args:
        hf_config: The config as ``AutoConfig`` parsed it.
        max_seq_len: Context bound for this deployment; also caps the KV cache.

    Raises:
        ValueError: If the geometry cannot be served by the Triton kernels, or if
            ``max_seq_len`` exceeds what the checkpoint was trained for.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        max_seq_len: int = 2048,
        kv_cache_dtype: str = "auto",
    ) -> None:
        self.hf_config = hf_config
        # Vision-language configs nest the decoder under ``text_config``; text
        # models are their own text config.
        self.text_config: PretrainedConfig = getattr(hf_config, "text_config", hf_config)
        self.max_seq_len = max_seq_len
        # Element type of the paged KV cache. ``"auto"`` follows the checkpoint's
        # dtype (see :attr:`dtype`); the fp8 spellings store e4m3 bytes (in a
        # ``uint8`` container) and make the decode kernel dequantise on read,
        # halving the cache footprint.
        self.kv_cache_dtype = kv_cache_dtype
        # Weight format the *checkpoint* is stored in, which decides both what the
        # model allocates and whether the loader may widen anything on the way in.
        # A runtime ``--quantization`` request is a separate, post-load step.
        self.quant: QuantizationConfig | None = self._parse_quant(hf_config)
        self.validate()

    @staticmethod
    def _parse_quant(hf_config: PretrainedConfig):
        """Lazy-import quantization parsing to avoid circular imports."""
        from ..modules.quantization import get_quant_config_from_hf

        return get_quant_config_from_hf(hf_config)

    @classmethod
    def from_pretrained(
        cls,
        checkpoints_dir: str | Path,
        max_seq_len: int = 2048,
        kv_cache_dtype: str = "auto",
    ) -> ModelConfig:
        """Load ``config.json`` from a checkpoint directory through ``AutoConfig``."""
        hf_config = AutoConfig.from_pretrained(str(checkpoints_dir), trust_remote_code=True)
        return cls(hf_config, max_seq_len=max_seq_len, kv_cache_dtype=kv_cache_dtype)

    @property
    def dtype(self) -> torch.dtype:
        """Element type of the checkpoint's tensors; bf16 when undeclared.

        Every parameter and the KV cache (under ``"auto"``) are allocated in
        this dtype, so bf16 checkpoints finally stop being narrowed to fp16.
        A checkpoint that declares ``float32`` is *downgraded* to bf16: the
        Triton kernels accumulate in fp32 but only load 16-bit inputs, and a
        full-fp32 deployment was never possible in lite_llama anyway.
        """
        raw = getattr(self.text_config, "torch_dtype", None)
        if isinstance(raw, torch.dtype):
            return torch.bfloat16 if raw == torch.float32 else raw
        return _TORCH_DTYPES.get(str(raw), torch.bfloat16)

    @property
    def kv_cache_torch_dtype(self) -> torch.dtype:
        """Torch dtype of the KV-cache buffers, resolved from :attr:`kv_cache_dtype`."""
        if self.kv_cache_dtype == "auto":
            return self.dtype
        return KV_CACHE_DTYPES[self.kv_cache_dtype]

    # ---- identity --------------------------------------------------------- #
    @property
    def model_type(self) -> str:
        """``model_type`` of the *outer* config, i.e. the registry key."""
        return str(self.hf_config.model_type)

    # ---- normalised geometry ---------------------------------------------- #
    @property
    def num_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @property
    def num_heads(self) -> int:
        return int(self.text_config.num_attention_heads)

    @property
    def num_kv_heads(self) -> int:
        """Key/value head count; equals ``num_heads`` unless the model uses GQA."""
        return int(getattr(self.text_config, "num_key_value_heads", None) or self.num_heads)

    @property
    def head_dim(self) -> int:
        explicit = getattr(self.text_config, "head_dim", None)
        return int(explicit or self.hidden_size // self.num_heads)

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)

    @property
    def intermediate_size(self) -> int:
        return int(self.text_config.intermediate_size)

    @property
    def vocab_size(self) -> int:
        return int(self.text_config.vocab_size)

    @property
    def max_position_embeddings(self) -> int:
        return int(self.text_config.max_position_embeddings)

    # ---- MLA geometry (DeepSeek-V2/V3) ------------------------------------- #
    @property
    def kv_lora_rank(self) -> int | None:
        """Latent KV rank; ``None`` marks a per-head-K/V (MHA/GQA) model."""
        return getattr(self.text_config, "kv_lora_rank", None)

    @property
    def q_lora_rank(self) -> int | None:
        """Latent query rank; ``None`` when q is projected in one shot (V2-Lite)."""
        return getattr(self.text_config, "q_lora_rank", None)

    @property
    def qk_rope_head_dim(self) -> int | None:
        """Per-head width of the rotary (pe) slice of q/k under MLA."""
        return getattr(self.text_config, "qk_rope_head_dim", None)

    @property
    def qk_nope_head_dim(self) -> int | None:
        """Per-head width of the non-rotary slice of q/k under MLA."""
        return getattr(self.text_config, "qk_nope_head_dim", None)

    @property
    def v_head_dim(self) -> int | None:
        """Per-head value width under MLA, which may differ from the q/k width."""
        return getattr(self.text_config, "v_head_dim", None)

    @property
    def is_mla(self) -> bool:
        """Whether attention caches one latent vector per token instead of per-head K/V."""
        return self.kv_lora_rank is not None

    # ---- DeepSeek MoE geometry --------------------------------------------- #
    @property
    def n_shared_experts(self) -> int:
        """Experts every token passes through; 0 for a purely routed MoE."""
        return int(getattr(self.text_config, "n_shared_experts", 0) or 0)

    @property
    def routed_scaling_factor(self) -> float:
        """Scale applied to the routed experts' combined output."""
        return float(getattr(self.text_config, "routed_scaling_factor", 1.0))

    @property
    def first_k_dense_replace(self) -> int:
        """Leading decoder layers that use a dense MLP instead of MoE."""
        return int(getattr(self.text_config, "first_k_dense_replace", 0) or 0)

    @property
    def scoring_func(self) -> str:
        """Router scoring: ``softmax`` (V2) or ``sigmoid`` (V2.5+/V3)."""
        return str(getattr(self.text_config, "scoring_func", "softmax"))

    @property
    def tie_word_embeddings(self) -> bool:
        """Whether ``lm_head`` shares the embedding table (and is absent from the checkpoint)."""
        return bool(getattr(self.text_config, "tie_word_embeddings", False))

    @property
    def q_size(self) -> int:
        """Width of the query projection, which may differ from ``hidden_size``."""
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        """Width of one of the key/value projections."""
        return self.num_kv_heads * self.head_dim

    @property
    def kv_cache_row(self) -> tuple[int, int]:
        """Shape of one token's per-layer KV-cache row, for a world of one rank.

        ``(2 * num_kv_heads, head_dim)`` for MHA/GQA — K heads first, then V
        heads, so a decode step writes both halves in one launch — and
        ``(1, kv_lora_rank + qk_rope_head_dim)`` under MLA, whose latent row has
        no head axis to shard. A tensor-parallel rank caches only the KV heads
        it owns, so :class:`~lite_llama.executor.model_runner.ModelRunner`
        divides the head count itself before asking the same question.
        """
        if self.is_mla:
            # validate() guarantees both MLA dims exist once kv_lora_rank is set.
            assert self.kv_lora_rank is not None and self.qk_rope_head_dim is not None
            return (1, self.kv_lora_rank + self.qk_rope_head_dim)
        return (2 * self.num_kv_heads, self.head_dim)

    # ---- mixture of experts ------------------------------------------------ #
    @property
    def num_experts(self) -> int:
        """Number of routed experts, under either HF generation's spelling.

        ``n_routed_experts`` is what DeepSeek's config.json — and the
        remote-code class its checkpoints auto-map to — writes, while
        transformers' built-in configs expose the same count as
        ``num_experts`` (every other MoE family's spelling). The two never
        coexist in one config; carrying neither means not an MoE at all, which
        stays an error so a typo cannot silently drop the router.
        """
        for name in ("num_experts", "n_routed_experts"):
            value = getattr(self.text_config, name, None)
            if value is not None:
                return int(value)
        raise AttributeError(
            f"no routed-expert count under either spelling (num_experts, "
            f"n_routed_experts) on {type(self.text_config).__name__}"
        )

    # ---- positional encoding --------------------------------------------- #
    @property
    def rope_parameters(self) -> dict[str, Any]:
        """RoPE settings as transformers 5.x groups them.

        ``rope_theta`` and ``mrope_section`` live inside this dict in 5.x but were
        loose attributes in 4.x, so both layouts are merged here and every RoPE
        consumer reads one shape.
        """
        params = dict(
            getattr(self.text_config, "rope_parameters", None)
            or getattr(self.text_config, "rope_scaling", None)
            or {}
        )
        params.setdefault("rope_type", params.pop("type", "default"))
        theta = getattr(self.text_config, "rope_theta", None)
        if theta is not None:
            params.setdefault("rope_theta", theta)
        params.setdefault("rope_theta", 10000.0)
        return params

    @property
    def rope_config(self) -> dict[str, Any]:
        """Flat mapping consumed by :mod:`lite_llama.modules.rotary_embedding`."""
        return {
            # MLA rotates only the rope slice of q/k, so the rotary table
            # is built at the pe width, not at the attention head dim.
            "head_dim": self.qk_rope_head_dim if self.is_mla else self.head_dim,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "partial_rotary_factor": getattr(self.text_config, "partial_rotary_factor", 1.0),
            # Lets the RoPE layer precompute one (cos, sin) row per position;
            # validate() keeps this >= every position id the engine can feed.
            "max_seq_len": self.max_seq_len,
            # YaRN derives its factor from this when the checkpoint states the
            # extended length instead (DeepSeek-V3 spelling).
            "max_position_embeddings": self.max_position_embeddings,
            **self.rope_parameters,
        }

    # ---- validation ------------------------------------------------------- #
    def validate(self) -> None:
        """Reject configurations the kernels cannot serve."""
        accepted = {"auto", *KV_CACHE_DTYPES}
        if self.kv_cache_dtype not in accepted:
            raise ValueError(
                f"kv_cache_dtype must be one of {sorted(accepted)}, got {self.kv_cache_dtype!r}"
            )
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
        if self.is_mla:
            missing = [
                name
                for name in ("qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"MLA model (kv_lora_rank={self.kv_lora_rank}) is missing {missing}"
                )

    # ---- fall-through ----------------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the text config.

        Keeps model code reading HF field names directly — ``config.rms_norm_eps``,
        ``config.num_experts``, ``config.moe_intermediate_size`` — instead of
        re-declaring each one here. Only reached for names this class does not
        define, so the normalised properties above always win.
        """
        # ``__getattr__`` also runs before ``__init__`` has set ``text_config``
        # (e.g. while unpickling), so guard against recursing on a bare instance.
        if name.startswith("__") or "text_config" not in self.__dict__:
            raise AttributeError(name)
        text_config = self.__dict__["text_config"]
        try:
            return getattr(text_config, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}, and neither does "
                f"{type(text_config).__name__}"
            ) from None

    def __repr__(self) -> str:
        return (
            f"ModelConfig(model_type={self.model_type!r}, num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, hidden_size={self.hidden_size}, "
            f"vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len})"
        )
