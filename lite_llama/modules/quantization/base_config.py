"""Quantisation base classes (mirrors sglang ``base_config.py``).

Three-layer method hierarchy:
    QuantizeMethodBase          — abstract root (create_weights + apply)
    ├── LinearMethodBase        — linear layer strategy
    └── FusedMoEMethodBase      — stacked expert strategy

QuantizationConfig is the per-checkpoint-format class (one subclass per
precision: Fp8Config, W8A8Fp8Config, AWQConfig …). It owns the registry
dispatch (``get_quant_method``) and the layout metadata used by linear
layers and the weight loader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    pass


# --------------------------------------------------------------------------- #
# Method base classes
# --------------------------------------------------------------------------- #

class QuantizeMethodBase(ABC):
    """Base class for all quantised (or unquantised) method strategies."""

    def create_weights(
        self, layer: nn.Module, *weight_args, **extra_weight_attrs
    ) -> None:
        """Allocate weight parameters on *layer*.

        The method decides what tensors exist (packed weight, scales, zeros)
        and registers them as attributes of *layer*.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        """Execute the quantised (or plain) computation on *layer*."""
        raise NotImplementedError()

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Post-load hook (e.g. transpose, repack). Default: no-op."""
        return

    def quantize_from_fp16(self, layer: nn.Module, config: "QuantizationConfig") -> None:
        """Convert a loaded fp16 weight to the quantised form, in-place.

        Used by the ``--quantization`` runtime path.
        Raises NotImplementedError if this scheme requires a pre-quantised
        checkpoint (e.g. AWQ/GPTQ).
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be computed from fp16 weights at load time"
        )


class LinearMethodBase(QuantizeMethodBase):
    """Strategy interface for a quantised linear layer."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        **extra_weight_attrs,
    ) -> None:
        """Allocate the weight (and scale/zeros) for a linear with given sizes."""
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ``x @ W.T (+ bias)`` with the layer's stored weight."""
        raise NotImplementedError()


class FusedMoEMethodBase(QuantizeMethodBase):
    """Strategy interface for stacked-expert grouped GEMM."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        """Allocate stacked expert tensors for *block*."""
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        block: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the routed grouped GEMM over the block's expert weights."""
        raise NotImplementedError()


# --------------------------------------------------------------------------- #
# QuantizationConfig ABC
# --------------------------------------------------------------------------- #

class QuantizationConfig(ABC):
    """Per-checkpoint-format configuration (one subclass per precision).

    Each concrete config (Fp8Config, AWQConfig, …) is registered in
    ``BASE_QUANTIZATION_METHODS`` and looked up by the checkpoint's
    ``quant_method`` string or the ``--quantization`` CLI flag.

    The config answers two questions:
    1. What quant-method object does a given layer get?  → ``get_quant_method``
    2. How is the quantised weight laid out?             → layout properties
    """

    # -- Subclass must set or override ----------------------------------------
    #: Block granularity of the quantisation scale grid.
    group_n: int = 1
    group_k: int = 1 << 30
    #: HF module names that the checkpoint left unquantised.
    ignored: tuple[str, ...] = ()
    #: Whether activations are quantised at runtime (W8A8 variants).
    is_dynamic: bool = False
    #: Checkpoint-specific method name (``"awq"``/``"gptq"``), if applicable.
    method: str = ""

    # -- Abstract interface ---------------------------------------------------
    @abstractmethod
    def get_name(self) -> str:
        """Short identifier (used in logs / error messages)."""
        raise NotImplementedError()

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """Activation dtypes this config can handle."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU SM capability (e.g. 80 for Ampere)."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        """Factory: build from HF ``quantization_config`` dict."""
        raise NotImplementedError()

    @abstractmethod
    def get_quant_method(
        self, layer: nn.Module, prefix: str = ""
    ) -> QuantizeMethodBase | None:
        """Return the method strategy for *layer* (or None to skip).

        When ``prefix`` matches one of the ``ignored`` names the config should
        return an unquantised method (or None).
        """
        raise NotImplementedError()

    # -- Layout helpers (shared implementation) --------------------------------
    @property
    @abstractmethod
    def storage_dtype(self) -> torch.dtype:
        """Container dtype of the packed weight."""
        raise NotImplementedError()

    def scale_shape(self, out_features: int, in_features: int) -> tuple[int, ...]:
        """Scale-grid shape for a ``[out_features, in_features]`` weight."""
        return (
            (out_features + self.group_n - 1) // self.group_n,
            (in_features + self.group_k - 1) // self.group_k,
        )

    def shard_is_aligned(self, size: int) -> bool:
        """Whether a TP shard of ``size`` channels keeps whole scale blocks."""
        if self.group_n <= 1 and self.group_k >= (1 << 30):
            return True
        return size % max(self.group_n, self.group_k) == 0

    def quantizes(self, module_name: str) -> bool:
        """Whether ``module_name`` (HF-style path) is quantised by this config."""
        return not any(
            module_name == ign or module_name.startswith(ign + ".")
            for ign in self.ignored
        )

    # -- Convenience static helpers (sglang compat) ---------------------------
    @property
    def is_fp8(self) -> bool:
        """True if this is an fp8 weight format."""
        return False

    @property
    def is_int4(self) -> bool:
        """True if this is an int4 weight format (AWQ/GPTQ)."""
        return False

    @property
    def format(self) -> str:
        """Backward-compat alias for :meth:`get_name`."""
        return self.get_name()

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in quantization config.")

    @staticmethod
    def get_from_keys_or(config: dict[str, Any], keys: list[str], default: Any) -> Any:
        """Get an optional value from the model's quantization config."""
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default
