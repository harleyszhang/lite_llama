"""Quantisation base classes (mirrors sglang ``base_config.py``).

:class:`QuantizationConfig` is the checkpoint-side contract and
:class:`LinearMethodBase` / :class:`FusedMoEMethodBase` the runtime one;
:func:`run_quant_linear` is the shared dispatch over runtime schemes.

Usage:
    y = run_quant_linear(scheme, x, weight)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from .parameter import RawParameter

# --------------------------------------------------------------------------- #
# Scale-grid allocation
# --------------------------------------------------------------------------- #


def scale_parameter(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> RawParameter:
    """Allocate a 2-D scale grid in the physical layout the kernels read.

    Every blockwise dequant kernel addresses scales as ``scale_ptr +
    (offs_bn // GROUP_N) * stride_sn + k_block * stride_sk``: one row of
    scales per N-block, stepping along K. Row-major storage makes that
    N-step a strided jump (``stride_sn = k_blocks``, one cache line per
    element); allocating the grid K-major and returning the logical
    ``[n_blocks, k_blocks]`` view puts the N axis at stride 1, so the
    per-k-step scale row loads contiguously. SGLang's fp8 path materialises
    the same layout (``scale.t().contiguous().t()``); deciding it here, at
    allocation, keeps the forward path free of relayout copies and works
    across backends since every launcher passes the tensor's real strides.
    """
    n_blocks, k_blocks = shape
    return RawParameter(torch.empty(k_blocks, n_blocks, dtype=dtype).t())


def expert_scale_parameter(
    num_experts: int, shape: tuple[int, ...], dtype: torch.dtype = torch.float32
) -> RawParameter:
    """Stacked-experts counterpart of :func:`scale_parameter`.

    Logical ``[E, n_blocks, k_blocks]``, physical ``[E, k_blocks, n_blocks]``
    — the fused grouped-GEMM reads the same per-N-block scale rows the dense
    kernels do.
    """
    n_blocks, k_blocks = shape
    return RawParameter(torch.empty(num_experts, k_blocks, n_blocks, dtype=dtype).transpose(1, 2))


def column_major_scale(scale: torch.Tensor) -> torch.Tensor:
    """Relayout a row-major scale grid to the kernel-facing column-major one.

    Runtime-quantisation counterpart of :func:`scale_parameter`: producers
    that compute scales on the fly (``quantize_from_fp16``) emit them
    N-contiguous here instead of leaving the layout to chance. Per-channel
    grids (``[..., N, 1]``) pass through — with a single k-block the two
    byte orders coincide.
    """
    if scale.dim() == 2 and scale.shape[1] > 1:
        return scale.t().contiguous().t()
    if scale.dim() == 3 and scale.shape[2] > 1:
        return scale.transpose(1, 2).contiguous().transpose(1, 2)
    return scale


# --------------------------------------------------------------------------- #
# Method base classes
# --------------------------------------------------------------------------- #


class QuantizeMethodBase(ABC):
    """Base class for all quantised (or unquantised) method strategies."""

    def create_weights(self, layer: nn.Module, *weight_args, **extra_weight_attrs) -> None:
        """Allocate weight parameters on *layer*.

        The method decides what tensors exist (packed weight, scales, zeros)
        and registers them as attributes of *layer*.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        """Execute the quantised (or plain) computation on *layer*."""
        raise NotImplementedError()

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        """Quantise a loaded fp16 weight in-place, for the ``--quantization`` path.

        Schemes needing a pre-quantised checkpoint (AWQ/GPTQ) leave this raising.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be computed from fp16 weights at load time"
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Repack loaded weights once, after every tensor is filled.

        The seam for a method whose kernel layout differs from the checkpoint's
        (vLLM's same-named hook): the repack happens here, while parameters are
        still on the load device. Default is nothing to do.
        """
        return None


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
# Weight allocation helpers
# --------------------------------------------------------------------------- #


def allocate_linear_weights(layer: nn.Module, input_size: int, output_size: int) -> None:
    """Allocate ``[N, K]`` weight and fp32 ``weight_scale_inv`` on *layer*.

    Shared by the methods storing one value per element (fp8, int8,
    smoothquant); the packed int4/int8 configs allocate their own
    ``K//pack_factor`` shapes instead.
    """
    config: QuantizationConfig = layer.quant  # type: ignore[assignment]
    layer.weight = RawParameter(torch.empty(output_size, input_size, dtype=config.storage_dtype))
    layer.weight_scale_inv = scale_parameter(config.scale_shape(output_size, input_size))


def allocate_expert_weights(block: nn.Module) -> dict[str, nn.Parameter]:
    """Allocate stacked experts and their fp32 ``*_scale_inv`` grids.

    The :func:`allocate_linear_weights` counterpart for fused MoE, with the
    same one-value-per-element restriction.
    """
    config: QuantizationConfig = block.quant  # type: ignore[assignment]
    num_experts = getattr(block, "num_local_experts", block.num_experts)
    shapes = {
        "gate_up_proj": (2 * block.moe_intermediate_size, block.hidden_size),
        "down_proj": (block.hidden_size, block.moe_intermediate_size),
    }
    weights: dict[str, nn.Parameter] = {}
    for name, (n, k) in shapes.items():
        weights[name] = RawParameter(torch.empty(num_experts, n, k, dtype=config.storage_dtype))
        weights[f"{name}_scale_inv"] = expert_scale_parameter(num_experts, config.scale_shape(n, k))
    return weights


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
    def from_config(cls, config: dict[str, Any]) -> QuantizationConfig:
        """Factory: build from HF ``quantization_config`` dict."""
        raise NotImplementedError()

    @abstractmethod
    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
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
            module_name == ign or module_name.startswith(ign + ".") for ign in self.ignored
        )

    def _dispatch(
        self,
        layer: nn.Module,
        prefix: str,
        linear_method: type[LinearMethodBase],
        moe_method: type[FusedMoEMethodBase],
    ) -> QuantizeMethodBase:
        """Pick *layer*'s strategy: stacked experts vs plain linear.

        Shared by every config, which only differ in the two method classes.
        An ignored prefix still gets real tensors, just the fp16 methods.
        """
        from ..moe import SparseMoeBlock
        from .unquant import UnquantizedFusedMoEMethod, UnquantizedLinearMethod

        if not self.quantizes(prefix):
            linear_method, moe_method = UnquantizedLinearMethod, UnquantizedFusedMoEMethod
        if isinstance(layer, SparseMoeBlock):
            return moe_method()
        return linear_method()

    # -- Convenience properties (sglang compat) -------------------------------
    @property
    def is_fp8(self) -> bool:
        """True if this is an fp8 weight format."""
        return False

    @property
    def is_int4(self) -> bool:
        """True if this is an int4 weight format (AWQ/GPTQ)."""
        return False

    @property
    def is_packed(self) -> bool:
        """True if the checkpoint packs several values per storage word.

        Such layouts (``[N, K//pack_factor]`` int32) are consumed by no kernel
        directly, so loading runs them through
        :func:`rapid_llm.modules.quantization.adapt_packed_checkpoint` first.
        """
        return False

    @property
    def format(self) -> str:
        """Backward-compat alias for :meth:`get_name`."""
        return self.get_name()


# --------------------------------------------------------------------------- #
# Kernel dispatch helper
# --------------------------------------------------------------------------- #


def run_quant_linear(
    scheme: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """Route one quantised (or plain) projection through kernel dispatch.

    The single call site ``LinearMethodBase.apply`` implementations need: the
    scheme string is the dispatch key, dtype and shape come from the tensors,
    and the selected kernel sits behind the common :class:`LinearOp` signature.
    """
    from rapid_llm.kernels.dispatcher import dispatch, dtype_label

    if x.device.type == "cpu":
        from ...kernels.backend.cpu import linear

        return linear(
            scheme,
            x,
            weight,
            bias=bias,
            weight_scale=weight_scale,
            weight_zeros=weight_zeros,
            weight_global_scale=weight_global_scale,
            group_n=group_n,
            group_k=group_k,
        )

    # For a sub-byte format this ``k`` is the *storage* width, not the logical
    # one. It only feeds the perf-lookup key, which needs consistency rather
    # than semantics, and a scheme's keys never mix with another's.
    n, k = weight.shape[-2:]
    m = x.numel() // x.shape[-1]
    selected = dispatch(
        "linear",
        dtype=dtype_label(x.dtype),
        scheme=scheme,
        shape={"m": m, "n": n, "k": k},
    )
    return selected.load()(
        x,
        weight,
        bias=bias,
        weight_scale=weight_scale,
        weight_zeros=weight_zeros,
        weight_global_scale=weight_global_scale,
        group_n=group_n,
        group_k=group_k,
    )
