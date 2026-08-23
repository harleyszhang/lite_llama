"""Quant-method interfaces: per-format strategy for Linear and MoE layers.

Follows vLLM's ``LinearMethodBase`` / ``FusedMoEMethodBase`` split: the layer
module owns tensor-parallel sharding and routing, the method object owns
everything that depends on the storage format — which parameters exist, which
kernel runs, and how a loaded fp16 weight is converted at runtime. Adding a
scheme means one new method class and one registry line; the layers themselves
do not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..config import QuantConfig


class LinearQuantMethod(ABC):
    """Per-format behaviour of a :class:`~lite_llama.modules.linear.LinearBase`.

    The layer passes itself to every call, so the method stays stateless and
    one instance per layer is cheap.
    """

    @abstractmethod
    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        """Allocate the weight (and any scale grids) on ``layer``."""

    @abstractmethod
    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run ``x @ W.T (+ bias)`` with the layer's stored weight."""

    def convert_from_fp16(self, layer: nn.Module, quant: QuantConfig) -> None:
        """Replace the layer's loaded fp16 weight with its quantised form.

        Raises:
            NotImplementedError: If the scheme cannot be computed at load time.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be computed from fp16 weights at load time"
        )


class MoeQuantMethod(ABC):
    """Per-format behaviour of the stacked expert weights in a sparse MoE block."""

    @abstractmethod
    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        """Allocate the stacked expert tensors (and scale grids) for ``block``."""

    @abstractmethod
    def apply(
        self,
        block: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the routed grouped GEMM over the block's expert weights."""

    def convert_from_fp16(self, block: nn.Module, quant: QuantConfig) -> None:
        """Replace the block's loaded fp16 expert weights with quantised ones.

        Raises:
            NotImplementedError: If the scheme cannot be computed at load time.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be computed from fp16 weights at load time"
        )
