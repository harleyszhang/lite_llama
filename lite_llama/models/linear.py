"""Linear layers: one place that knows about both quantisation and sharding.

Every projection in the decoder goes through one of the three classes here, and
each of them answers two questions that used to be spread across the model code:
*is this weight 8-bit* (then the fp16 ``F.linear`` becomes
:func:`~lite_llama.kernels.w8a16.w8a16_matmul`) and *is this weight split across
tensor-parallel ranks* (then how, and does the result need an all-reduce).

Which of the three a projection uses follows from where its matrix is contracted,
the standard Megatron split::

    x @ [W1 | W2].T          -> ColumnParallelLinear, no communication, output is
                                already the rank's slice of the feature dim
    [x1 | x2] @ [W1 ; W2].T  -> RowParallelLinear, each rank holds a partial sum,
                                so the forward ends in an all-reduce

Chaining a column-parallel layer into a row-parallel one — ``gate/up`` then
``down``, ``q/k/v`` then ``o`` — is what makes one all-reduce per block enough.

Parameter names deliberately match HuggingFace (``weight``, ``weight_scale_inv``),
so :mod:`lite_llama.models.weights` needs no rule for them.

Usage:
    self.q_proj = ColumnParallelLinear(hidden, q_size, quant=quant)
    self.o_proj = RowParallelLinear(q_size, hidden, quant=quant)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.parallel_state import all_reduce_tp, divide, get_tp_world_size
from ..kernels.w8a16 import w8a16_matmul
from ..kernels.w4a16 import w4a16_matmul
from ..kernels.smoothquant import smoothquant_matmul
from .quantization import (
    FP8,
    INT4,
    INT8,
    SMOOTHQUANT,
    QuantConfig,
    RawParameter,
    quantize_int8_per_channel,
)


class LinearBase(nn.Module):
    """``y = x @ W.T (+ b)`` with the weight held at fp16 or 8 bit.

    Subclasses decide how ``input_size``/``output_size`` are split; this class
    owns the parameters and the multiply, and the two are kept together because
    the choice of kernel follows from how the weight was stored.

    Args:
        input_size: Contracted (in-feature) width of the local weight.
        output_size: Output width of the local weight.
        bias: Whether to allocate a bias.
        quant: Quantisation layout, or ``None`` for a plain fp16 weight.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant = quant

        if quant is None:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float16), requires_grad=False
            )
        elif quant.format in (FP8, INT8, SMOOTHQUANT):
            # 8-bit weight + scale grid.
            self.weight = RawParameter(
                torch.empty(output_size, input_size, dtype=quant.storage_dtype)
            )
            self.weight_scale_inv = RawParameter(
                torch.empty(*quant.scale_shape(output_size, input_size), dtype=torch.float32)
            )
        elif quant.format == INT4:
            # AWQ/GPTQ: packed int32 weight + group-wise scales + zeros.
            # 8 int4 values per int32 word along the K dimension.
            packed_k = (input_size + 7) // 8
            self.weight = RawParameter(
                torch.empty(output_size, packed_k, dtype=torch.int32)
            )
            self.weight_scale = RawParameter(
                torch.empty(*quant.scale_shape(output_size, input_size), dtype=torch.float32)
            )
            self.weight_zeros = RawParameter(
                torch.empty(*quant.scale_shape(output_size, input_size), dtype=torch.float32)
            )
        else:
            raise ValueError(f"unsupported quantisation format: {quant.format}")

        self.bias = (
            nn.Parameter(torch.empty(output_size, dtype=torch.float16), requires_grad=False)
            if bias
            else None
        )

    def apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        """The multiply itself, without any tensor-parallel communication."""
        if self.quant is None:
            return F.linear(x, self.weight, self.bias)

        if self.quant.format in (FP8, INT8):
            return w8a16_matmul(
                x,
                self.weight,
                self.weight_scale_inv,
                group_n=self.quant.group_n,
                group_k=min(self.quant.group_k, self.input_size),
                bias=self.bias,
            )

        if self.quant.format == SMOOTHQUANT:
            return smoothquant_matmul(
                x,
                self.weight,
                self.weight_scale_inv,
                bias=self.bias,
            )

        if self.quant.format == INT4:
            return w4a16_matmul(
                x,
                self.weight,
                self.weight_scale,
                self.weight_zeros,
                group_size=self.quant.group_k,
                bias=self.bias,
            )

        raise ValueError(f"unsupported quantisation format: {self.quant.format}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x)

    @torch.no_grad()
    def quantize_(self, quant: QuantConfig) -> None:
        """Replace a loaded fp16 weight with its int8 quantisation, in place.

        Used by the ``--quantization int8`` path, where the checkpoint is fp16 and
        the 8-bit weight has to be computed rather than read. The fp16 storage is
        dropped as each layer is converted, so peak memory stays at the size of
        the fp16 checkpoint rather than the sum of both.
        """
        if self.quant is not None:
            return
        qweight, scale = quantize_int8_per_channel(self.weight.data)
        self.weight = RawParameter(qweight)
        self.weight_scale_inv = RawParameter(scale)
        self.quant = quant

    def extra_repr(self) -> str:
        fmt = self.quant.format if self.quant else "fp16"
        return f"in={self.input_size}, out={self.output_size}, bias={self.bias is not None}, {fmt}"


class ReplicatedLinear(LinearBase):
    """Full weight on every rank; for layers too small or too awkward to split."""


class ColumnParallelLinear(LinearBase):
    """Splits the output features across ranks; no communication.

    The result is each rank's slice of the output feature dimension, which the
    next :class:`RowParallelLinear` consumes as its own slice of the contracted
    dimension — that pairing is what keeps the all-reduce count at one per block.

    Args:
        input_size: Full contracted width (not split).
        output_size: Full output width, split ``world_size`` ways.
        bias: Whether to allocate a bias; sharded with the weight.
        quant: Quantisation layout, or ``None``.
        what: Name of the dimension being split, for the error message when it
            does not divide.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantConfig | None = None,
        what: str = "output features",
    ) -> None:
        world_size = get_tp_world_size()
        local_out = divide(output_size, world_size, what)
        _check_shard_alignment(quant, local_out, what)
        super().__init__(input_size, local_out, bias=bias, quant=quant)
        self.full_output_size = output_size


class RowParallelLinear(LinearBase):
    """Splits the contracted features across ranks and all-reduces the result.

    Each rank multiplies its slice of ``x`` by its slice of ``W``, so what comes
    out is a partial sum; :func:`~lite_llama.distributed.parallel_state.all_reduce_tp`
    completes it. A bias is rejected rather than silently added ``world_size``
    times — no projection in the supported models has one.

    Args:
        input_size: Full contracted width, split ``world_size`` ways.
        output_size: Full output width (not split).
        quant: Quantisation layout, or ``None``.
        what: Name of the dimension being split, for the error message.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantConfig | None = None,
        what: str = "input features",
    ) -> None:
        if bias:
            raise ValueError("RowParallelLinear cannot carry a bias: it would be added once per rank")
        world_size = get_tp_world_size()
        local_in = divide(input_size, world_size, what)
        _check_shard_alignment(quant, local_in, what)
        super().__init__(local_in, output_size, quant=quant)
        self.full_input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return all_reduce_tp(self.apply_linear(x))


def _check_shard_alignment(quant: QuantConfig | None, local_size: int, what: str) -> None:
    """Reject a split that would cut a quantisation scale block in half.

    Raises:
        ValueError: If ``local_size`` is not a whole number of scale blocks, i.e.
            the requested tensor-parallel size is too large for this checkpoint's
            block size.
    """
    if quant is not None and not quant.shard_is_aligned(local_size):
        raise ValueError(
            f"tensor-parallel shard of {what} is {local_size} channels, which is not a "
            f"multiple of the {quant.format} scale block ({quant.group_n}x{quant.group_k}); "
            "use a smaller tensor_parallel_size"
        )
