"""Linear layers: one place that knows about both quantisation and sharding.

Every projection in the decoder goes through one of the three classes here.
The classes themselves answer only the *sharding* question — the Megatron rule
that chaining a column-parallel layer into a row-parallel one (``gate/up`` then
``down``, ``q/k/v`` then ``o``) makes one all-reduce per block enough. The
*storage* question — fp16 ``F.linear`` or which quantised kernel — is delegated
to a quant-method object from :mod:`lite_llama.modules.quantization`,
so adding a scheme touches no layer code.

Parameter names deliberately match HuggingFace (``weight``, ``weight_scale_inv``),
so :mod:`lite_llama.models.weights` needs no rule for them.

Usage:
    self.q_proj = ColumnParallelLinear(hidden, q_size, quant=quant)
    self.o_proj = RowParallelLinear(q_size, hidden, quant=quant)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..distributed.parallel_state import all_reduce_tp, divide, get_tp_world_size
from .quantization import QuantizationConfig, UnquantizedLinearMethod


class LinearBase(nn.Module):
    """``y = x @ W.T (+ b)`` with the weight stored however the quant method says.

    Subclasses decide how ``input_size``/``output_size`` are split; the
    :attr:`quant_method` owns the parameters and the multiply, and the two are
    composed rather than subclassed because sharding and storage format are
    orthogonal choices.

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
        quant: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Legacy QuantConfig shim (backward compat for tests)
        if quant is not None and not isinstance(quant, QuantizationConfig):
            quant = _convert_legacy_quant(quant)
        self.quant = quant
        self.quant_method = (
            quant.get_quant_method(self) if quant is not None else UnquantizedLinearMethod()
        )
        self.quant_method.create_weights(self, input_size, output_size)
        self.bias = (
            nn.Parameter(torch.empty(output_size, dtype=torch.float16), requires_grad=False)
            if bias
            else None
        )

    def apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        """The multiply itself, without any tensor-parallel communication."""
        return self.quant_method.apply(self, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x)

    @torch.no_grad()
    def quantize_(self, quant) -> None:
        """Replace a loaded fp16 weight with its quantised form, in place.

        Used by the ``--quantization <scheme>`` path, where the checkpoint is
        fp16 and the low-bit weight has to be computed rather than read. The
        fp16 storage is dropped as each layer is converted, so peak memory
        stays at the size of the fp16 checkpoint. Layers that were already
        quantised (an fp8 checkpoint) are left alone.

        Accepts both new :class:`QuantizationConfig` and legacy :class:`QuantConfig`.
        """
        if self.quant is not None:
            return
        # Legacy QuantConfig shim (backward compat)
        if not isinstance(quant, QuantizationConfig):
            quant = _convert_legacy_quant(quant)
        method = quant.get_quant_method(self)
        method.quantize_from_fp16(self, quant)
        self.quant = quant
        self.quant_method = method

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
        quant: QuantizationConfig | None = None,
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
        quant: QuantizationConfig | None = None,
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


def _check_shard_alignment(quant: QuantizationConfig | None, local_size: int, what: str) -> None:
    """Reject a split that would cut a quantisation scale block in half.

    Raises:
        ValueError: If ``local_size`` is not a whole number of scale blocks, i.e.
            the requested tensor-parallel size is too large for this checkpoint's
            block size.
    """
    if quant is not None and not quant.shard_is_aligned(local_size):
        raise ValueError(
            f"tensor-parallel shard of {what} is {local_size} channels, which is not a "
            f"multiple of the {quant.get_name()} scale block ({quant.group_n}x{quant.group_k}); "
            "use a smaller tensor_parallel_size"
        )


def _convert_legacy_quant(quant) -> QuantizationConfig:
    """Convert a legacy :class:`QuantConfig` dataclass to a new :class:`QuantizationConfig`."""
    from .quantization.blockwise_int8 import BlockInt8Config
    from .quantization.awq import AWQConfig
    from .quantization.fp8 import Fp8Config
    from .quantization.w8a8_fp8 import W8A8Fp8Config
    from .quantization.w8a8_int8 import W8A8Int8Config

    fmt = quant.format
    if fmt == "int8":
        if quant.group_k < (1 << 30):
            return BlockInt8Config(group_k=quant.group_k, ignored=quant.ignored)
        return BlockInt8Config.per_channel()
    if fmt == "fp8":
        if quant.is_dynamic:
            return W8A8Fp8Config(group_n=quant.group_n, group_k=quant.group_k, ignored=quant.ignored)
        return Fp8Config(group_n=quant.group_n, group_k=quant.group_k, ignored=quant.ignored)
    if fmt == "int4":
        return AWQConfig(group_size=quant.group_k, ignored=quant.ignored)
    if fmt == "smoothquant":
        return W8A8Int8Config(ignored=quant.ignored)
    raise ValueError(f"cannot convert legacy QuantConfig with format {fmt!r}")
