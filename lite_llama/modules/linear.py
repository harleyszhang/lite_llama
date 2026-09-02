"""Linear layers: one place that knows about both quantisation and sharding.

Every layer is a :class:`LinearBase` subclass; the column/row/QKV variants
own their TP shard maths, and the weight layout (quant method, scales) is
delegated to the quantisation sub-package's method objects.

Usage:
    qkv = QKVParallelLinear(hidden_size, num_heads, num_kv_heads, head_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..distributed.parallel_state import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from .quantization import QuantizationConfig, UnquantizedLinearMethod


class LinearBase(nn.Module):
    """``y = x @ W.T (+ b)`` with the weight stored however the quant method says.

    Subclasses decide how ``input_size``/``output_size`` are split; the
    :attr:`quant_method` owns the parameters and the multiply — composition,
    not subclassing, because sharding and storage format are orthogonal
    choices. ``dtype`` is the checkpoint's element type, defaulting to bf16
    for the undeclared-checkpoint case, and is threaded into
    :meth:`create_weights` so a bf16 checkpoint never allocates fp16.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant = quant
        # The checkpoint's element type; the model layer passes ``config.dtype``
        # (auto: follow config.json), and it is threaded into ``create_weights``
        # so an unquantised layer allocates in that type rather than in fp16.
        self.dtype = dtype
        self.quant_method = (
            quant.get_quant_method(self) if quant is not None else UnquantizedLinearMethod()
        )
        self.quant_method.create_weights(self, input_size, output_size, dtype=dtype)
        self.bias = (
            nn.Parameter(torch.empty(output_size, dtype=dtype), requires_grad=False)
            if bias
            else None
        )
        # One rule fills every parameter this layer owns: the weight, the quant
        # method's scale grids and the bias are all direct attributes, so the
        # non-recursive parameter iterator covers them all.
        for param in self.parameters(recurse=False):
            param.weight_loader = self._weight_loader

    def apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        """The multiply itself, without any tensor-parallel communication."""
        return self.quant_method.apply(self, x, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x)

    def _weight_loader(
        self, param: torch.Tensor, loaded: torch.Tensor, shard_id=None
    ) -> torch.Tensor:
        """Fill ``param`` from one checkpoint tensor and return the view written.

        The terminal step every subclass lands on: check the incoming tensor
        fits, copy, report what was written (the loader loop counts elements
        to verify coverage). Subclasses with a sharding rule override to
        compute the destination view and narrow ``loaded`` first, then call
        ``super()._weight_loader(view, loaded)``. ``shard_id`` names a block
        of a packed parameter (q/k/v, gate/up); ``None`` for unpacked.
        """
        if param.shape != loaded.shape:
            raise ValueError(
                f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                f"parameter view of shape {tuple(param.shape)}"
            )
        param.copy_(loaded)
        return param

    @torch.no_grad()
    def quantize_(self, quant: QuantizationConfig) -> None:
        """Replace a loaded 16-bit weight with its quantised form, in place.

        The ``--quantization <scheme>`` path: the fp16 storage is dropped as
        each layer converts, so peak memory stays at the size of the fp16
        checkpoint. Already-quantised layers (an fp8 checkpoint) are left alone.
        """
        if self.quant is not None:
            return
        method = quant.get_quant_method(self)
        method.quantize_from_fp16(self, quant)
        # Set quant before the hook: a method whose kernel layout differs from
        # what quantize_from_fp16 produced — GPTQ bits=8 unpacks its packed
        # words to int8 bytes — reads self.quant.bits inside the hook.
        self.quant = quant
        self.quant_method = method
        # Same post-load hook the checkpoint path runs (CausalLM.load_weights):
        # for every other method it is a no-op.
        method.process_weights_after_loading(self)

    def extra_repr(self) -> str:
        fmt = self.quant.format if self.quant else "unquantized"
        return f"in={self.input_size}, out={self.output_size}, bias={self.bias is not None}, {fmt}"


class ReplicatedLinear(LinearBase):
    """Full weight on every rank; for layers too small or too awkward to split."""


class ColumnParallelLinear(LinearBase):
    """Splits the output features across ranks; no communication.

    Each rank returns its slice of the output feature dimension, which the
    next :class:`RowParallelLinear` consumes as its own slice of the
    contracted dimension — that pairing keeps the all-reduce count at one
    per block. ``what`` names the split dimension for the error message.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
        what: str = "output features",
    ) -> None:
        world_size = get_tensor_model_parallel_world_size()
        local_out = divide(output_size, world_size, what)
        _check_shard_alignment(quant, local_out, what)
        super().__init__(input_size, local_out, bias=bias, quant=quant, dtype=dtype)
        self.full_output_size = output_size

    def _weight_loader(self, param, loaded, shard_id=None):
        # ``shard_id`` selects a half of the packed gate/up pair (``FusedMLP``
        # builds its fused projection from this class). The narrow is by
        # proportion of the incoming tensor, so scale grids — the same matrix
        # at scale-block resolution — follow the same rule as the weight.
        view = param.data
        if shard_id is not None:
            half = view.shape[0] // 2
            view = view.narrow(0, shard_id * half, half)

        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            size = loaded.shape[0] // world_size
            loaded = loaded.narrow(0, get_tensor_model_parallel_rank() * size, size)

        return super()._weight_loader(view, loaded)


class QKVParallelLinear(LinearBase):
    """The query, key and value projections as one column-parallel weight.

    Three GEMMs over the same activation become one over ``[q | k | v]``:
    the activation is read once and one kernel launch replaces three, which
    matters most on the memory-bound decode path where TP disables CUDA
    graphs and nothing hides launch overhead.

    Not simply ``ColumnParallelLinear(hidden, q + 2*kv)``: grouped-query
    attention gives more query heads than key/value heads, and the two block
    boundaries must be split independently — one cut of ``q + 2*kv`` would
    hand the low ranks nothing but query heads and the high ranks nothing but
    key/value heads. The local layout is ``[q | k | v]`` with heads adjacent
    inside a row, so :meth:`split` never copies: RoPE rotates the q and k
    blocks in place and the KV write addresses them by stride.

    Raises:
        ValueError: If either head count does not divide across the ranks, or
            a block's local width is not a whole number of scale blocks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        world_size = get_tensor_model_parallel_world_size()
        local_heads = divide(num_heads, world_size, "attention heads")
        local_kv_heads = divide(num_kv_heads, world_size, "key/value heads")
        q_size = local_heads * head_dim
        kv_size = local_kv_heads * head_dim
        # Checked per block, not on the total: a fused width can be a whole
        # number of scale blocks while the query block alone is not.
        _check_shard_alignment(quant, q_size, "query features")
        _check_shard_alignment(quant, kv_size, "key/value features")
        super().__init__(hidden_size, q_size + 2 * kv_size, bias=bias, quant=quant, dtype=dtype)
        self.num_heads = local_heads
        self.num_kv_heads = local_kv_heads
        self.head_dim = head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.full_output_size = (num_heads + 2 * num_kv_heads) * head_dim

    def split(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cut a fused output into ``(q, k, v)`` views along the last dimension.

        Views, not copies: each keeps the fused row stride, which every kernel
        downstream accepts.
        """
        return torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One GEMM, then three views — the call every attention block wants."""
        return self.split(self.apply_linear(x))

    def _weight_loader(self, param, loaded, shard_id=None):
        # ``shard_id`` is which of [q | k | v] arrived. Block boundaries come
        # from this layer's own head geometry, scaled to the parameter's
        # resolution: factor 1 for weight/bias, ``group_n`` for a scale grid
        # whose rows count scale blocks instead of channels.
        if shard_id is None:
            raise ValueError("qkv_proj is packed: a checkpoint tensor must name its block")
        factor = (self.q_size + 2 * self.kv_size) // param.shape[0]
        q_rows, kv_rows = self.q_size // factor, self.kv_size // factor
        offset, rows = ((0, q_rows), (q_rows, kv_rows), (q_rows + kv_rows, kv_rows))[shard_id]
        view = param.data.narrow(0, offset, rows)

        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            size = loaded.shape[0] // world_size
            loaded = loaded.narrow(0, get_tensor_model_parallel_rank() * size, size)

        return super()._weight_loader(view, loaded)

    def extra_repr(self) -> str:
        fmt = self.quant.format if self.quant else "fp16"
        return (
            f"in={self.input_size}, q_heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, bias={self.bias is not None}, {fmt}"
        )


class RowParallelLinear(LinearBase):
    """Splits the contracted features across ranks and all-reduces the result.

    Each rank multiplies its slice of ``x`` by its slice of ``W``, so what comes
    out is a partial sum; :func:`~lite_llama.distributed.parallel_state.tensor_model_parallel_all_reduce`
    completes it. The collective is inserted after the local multiply and gated
    on ``reduce_results`` and the world size — the same insertion point vLLM's
    ``RowParallelLinear`` uses, with the same flag. A bias is rejected rather
    than silently added ``world_size`` times — no projection in the supported
    models has one.

    Args:
        input_size: Full contracted width, split ``world_size`` ways.
        output_size: Full output width (not split).
        quant: Quantisation layout, or ``None``.
        reduce_results: Whether ``forward`` all-reduces the partial sums into the
            full output. ``True`` — the o_proj/down_proj default — keeps the
            collective inside the layer; ``False`` hands back the partial sum so
            a caller can time, batch or defer the collective itself (vLLM's
            ``ParallelLMHead`` uses the same escape hatch).
        what: Name of the dimension being split, for the error message.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant: QuantizationConfig | None = None,
        reduce_results: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        what: str = "input features",
    ) -> None:
        if bias:
            raise ValueError(
                "RowParallelLinear cannot carry a bias: it would be added once per rank"
            )
        world_size = get_tensor_model_parallel_world_size()
        local_in = divide(input_size, world_size, what)
        _check_shard_alignment(quant, local_in, what)
        super().__init__(local_in, output_size, quant=quant, dtype=dtype)
        self.full_input_size = input_size
        self.reduce_results = reduce_results

    def _weight_loader(self, param, loaded, shard_id=None):
        # The split is along the contracted dimension (columns of the weight
        # and of the scale grid alike); no packed form of this layer exists.
        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            size = loaded.shape[1] // world_size
            loaded = loaded.narrow(1, get_tensor_model_parallel_rank() * size, size)
        return super()._weight_loader(param.data, loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_parallel = self.apply_linear(x)
        # vLLM's insertion point: the collective rides on the layer itself, after
        # the multiply and before anything downstream can consume the output. The
        # world-of-one early return lives inside the collective, so the only
        # decision a call site makes is the reduce_results policy.
        if self.reduce_results:
            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel


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
