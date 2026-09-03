"""Scale-tensor layout descriptors: allocate in the consumer's format up front.

A quantised GEMM does not read its scale grid the way the quantiser happens to
write it. Every consumer has its own demand -- a Triton block GEMM walks scales
row-major, a TMA-fed operand (DeepGEMM, cutlass ``fp8_blockwise_scaled_mm``)
wants them column-major with the token stride padded to a vector-load multiple,
and a UE8M0 consumer wants four exponent bytes packed per ``int32``. Deciding
that at *consumption* time means a transpose, a pad and a ``cat`` on the
critical path of every layer; deciding it at *allocation* time costs nothing,
because the buffer is empty anyway and ``torch.empty`` of a padded shape is the
same call as of an exact one.

Usage:
    s = create_scale_output(x.shape, x.device, 128, COLUMN_MAJOR_TMA)
    q, s = per_token_group_quant(x, 128, output_q=q, output_s=s)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Rows one TMA-aligned scale grid is padded to a multiple of. Four fp32 words
#: is 16 bytes -- the width of a 128-bit vector load, and the granularity TMA
#: boxes are described in. sglang uses the same constant
#: (``aligned_size = (x_shape[-2] + 3) // 4 * 4``).
TMA_SCALE_ALIGNMENT = 4


@dataclass(frozen=True)
class ScaleLayout:
    """The physical layout a scale grid must be allocated in.

    Attributes:
        column_major: Store the ``[T, G]`` grid with token stride 1 -- allocate
            ``[G, T]`` and hand back a transposed view. What a TMA consumer
            reads; a row-major Triton GEMM would rather have the other one.
        tma_aligned: Pad the token stride up to :data:`TMA_SCALE_ALIGNMENT`.
            Only meaningful with ``column_major`` (a row-major grid already
            has its group stride contiguous, so there is nothing to pad) -- the
            ``__post_init__`` check rejects the combination rather than
            silently ignoring half of it.

    The three module constants below cover what the current backends need;
    construct one directly for anything else.
    """

    column_major: bool = False
    tma_aligned: bool = False

    def __post_init__(self) -> None:
        if self.tma_aligned and not self.column_major:
            raise ValueError(
                "tma_aligned pads the token stride, which only a column-major "
                "grid has; row-major scales are already contiguous along groups"
            )


#: ``[T, G]`` contiguous -- what the Triton block GEMMs read.
ROW_MAJOR = ScaleLayout()

#: ``[T, G]`` with token stride 1, unpadded -- the transposed view, for a
#: consumer that wants column-major but does its own alignment.
COLUMN_MAJOR = ScaleLayout(column_major=True)

#: Column-major with the token stride padded to a multiple of
#: :data:`TMA_SCALE_ALIGNMENT` -- what a TMA-fed operand wants, and the layout
#: that lets the consumer's own pad-and-cat step short-circuit.
COLUMN_MAJOR_TMA = ScaleLayout(column_major=True, tma_aligned=True)


def create_scale_output(
    x_shape: tuple[int, ...],
    device: torch.device,
    group_size: int,
    layout: ScaleLayout = ROW_MAJOR,
) -> torch.Tensor:
    """Allocate the scale grid for an ``x_shape`` activation, already in *layout*.

    The buffer comes back *logically* shaped ``[..., rows, cols // group_size]``
    whichever way it is stored, so the caller indexes it identically in every
    layout and only ``.stride()`` tells them apart.

    Args:
        x_shape: Shape of the activation the scales belong to. Column-major
            layouts need exactly two dimensions (a transposed view of a
            batched grid has no single token stride to describe).
        device: Device to allocate on.
        group_size: Elements sharing one scale.
        layout: The layout to allocate in.

    Returns:
        fp32 scale grid. Row-major is contiguous; column-major is a
        transposed-stride view whose storage stays what the kernel writes.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    rows, cols = x_shape[-2], x_shape[-1]
    if cols % group_size:
        raise ValueError(f"row width {cols} is not a multiple of group_size {group_size}")
    num_groups = cols // group_size

    if not layout.column_major:
        return torch.empty((*x_shape[:-1], num_groups), device=device, dtype=torch.float32)

    if len(x_shape) != 2:
        raise ValueError(
            f"column-major scales support 2D activations only, got {len(x_shape)}D"
        )
    # Pad the *storage* rows, then slice the view back to the real row count:
    # the kernel writes rows [0, rows) while a TMA descriptor can span the
    # padded extent, and the consumer's own pad step finds nothing to do.
    stride = (
        (rows + TMA_SCALE_ALIGNMENT - 1) // TMA_SCALE_ALIGNMENT * TMA_SCALE_ALIGNMENT
        if layout.tma_aligned
        else rows
    )
    grid = torch.empty((num_groups, stride), device=device, dtype=torch.float32)
    return grid.transpose(0, 1)[:rows, :]


def infer_scale_layout(output_s: torch.Tensor) -> ScaleLayout:
    """Read the layout back off a caller-allocated scale buffer.

    The mirror of :func:`create_scale_output`: a caller that allocated its own
    grid -- from a previous layer's buffer, a CUDA-graph-captured slab, another
    framework's allocator -- should not have to restate what it did. sglang
    resolves this the same way in ``_infer_scale_layout``, and it is what lets
    one quantiser serve consumers with different demands.

    Args:
        output_s: ``[T, G]`` fp32 scale grid.

    Returns:
        The :class:`ScaleLayout` the buffer's dtype and strides describe.
    """
    if output_s.dtype != torch.float32:
        raise ValueError(f"scale buffers must be fp32, got {output_s.dtype}")
    if output_s.dim() < 2:
        raise ValueError(f"scale buffers must be at least 2D, got {output_s.dim()}D")

    # Row-major grids are contiguous along groups (token stride >= group
    # stride); column-major ones put the token stride at 1 instead.
    if output_s.stride(-2) >= output_s.stride(-1):
        return ROW_MAJOR

    # Column-major. A padded token stride is a multiple of the alignment; an
    # unpadded one is the row count -- which for a row count already a multiple
    # of the alignment is indistinguishable, and equivalent, since such a grid
    # needs no padding to satisfy a TMA descriptor anyway.
    return ScaleLayout(
        column_major=True,
        tma_aligned=output_s.stride(-1) % TMA_SCALE_ALIGNMENT == 0,
    )
