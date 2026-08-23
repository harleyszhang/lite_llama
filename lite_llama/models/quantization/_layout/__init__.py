"""Weight layout rearrangement per checkpoint backend (private).

Every quantised checkpoint format packs its int4 weights its own way; the
w4a16 kernel wants one canonical layout (``[N, K//8]`` int32 along the input
dim in sequential bit order, fp32 scales/zeros of shape ``[N, G]``). Each
backend gets its own module — ``_layout/awq.py``, ``_layout/gptq.py`` —
exposing per-tensor converters plus an ``adapt_key`` that renames a
checkpoint leaf to its canonical parameter name.
:func:`adapt_int4_checkpoint` applies that to a whole checkpoint stream, so
the copy loop in :mod:`lite_llama.models.weights` stays layout-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from . import awq, gptq

if TYPE_CHECKING:
    from ..config import QuantConfig

#: Checkpoint ``quant_method`` -> its backend's key adapter.
_ADAPTERS = {"awq": awq.adapt_key, "gptq": gptq.adapt_key}


def adapt_int4_checkpoint(
    checkpoint: Iterable[tuple[str, torch.Tensor]], quant: QuantConfig
) -> Iterable[tuple[str, torch.Tensor]]:
    """Rewrite an int4 checkpoint stream to the canonical w4a16 layout.

    Args:
        checkpoint: ``(key, tensor)`` pairs as yielded by
            :func:`lite_llama.executor.weight_utils.hf_weights_iterator`.
        quant: The checkpoint's quantisation config; ``quant.method`` selects
            the backend adapter.

    Raises:
        ValueError: If the checkpoint's ``quant_method`` has no adapter.
    """
    try:
        adapt = _ADAPTERS[quant.method]
    except KeyError:
        raise ValueError(
            f"no checkpoint layout adapter for quant_method {quant.method!r}; "
            f"supported: {sorted(_ADAPTERS)}"
        ) from None
    for key, tensor in checkpoint:
        out = adapt(key, tensor)
        if out is not None:
            yield out
