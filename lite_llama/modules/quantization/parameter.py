"""RawParameter: marker for "do not cast this".

Casting passes (dtype moves, ``to`` calls) skip :class:`RawParameter`
instances so quantised payloads — int4 packs, fp8 blocks, scale tables —
keep their storage layout while the rest of the model casts.

Usage:
    weight = RawParameter(data, requires_grad=False)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RawParameter(nn.Parameter):
    """A parameter the loader must leave alone instead of casting to fp16.

    The loader keeps the original dtype for every parameter that satisfies
    ``isinstance(param, RawParameter)`` (see
    :meth:`lite_llama.executor.loader.materialise_parameters`).
    """

    def __new__(cls, data: torch.Tensor, requires_grad: bool = False) -> RawParameter:
        # nn.Parameter defaults to requires_grad=True, which is rejected
        # outright for the integer/byte tensors quantised weights live in;
        # inference-only quantised parameters default to False instead.
        return super().__new__(cls, data, requires_grad=requires_grad)
