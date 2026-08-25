"""RawParameter: marker for "do not cast this".

:func:`lite_llama.executor.loader.materialise_parameters` gives every
floating-point parameter fp16 storage, which is right for weights and wrong
for the two things quantisation adds: the 8-bit weight itself (``uint8`` /
``int8``, so it is not floating point anyway) and its fp32 scales, whose
dynamic range is the reason the fp8 format works at all.
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
