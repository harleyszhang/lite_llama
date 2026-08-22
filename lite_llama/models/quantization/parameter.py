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
    """A parameter the loader must leave alone instead of casting to fp16."""

    def __new__(cls, data: torch.Tensor, requires_grad: bool = False) -> RawParameter:
        return super().__new__(cls, data, requires_grad=requires_grad)
