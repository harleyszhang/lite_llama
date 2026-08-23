"""AWQ checkpoint method (corresponds to vLLM's ``auto_awq.py``).

What makes AWQ distinct is only the checkpoint layout — 4-bit values packed
along the *output* dim in an interleaved bit order — which the load stream
converts to the canonical w4a16 layout (see
:mod:`lite_llama.models.quantization._layout.awq`). The runtime behaviour is
the shared group-wise int4 kernel, so this class adds the name and the
documentation, not new code.
"""

from __future__ import annotations

from .w4a16 import W4A16LinearMethod


class AWQLinearMethod(W4A16LinearMethod):
    """Group-wise int4 from an AutoAWQ checkpoint; runs the w4a16 kernel."""
