"""GPTQ checkpoint method (corresponds to vLLM's ``auto_gptq.py``).

What makes GPTQ distinct is only the checkpoint layout — ``[K//8, N]`` word
matrix plus biased packed zero points — which the load stream converts to the
canonical w4a16 layout (see
:mod:`lite_llama.models.quantization._layout.gptq`). The runtime behaviour is
the shared group-wise int4 kernel, so this class adds the name and the
documentation, not new code.

Only ``desc_act=False`` checkpoints are accepted: activation ordering
scatters each group over non-contiguous input channels, which the w4a16
kernel cannot index.
"""

from __future__ import annotations

from .w4a16 import W4A16LinearMethod


class GPTQLinearMethod(W4A16LinearMethod):
    """Group-wise int4 from an AutoGPTQ checkpoint; runs the w4a16 kernel."""
