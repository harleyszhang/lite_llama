"""FlashInfer sampling wrapper behind the native signature.

``sample`` delegates the top-p draw to FlashInfer's sampling kernel,
taking the same logits tensor the native sampler path receives.

Usage:
    token = sample(logits)
"""

from __future__ import annotations

import torch


def sample(
    logits,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    deterministic: bool = False,
):
    """Draw one token id per row, via FlashInfer's fused top-k/top-p kernels.

    Args follow the SampleOp contract; ``logits`` is this rank's TP slice.
    """
    if deterministic:
        return torch.argmax(logits, dim=-1)

    import flashinfer

    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)
