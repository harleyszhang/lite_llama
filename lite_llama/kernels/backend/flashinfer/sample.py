"""FlashInfer sampling wrapper behind the native signature.

The contract (:class:`~lite_llama.kernels.ops.interfaces.SampleOp`) is the
sampler's kernel core: temperature, top-k, top-p, one token per row. The
deterministic (greedy) path is plain argmax on purpose — golden runs and the
argmax-parity check need both implementations to agree exactly, and routing
greedy through a sampling kernel would only add a place to disagree.

Usage (from a spec row's ``target``):
    from lite_llama.kernels.backend.flashinfer.sample import sample
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
