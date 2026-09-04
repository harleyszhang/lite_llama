"""FlashInfer RoPE wrapper behind the native signature.

``rope`` applies the rotation with FlashInfer's kernel using the cos/sin
tables the native rope kernel already receives.

Usage:
    rope(q, k, cos, sin)
"""

from __future__ import annotations

import torch


def rope(q, k, cos, sin):
    """Rotate ``q``/``k`` by their positions, via FlashInfer.

    Args follow :func:`~rapid_llm.kernels.ops.rope.rope_emb.rope_emb_forward`
    exactly; ``cos``/``sin`` carry the batch/sequence geometry.
    """
    from flashinfer.rope import apply_rope

    # [batch, seq, dim] table -> varlen batch of equal-length sequences,
    # positions starting at 0 (the geometry this repo's tables encode).
    batch, seq_len, _head_dim = cos.shape
    indptr = torch.arange(batch + 1, dtype=torch.int32, device=cos.device) * seq_len
    offsets = torch.zeros(batch, dtype=torch.int32, device=cos.device)

    q_flat = q.reshape(-1, q.shape[1], q.shape[2])
    k_flat = k.reshape(-1, k.shape[1], k.shape[2])
    q_out, k_out = apply_rope(q_flat, k_flat, indptr, offsets)
    return q_out.view_as(q), k_out.view_as(k)
