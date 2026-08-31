"""FlashInfer RoPE wrapper behind the native signature.

The contract (:class:`~lite_llama.kernels.ops.interfaces.RopeOp`) is
table-driven: the module layer computes ``cos``/``sin`` for the batch's
geometry and hands them over. FlashInfer's ``apply_rope`` is varlen- and
offset-driven — packed ``q``/``k``, batch ``indptr``, per-sequence position
``offsets``, tables derived internally from ``rope_theta`` — so the wrapper
rebuilds those from the table geometry: this repo's tables cover contiguous
positions 0..seq-1 per sequence, which is ``indptr = arange * seq_len`` with
zero offsets. The rotation bases (``rope_theta``) must match too: the contract
carries no theta, so the row's golden record is measured on the default
10000-base models and a theta-divergent model would surface as a golden
failure, not a silent wrong rotation.

Usage (from a spec row's ``target``):
    from lite_llama.kernels.backend.flashinfer.rope import rope
"""

from __future__ import annotations

import torch


def rope(q, k, cos, sin):
    """Rotate ``q``/``k`` by their positions, via FlashInfer.

    Args follow :func:`~lite_llama.kernels.ops.rope.rope_emb.rope_emb_forward`
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
