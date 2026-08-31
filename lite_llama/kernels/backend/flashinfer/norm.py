"""FlashInfer RMSNorm wrapper behind the native signature.

The contract (:class:`~lite_llama.kernels.ops.interfaces.RmsNormOp`) returns a
pair — ``(normalised, residual)`` — with the second element being the summed
input on the fused path and ``x`` itself on the plain one. FlashInfer's plain
``rmsnorm`` is functional (returns the normalised tensor), while
``fused_add_rmsnorm`` is in-place over its two arguments (reading each before
writing both), so the wrapper's whole job is to make it look like the same
functional pair: clone both operands before the kernel consumes them, hand
back the pair the decoder layer threads into the next norm without branching.

Usage (from a spec row's ``target``):
    from lite_llama.kernels.backend.flashinfer.norm import rmsnorm
"""

from __future__ import annotations


def rmsnorm(x, residual, weight, eps: float = 1e-5):
    """RMSNorm via FlashInfer, fused with the residual add when asked.

    Args follow :func:`~lite_llama.kernels.ops.layernorm.skip_rmsnorm.
    skip_rmsnorm` exactly: ``residual`` is positional — ``None`` selects the
    plain path — and the return is always ``(normalised, residual)``.
    """
    import flashinfer

    if residual is None:
        # Plain path: functional kernel; the returned residual is x itself.
        out = flashinfer.norm.rmsnorm(x, weight, eps)
        return out, x

    # Fused path: the kernel reads its first argument as the input and its
    # second as the residual addend, then writes norm(input + residual) into
    # the first slot and the sum into the second — both in place. The residual
    # slot is read *before* it is written, so it must carry the plain
    # ``residual`` (not a pre-summed tensor: that would double-add x). One
    # clone per operand keeps the functional contract; after the call the
    # clones hold exactly the pair the decoder layer threads onward.
    xi = x.clone()
    ri = residual.clone()
    flashinfer.norm.fused_add_rmsnorm(xi, ri, weight, eps)
    return xi, ri
