"""LayerNorm domain: RMSNorm, fused with the residual add when there is one.

Registers the domain's spec row and re-exports
:func:`~lite_llama.kernels.ops.layernorm.skip_rmsnorm.skip_rmsnorm`,
the fused residual-add + RMSNorm kernel.

Usage:
    from lite_llama.kernels import skip_rmsnorm
"""

from lite_llama.kernels.backend.flashinfer import CUDA_SM75
from lite_llama.kernels.dispatcher import (
    NATIVE_BASELINE,
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    register,
)

register(
    KernelSpec(
        name="native/skip_rmsnorm",
        op="rmsnorm",
        backend="native",
        target="lite_llama.kernels.ops.layernorm.skip_rmsnorm:skip_rmsnorm",
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
# Golden snapshot from bench_flashinfer.py (b8_s8_h4096, bf16): the fused
# pair agrees exactly on the residual and to one bf16 output ulp on the
# normalised activation — the 6.25e-2 max-abs-diff lands on an output around
# 8.0, where one bf16 ulp is exactly that, so the rtol window of the bench
# covers it.
register(
    KernelSpec(
        name="flashinfer/rmsnorm",
        op="rmsnorm",
        backend="flashinfer",
        target="lite_llama.kernels.backend.flashinfer.norm:rmsnorm",
        available="lite_llama.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16"),
        golden=GoldenRecord(verified=True, max_abs_diff=6.25e-2, baseline="native/skip_rmsnorm"),
        priority=UNMEASURED,
    )
)
