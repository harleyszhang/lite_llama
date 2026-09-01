"""Sampling domain: turning logits into token ids.

Registers the domain's spec rows; the stochastic draw itself is served
by the native sampler or an external backend's sampling wrapper.

Usage:
    from lite_llama.kernels.ops import SampleOp
"""

from lite_llama.kernels.backend.flashinfer import CUDA_SM75
from lite_llama.kernels.dispatcher import (
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    register,
)

register(
    KernelSpec(
        name="flashinfer/sample",
        op="sample",
        backend="flashinfer",
        target="lite_llama.kernels.backend.flashinfer.sample:sample",
        available="lite_llama.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16", "fp32"),
        # Sampling is compared on the greedy path (argmax parity), where the
        # two implementations must agree exactly.
        golden=GoldenRecord(verified=True, max_abs_diff=0.0, baseline="greedy argmax parity"),
        priority=UNMEASURED,
    )
)
