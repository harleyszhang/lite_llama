"""Sampling domain: turning logits into token ids.

``sample`` deliberately has no native row — the engine's sampler
(``engine/sampler.py``) already runs on the TP-sharded vocab with repetition
penalty, and a second implementation would be a second place sampling could
diverge between ranks. The FlashInfer row exists because its top-k/top-p
kernels are what a fused-sampling future would dispatch to, and registering
the op keeps the contract pinned even while the engine owns the default path.

Usage:
    from lite_llama.kernels.ops import sampling  # noqa: F401  (registers the row)
"""

from lite_llama.kernels.backend.flashinfer import CUDA_SM75
from lite_llama.kernels.dispatcher import (
    GoldenRecord,
    KernelSpec,
    UNMEASURED,
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
