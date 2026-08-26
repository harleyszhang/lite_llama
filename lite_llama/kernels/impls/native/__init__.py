"""Native floor implementations: always present, never capability-gated.

Importing this package registers the native KernelSpec rows (torch-free,
via :mod:`registry`) without importing any kernel module — torch and triton
load lazily when :meth:`~lite_llama.kernels.ops.Selected.load` resolves a
target. One module per logical-op family lands here as the restructure
proceeds: ``linear`` first (M1.4), then attention, moe, elementwise, sample.
"""

from . import registry as registry
