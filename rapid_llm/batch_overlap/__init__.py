"""Batch overlap: both overlap axes, in sglang's ``batch_overlap`` layout.

Five pieces:

* :mod:`~rapid_llm.batch_overlap.overlap` — the host↔device axis (L1):
  :class:`StreamPool` stages async uploads and pinned readbacks on a copy
  stream, :class:`Timeline` records CUDA-event regions as overlap evidence.
* :mod:`~rapid_llm.batch_overlap.operations` — the stage/yield interleaving
  primitives (:class:`YieldOperation`, :class:`StateDict`,
  :func:`execute_overlapped_operations`).
* :mod:`~rapid_llm.batch_overlap.comm_overlap` — the comm-stream plumbing
  both overlap policies ride: deferred all-reduces for L2 (TBO), chunked
  all-reduces for L3, and the async all-to-all EP dispatch/combine uses.
* :mod:`~rapid_llm.batch_overlap.two_batch_overlap` — the L2 decode
  ping-pong executor and its policy.
* :mod:`~rapid_llm.batch_overlap.operations_strategy` — per-layer operation
  streams: the layers' own bound methods, ordered with the yields placed
  (DeepSeek EP+TBO follows sglang's decode strategy).

Usage:
    from rapid_llm.batch_overlap import deferred_all_reduce, StateDict
    from rapid_llm.batch_overlap.two_batch_overlap import tbo_policy

The package root re-exports only the kernel-free modules (``overlap``,
``operations``, ``comm_overlap``): ``modules/`` components import from here,
and the eager import boundary (:mod:`tests.test_imports`) forbids them from
pulling the Triton kernels ``two_batch_overlap`` rides. Executor and model
call sites import that submodule directly.
"""

from .comm_overlap import (
    CommOverlapPolicy,
    CommStreamPool,
    DeferredArContext,
    comm_overlap_policy,
    current_deferred_ar,
    deferred_all_reduce,
    reset_comm_overlap_policy,
    row_parallel_forward,
    skip_row_parallel_all_reduce,
)
from .operations import (
    StateDict,
    YieldOperation,
    execute_operations,
    execute_overlapped_operations,
)
from .overlap import OverlapPolicy, RegionRecord, StreamPool, Timeline

__all__ = [
    "CommOverlapPolicy",
    "CommStreamPool",
    "DeferredArContext",
    "OverlapPolicy",
    "RegionRecord",
    "StateDict",
    "StreamPool",
    "Timeline",
    "YieldOperation",
    "comm_overlap_policy",
    "current_deferred_ar",
    "deferred_all_reduce",
    "execute_operations",
    "execute_overlapped_operations",
    "reset_comm_overlap_policy",
    "row_parallel_forward",
    "skip_row_parallel_all_reduce",
]
