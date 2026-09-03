"""Single-batch overlap (SBO): MoE two-stream overlap *inside* one batch.

The counterpart of sglang's ``srt/batch_overlap/single_batch_overlap.py``.
Where TBO splits a *batch* into two halves that ping-pong, SBO splits the
*work inside one MoE layer* across two streams — so it pays on the EP decode
shape, where there is only one batch and no second half to interleave with.

What ships is sglang's dispatch↔shared pair: the forward exchange goes on the
wire first, and the shared MLP moves onto an alternate compute stream so it
computes while the tokens travel. :meth:`SparseMoeBlock._forward_ep` drives it
and owns both fences; this module supplies the switch and the stream.

Two of sglang's three overlaps are deliberately absent, and the reasons are
worth stating rather than leaving implicit:

* **combine↔down GEMM** (tile signaled) — needs the down GEMM to publish each
  output tile and the reduction to wait on it. ``fused_moe``'s second GEMM
  scatters its output by ``sorted_token_ids`` while ``_moe_sum_kernel`` reads
  contiguous token rows, so a finished tile does not line up with a ready row
  block; wiring it needs an inverse mapping plus an atomic count.
* **combine↔shared** — wants the same alternate stream the dispatch pair
  already occupies.

One more adaptation: sglang sizes the communication side through
``DeepEPConfig.num_sms``, pinning the exchange to a fixed subset of SMs.
lite_llama's combine rides ``all_to_all_single``, whose NCCL kernels take no
SM budget from the caller — how many SMs the exchange actually occupies is an
external variable here, measured by the benchmark rather than controlled.

Usage:
    os.environ["LITE_LLAMA_SBO"] = "1"   # the MoE block picks the overlap up itself
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

#: Environment variable switching single-batch overlap on (``0`` disables).
SBO_ENV = "LITE_LLAMA_SBO"

#: Row count a MoE layer must reach before SBO splits its streams. The cost
#: SBO pays is two event fences and a ``record_stream`` mark — microseconds —
#: while what it hides is an exchange whose wire time grows with the payload,
#: so the floor sits where the exchange stops being cheaper than the fences.
#: Unlike L4's tile-signaling there is no persistent-kernel occupancy to pay,
#: which is why this floor is an order of magnitude lower than that one's.
SBO_MIN_ROWS_ENV = "LITE_LLAMA_SBO_MIN_ROWS"


@dataclass(frozen=True)
class SboPolicy:
    """The SBO switch: overlap inside one MoE layer, on two streams.

    Off by default. Like every other overlap policy here, it changes the order
    reductions happen in, so opting in is explicit rather than silent.

    Args:
        enabled: Whether eligible MoE layers split their streams.
        min_rows: Token count a layer must reach before it is eligible.
    """

    enabled: bool = False
    min_rows: int = 32

    @classmethod
    def from_env(cls) -> SboPolicy:
        """Read ``LITE_LLAMA_SBO``; anything but ``0``/``false``/``off`` means on."""
        raw = os.environ.get(SBO_ENV, "0").strip().lower()
        return cls(
            enabled=raw not in ("", "0", "false", "off"),
            min_rows=max(1, int(os.environ.get(SBO_MIN_ROWS_ENV, "32"))),
        )


_policy_cache: SboPolicy | None = None

#: One compute-side alternate stream per device. Distinct from the comm
#: stream: SBO moves *compute* (the shared MLP) off the main stream so it
#: runs beside an exchange, while the exchange itself rides the comm pool.
_alt_streams: dict[str, torch.cuda.Stream] = {}


def sbo_alt_stream(device: str | torch.device) -> torch.cuda.Stream:
    """The compute-side alternate stream SBO moves the shared MLP onto.

    Created on first use per device, so a CPU-only or single-stream run never
    pays for it. Callers fence both ways: the alternate stream waits on the
    main stream before reading inputs the main stream produced, and the main
    stream waits on the alternate before consuming the shared MLP's output.
    """
    key = str(device)
    stream = _alt_streams.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _alt_streams[key] = stream
    return stream


def reset_sbo_streams() -> None:
    """Drop the cached alternate streams — test hook between device contexts."""
    _alt_streams.clear()


def sbo_policy() -> SboPolicy:
    """The SBO policy, read once per process.

    An environment lookup per MoE layer would land on the decode hot path; the
    process is the natural lifetime because benchmark arms run as separate
    processes.
    """
    global _policy_cache
    if _policy_cache is None:
        _policy_cache = SboPolicy.from_env()
    return _policy_cache


def reset_sbo_policy() -> None:
    """Forget the cached policy — test hook after monkeypatching the env."""
    global _policy_cache
    _policy_cache = None


class SboFlags:
    """Whether a layer of ``rows`` tokens may overlap its shared MLP.

    sglang's ``SboFlags`` names three overlaps: combine↔down GEMM (tile
    signaled), combine↔shared, and dispatch↔shared. lite_llama ships the third.
    The first needs tile-level synchronization that ``fused_moe``'s scattered
    writes cannot support today (see the module docstring); the second wants
    the same alternate stream the third already occupies. So one predicate
    covers what ships, under the same condition sglang applies to its
    dispatch↔shared pair — the switch, plus enough rows for the exchange to be
    worth hiding.
    """

    @staticmethod
    def enable_dispatch_shared_overlap(rows: int) -> bool:
        """Dispatch exchange overlapping the shared MLP on one stream."""
        policy = sbo_policy()
        return policy.enabled and rows >= policy.min_rows
