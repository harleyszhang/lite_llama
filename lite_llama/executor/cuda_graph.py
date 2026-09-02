"""CUDA Graph capture and replay for the decode phase.

:class:`CUDAGraphManager` captures one graph per (batch, seq-bucket)
shape via :class:`CUDAGraphRunner`; ``try_replay`` launches the matching
graph when a decode step fits one, else returns None so eager runs.

Lazy mode (O13) captures only a seed pair at startup — batch 1 on the
smallest bucket and the largest batch on the largest bucket — and captures
the remaining shapes the first time a step actually needs them, so cold
start stays seconds-scale instead of waiting for the whole grid. A shape
whose on-demand capture runs out of memory is remembered and never retried;
those steps run eager.

Usage:
    mgr = CUDAGraphManager(model)
    logits = mgr.try_replay(input_ids, position_ids, attn_info)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention_metadata import AttentionMetadata

logger = logging.getLogger(__name__)

# One graph per (batch_size, seq_len_bucket): a graph fixes both the input shapes
# and ``max_actual_seq_len``, so both axes must be enumerated. More buckets cover
# more requests but pin more workspace — a few-hundred-token prompt lands in the
# 512 bucket, anything past the largest bucket falls back to eager.
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_SEQ_LEN_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096)

# ~38 MB per graph measured on a 0.5B fp16 model, rounded up for headroom. The KV
# profiler withholds this much per graph; the OOM fallback in
# ``ModelRunner.enable_cuda_graph`` covers models that exceed the estimate.
WORKSPACE_BYTES_PER_GRAPH: int = 64 * 1024**2

# Lazy mode seeds the grid with these two shapes (O13): the first serves a
# single short request immediately, the second keeps a saturated batch with
# long contexts on the graph path. Everything in between pays ~0.5–1 s once,
# on the first step that needs the shape.
LAZY_SEED_SHAPES: int = 2


def estimate_capture_workspace(max_seq_len: int, *, lazy: bool = False) -> int:
    """Upper bound on the bytes the capture plan will pin.

    An upper bound by necessity: the KV profiler runs before ``max_request_num``
    exists, so every default batch size is assumed to survive clamping. Lazy
    mode (O13) reserves only the seed pair — graphs captured on demand take
    their workspace from whatever is free at that moment, and a shape whose
    capture OOMs simply stays eager rather than dragging the KV pool down for
    shapes the workload may never produce.
    """
    n_buckets = sum(1 for b in DEFAULT_SEQ_LEN_BUCKETS if b <= max_seq_len)
    graphs = LAZY_SEED_SHAPES if lazy else len(DEFAULT_BATCH_SIZES) * max(n_buckets, 1)
    return graphs * WORKSPACE_BYTES_PER_GRAPH


@dataclass(frozen=True)
class _GraphKey:
    """Identifies one captured graph by its persistent shape."""

    batch_size: int
    seq_len_bucket: int


class CUDAGraphRunner:
    """One captured decode step for a fixed ``(batch_size, seq_len_bucket)`` pair.

    All tensors the graph reads or writes live inside the runner. Callers push
    new values in via :meth:`replay` and receive the output logits back.

    Args:
        model: The eager :class:`torch.nn.Module` to capture.
        batch_size: Number of sequences the captured graph serves.
        seq_len_bucket: Value used for ``max_actual_seq_len`` at capture time;
            replay is only valid when the current context length is ``<=`` this.
        kv_buffer: Layer-wise paged KV cache tensors (shared with the executor).
        b_req_tokens_table: Request-to-cache-row mapping (shared with the executor).
        device: Torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        batch_size: int,
        seq_len_bucket: int,
        kv_buffer: list[torch.Tensor],
        b_req_tokens_table: torch.Tensor,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.seq_len_bucket = seq_len_bucket
        self.device = device

        # Persistent input surface — everything the graph reads goes through these.
        self.input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        self.atten_info = AttentionMetadata()
        self.atten_info.kv_buffer = kv_buffer  # shared list; storage is persistent
        self.atten_info.b_req_tokens_table = b_req_tokens_table
        self.atten_info.cur_select_index = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.atten_info.b_seq_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.atten_info.b_req_idx = torch.arange(batch_size, dtype=torch.long, device=device)

        # A Python int, so it is baked in: it fixes flash_decoding's mid_o shape and grid.
        self.atten_info.max_actual_seq_len = seq_len_bucket
        self.atten_info.is_prefill = False

        self._graph: torch.cuda.CUDAGraph | None = None
        self._output: torch.Tensor | None = None

    def capture(self, warmup_metadata: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        """Warm up on a side stream, then record the graph on the current stream.

        ``warmup_metadata`` is ``(b_req_idx, cur_select_index)`` from the live
        step that triggered an on-demand (lazy) capture: the warmup forwards
        write their throwaway K/V exactly where that step's real pass is about
        to write, so the replay that immediately follows overwrites the garbage
        and no other request's rows are touched. ``None`` (capture at startup)
        keeps the zero buffers — the cache is empty, so rows 0..batch are safe.
        """
        if warmup_metadata is not None:
            b_req_idx, cur_select_index = warmup_metadata
            # Warm up on real work: at zero length stage 1 visits no K/V rows, so any
            # fault would surface inside the capture rather than in the warmup below.
            self.atten_info.b_req_idx.copy_(b_req_idx)
            self.atten_info.cur_select_index.copy_(cur_select_index)
            # The longest legal length walks every split branch the kernel has,
            # and writes at a position the real request has not reached yet.
            self.atten_info.b_seq_len.fill_(min(self.seq_len_bucket, self.atten_info.b_req_tokens_table.shape[1] - 1))
        else:
            # Warm up on real work: at zero length stage 1 visits no K/V rows, so any
            # fault would surface inside the capture rather than in the warmup below.
            self.atten_info.b_seq_len.fill_(min(self.seq_len_bucket, 32))

        # The capture stream must be idle, so warmup runs on its own stream, fenced
        # both ways. Those passes force Triton JIT, cuBLAS workspaces and allocator
        # blocks to happen *before* the capture, which cannot allocate.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _ = self.model(self.input_ids, self.position_ids, self.atten_info)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._output = self.model(self.input_ids, self.position_ids, self.atten_info)

    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cur_select_index: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_req_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Push new inputs into the persistent buffers and replay the graph.

        The caller is responsible for keeping the current ``max_actual_seq_len``
        no greater than this runner's :attr:`seq_len_bucket`; the executor
        enforces that via bucket selection.
        """
        if self._graph is None or self._output is None:
            raise RuntimeError("capture() must be called before replay()")

        # ``.copy_()`` writes into the SAME storage the graph captured, so the
        # graph's baked pointers keep pointing at valid values.
        self.input_ids.copy_(input_ids.view(self.batch_size, 1))
        self.position_ids.copy_(position_ids.view(self.batch_size, 1))
        self.atten_info.cur_select_index.copy_(cur_select_index)
        self.atten_info.b_seq_len.copy_(b_seq_len)
        self.atten_info.b_req_idx.copy_(b_req_idx.to(self.atten_info.b_req_idx.dtype))

        self._graph.replay()
        return self._output


class CUDAGraphManager:
    """Holds one :class:`CUDAGraphRunner` per ``(batch_size, seq_len_bucket)``.

    The manager also captures each layer's ``atten_info.b_req_tokens_table`` update
    that decode performs *outside* the graph — ``update_kv_index`` writes into a
    persistent tensor, so it stays out of the recorded region.

    Args:
        model: The eager model to capture.
        kv_buffer: Layer-wise KV cache tensors owned by the executor.
        b_req_tokens_table: Request-to-cache-row map owned by the executor.
        batch_sizes: Batch sizes to capture; smaller decode batches fall back to eager.
        seq_len_buckets: Ascending ``max_actual_seq_len`` ceilings to capture.
        device: Torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        kv_buffer: list[torch.Tensor],
        b_req_tokens_table: torch.Tensor,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
        seq_len_buckets: tuple[int, ...] = DEFAULT_SEQ_LEN_BUCKETS,
        device: str = "cuda",
        lazy: bool = False,
    ) -> None:
        self.model = model
        self.kv_buffer = kv_buffer
        self.b_req_tokens_table = b_req_tokens_table
        self.batch_sizes = tuple(sorted(set(batch_sizes)))
        self.seq_len_buckets = tuple(sorted(set(seq_len_buckets)))
        self.device = device
        self._lazy = lazy
        self._runners: dict[_GraphKey, CUDAGraphRunner] = {}
        # Shapes whose on-demand capture already failed (usually OOM): never
        # retried, those steps run eager instead of paying the attempt again.
        self._failed: set[_GraphKey] = set()

    def capture_all(self) -> None:
        """Capture a graph for every ``(batch_size, seq_len_bucket)`` pair."""
        for bs in self.batch_sizes:
            for bucket in self.seq_len_buckets:
                key = _GraphKey(bs, bucket)
                runner = CUDAGraphRunner(
                    self.model,
                    batch_size=bs,
                    seq_len_bucket=bucket,
                    kv_buffer=self.kv_buffer,
                    b_req_tokens_table=self.b_req_tokens_table,
                    device=self.device,
                )
                runner.capture()
                self._runners[key] = runner

    def capture_seed(self) -> None:
        """Capture only the seed pair; the rest wait for their first use.

        The seeds are batch 1 on the smallest bucket (a single fresh request
        starts on the graph path immediately) and the largest batch on the
        largest bucket (a saturated batch with long contexts too). Under load
        the in-between shapes are captured on demand inside :meth:`try_replay`.
        """
        if not self.batch_sizes or not self.seq_len_buckets:
            return
        seeds = (
            _GraphKey(self.batch_sizes[0], self.seq_len_buckets[0]),
            _GraphKey(self.batch_sizes[-1], self.seq_len_buckets[-1]),
        )
        for key in dict.fromkeys(seeds):  # de-duplicated, order kept
            runner = CUDAGraphRunner(
                self.model,
                batch_size=key.batch_size,
                seq_len_bucket=key.seq_len_bucket,
                kv_buffer=self.kv_buffer,
                b_req_tokens_table=self.b_req_tokens_table,
                device=self.device,
            )
            runner.capture()
            self._runners[key] = runner

    def _on_grid(self, key: _GraphKey) -> bool:
        """Whether the key belongs to the configured capture grid."""
        return key.batch_size in self.batch_sizes and key.seq_len_bucket in self.seq_len_buckets

    def _capture_on_miss(
        self, key: _GraphKey, atten_info: AttentionMetadata
    ) -> CUDAGraphRunner | None:
        """Capture a missing shape right where a step asked for it (O13).

        The warmup borrows the live step's ``b_req_idx`` and
        ``cur_select_index`` so its throwaway writes land exactly where the
        real pass — which replays immediately after — is about to write. A
        failure (typically OOM) blacklists the shape and this step runs eager.
        """
        if key in self._failed or not self._on_grid(key):
            return None
        try:
            # The capture stream must be the only work in flight: the pipeline's
            # readback copies and any pending kernels must land first.
            torch.cuda.synchronize()
            runner = CUDAGraphRunner(
                self.model,
                batch_size=key.batch_size,
                seq_len_bucket=key.seq_len_bucket,
                kv_buffer=self.kv_buffer,
                b_req_tokens_table=self.b_req_tokens_table,
                device=self.device,
            )
            runner.capture(
                warmup_metadata=(atten_info.b_req_idx, atten_info.cur_select_index)
            )
            self._runners[key] = runner
            logger.info(
                "Lazy-captured decode graph batch=%d bucket=%d on first use "
                "(%d/%d shapes captured)",
                key.batch_size,
                key.seq_len_bucket,
                len(self._runners),
                len(self.batch_sizes) * len(self.seq_len_buckets),
            )
            return runner
        except torch.cuda.OutOfMemoryError:
            self._failed.add(key)
            logger.warning(
                "Lazy capture of decode graph batch=%d bucket=%d ran out of "
                "memory; that shape stays eager",
                key.batch_size,
                key.seq_len_bucket,
            )
            return None

    def _pick_bucket(self, current_max_seq_len: int) -> int | None:
        """Smallest bucket ceiling that fits; ``None`` if the request is too long."""
        for bucket in self.seq_len_buckets:
            if bucket >= current_max_seq_len:
                return bucket
        return None

    def pad_to(self, batch_size: int) -> int | None:
        """Smallest captured batch size that can absorb ``batch_size``.

        Continuous batching submits whatever batch the workload produces, which
        rarely lands on the captured grid. Padding the batch out to a captured
        size and discarding the filler rows keeps those steps on the graph path;
        the extra rows cost a little attention work, which is far less than the
        ~300 kernel launches an eager decode step pays. Returns ``None`` when the
        batch is larger than anything captured.
        """
        for bs in self.batch_sizes:
            if bs >= batch_size:
                return bs
        return None

    def try_replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionMetadata,
    ) -> torch.Tensor | None:
        """Run the matching captured graph if one exists, else return ``None``.

        Only decode steps (``seq_len == 1``) with a supported batch size and a
        max-sequence length that fits inside the largest bucket are eligible; the
        caller must fall back to eager execution when this returns ``None``.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len != 1:
            return None
        bucket = self._pick_bucket(atten_info.max_actual_seq_len)
        if bucket is None:
            return None
        key = _GraphKey(batch_size, bucket)
        runner = self._runners.get(key)
        if runner is None and self._lazy:
            runner = self._capture_on_miss(key, atten_info)
        if runner is None:
            return None
        return runner.replay(
            input_ids,
            position_ids,
            atten_info.cur_select_index,
            atten_info.b_seq_len,
            atten_info.b_req_idx,
        )
