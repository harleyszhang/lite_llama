"""CUDA Graph capture and replay for the decode phase.

Why this file exists
--------------------
The previous implementation (deleted before this refactor) produced garbage output
once a second decode step ran. Root cause: CUDA Graph records *tensor pointers*
at capture time. The decode path in :class:`ModelExecutor` used to reassign

    self.atten_info.cur_select_index = self.kv_mem_manager.alloc_kvcache_index(...)

on every step, handing the model a freshly allocated tensor each time. The graph
still held pointers into the *first-step* tensors, so subsequent replays read
whatever the caching allocator had recycled into those addresses — random KV
cache row indices, garbage attention output.

The fix has three parts, implemented below:

1. **Persistent buffers.** :class:`CUDAGraphRunner` owns its own ``input_ids`` /
   ``position_ids`` / ``cur_select_index`` / ``b_seq_len`` / ``b_req_idx`` tensors
   for the whole generation. Each replay ``.copy_()`` new values in place; the
   graph's baked pointers stay valid because the underlying storage never moves.

2. **Bucketing by ``max_actual_seq_len``.** The FlashDecoding kernel allocates
   intermediate ``mid_o`` / ``mid_o_logexpsum`` tensors sized by
   ``max_num_partitions = ceil(max_actual_seq_len / PARTITION_SIZE)``. Those
   allocations are baked into the graph, so replay must fit within the captured
   size. We capture one graph per ``(batch_size, seq_len_bucket)`` pair; at replay
   we round the current ``max_actual_seq_len`` up to the nearest bucket ceiling.

3. **Decode-only capture.** Prefill uses a different attention kernel and the
   sequence length varies per prompt, so it is never graph-friendly. Graphs are
   captured only for ``seq_len == 1``; prefill always falls through to eager.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .executor_struct import AttentionInfo

# Default buckets balance capture memory (each graph pins tens of MB of
# workspace) against replay coverage.  A prompt of a few hundred tokens fits
# in the 512 bucket; long contexts fall through to eager once past the
# largest bucket.
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_SEQ_LEN_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096)

# Measured on a 0.5B fp16 model: ~38 MB per captured graph. Reserve 64 MB per
# graph so the KV-cache profiler leaves headroom for capture; the OOM fallback
# in ``ModelExecutor.enable_cuda_graph`` covers models that exceed the estimate.
WORKSPACE_BYTES_PER_GRAPH: int = 64 * 1024**2


def estimate_capture_workspace(max_seq_len: int) -> int:
    """Upper-bound bytes the default capture grid will pin on this model.

    The KV profiler runs before any request index bound is known, so the
    estimate conservatively assumes every default batch size survives clamping.
    """
    n_buckets = sum(1 for b in DEFAULT_SEQ_LEN_BUCKETS if b <= max_seq_len)
    return len(DEFAULT_BATCH_SIZES) * max(n_buckets, 1) * WORKSPACE_BYTES_PER_GRAPH


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

        self.atten_info = AttentionInfo()
        self.atten_info.kv_buffer = kv_buffer  # shared list; storage is persistent
        self.atten_info.b_req_tokens_table = b_req_tokens_table
        self.atten_info.cur_select_index = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.atten_info.b_seq_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.atten_info.b_req_idx = torch.arange(batch_size, dtype=torch.long, device=device)
        # Python int, gets baked into the launched kernel: sets mid_o's shape.
        self.atten_info.max_actual_seq_len = seq_len_bucket

        self._graph: torch.cuda.CUDAGraph | None = None
        self._output: torch.Tensor | None = None

    def capture(self) -> None:
        """Warm up on a side stream, then record the graph on the current stream."""
        # Seed b_seq_len with a plausible non-zero value so that the FlashDecoding
        # kernel actually visits the K/V rows during capture — otherwise Triton's
        # autotuner may cache a degenerate specialization keyed on zero-length work.
        self.atten_info.b_seq_len.fill_(min(self.seq_len_bucket, 32))

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
    ) -> None:
        self.model = model
        self.kv_buffer = kv_buffer
        self.b_req_tokens_table = b_req_tokens_table
        self.batch_sizes = tuple(sorted(set(batch_sizes)))
        self.seq_len_buckets = tuple(sorted(set(seq_len_buckets)))
        self.device = device
        self._runners: dict[_GraphKey, CUDAGraphRunner] = {}

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

    def _pick_bucket(self, current_max_seq_len: int) -> int | None:
        """Smallest bucket ceiling that fits; ``None`` if the request is too long."""
        for bucket in self.seq_len_buckets:
            if bucket >= current_max_seq_len:
                return bucket
        return None

    def try_replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionInfo,
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
        runner = self._runners.get(_GraphKey(batch_size, bucket))
        if runner is None:
            return None
        return runner.replay(
            input_ids,
            position_ids,
            atten_info.cur_select_index,
            atten_info.b_seq_len,
            atten_info.b_req_idx,
        )
