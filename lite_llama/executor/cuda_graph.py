"""CUDA Graph capture and replay for the decode phase.

A CUDA Graph bakes in *tensor pointers* at capture time, so the decode path must
never reallocate the tensors the model reads. Three rules make that hold:
persistent input buffers that each replay ``copy_()`` into in place; one graph per
``(batch_size, seq_len_bucket)`` because FlashDecoding sizes scratch by
``max_actual_seq_len``; and decode-only capture (``seq_len == 1``), with prefill
always eager.

Usage:
    mgr = CUDAGraphManager(...); logits = mgr.decode(input_ids, positions, ...)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention_metadata import AttentionMetadata

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


def estimate_capture_workspace(max_seq_len: int) -> int:
    """Upper bound on the bytes the default capture grid will pin.

    An upper bound by necessity: the KV profiler runs before ``max_request_num``
    exists, so every default batch size is assumed to survive clamping.
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

        self.atten_info = AttentionMetadata()
        self.atten_info.kv_buffer = kv_buffer  # shared list; storage is persistent
        self.atten_info.b_req_tokens_table = b_req_tokens_table
        self.atten_info.cur_select_index = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.atten_info.b_seq_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.atten_info.b_req_idx = torch.arange(batch_size, dtype=torch.long, device=device)
        # A Python int, so it is baked in: it fixes flash_decoding's mid_o shape and grid.
        self.atten_info.max_actual_seq_len = seq_len_bucket

        self._graph: torch.cuda.CUDAGraph | None = None
        self._output: torch.Tensor | None = None

    def capture(self) -> None:
        """Warm up on a side stream, then record the graph on the current stream."""
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
