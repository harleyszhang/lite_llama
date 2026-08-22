"""Slot-based KV layout and per-step attention metadata for continuous batching.

The one-shot batch path (:class:`~lite_llama.engine.llm_engine._DecodeSession`)
can bump-allocate cache rows because it owns the whole cache and only ever
appends. Continuous batching cannot: requests join and leave mid-flight, so
every step would hit :meth:`KVCacheManager.alloc_contiguous_kvcache`, whose
``nonzero`` scan plus two ``.item()`` reads costs three device synchronisations
per decode step. This module removes the allocator from the decode path
entirely.

Usage:
    batch = SlotBatch(runner)
    batch.begin_prefill([0, 1], [17, 23]); logits = runner.forward(...)
    padded = batch.begin_decode([0, 1], [18, 24]); logits = runner.forward(...)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .model_runner import ModelRunner


class SlotBatch:
    """Continuous-batching view of the KV cache: fixed slot regions, stable metadata.

    Two decisions keep a steady-state decode step free of host-device traffic.

    **Fixed slot regions.** Slot ``s`` permanently owns cache rows
    ``[s * max_seq_len, (s + 1) * max_seq_len)``, so ``b_req_tokens_table`` is the
    identity map and is written once here instead of being patched by
    ``update_kv_index`` on every step. Reserving a request's cache becomes a
    host-side slot handout and releasing it a host-side push: no allocator search,
    no fragmentation, no synchronisation. The price is that a slot reserves
    ``max_seq_len`` rows whether the request fills them or not, which caps
    concurrency at :attr:`num_slots`; the paged allocator in
    :class:`~lite_llama.executor.kv_cache_manager.KVCacheManager` stays the denser
    choice for the one-shot batch path, which knows all its prompts up front.

    **Composition-stable metadata.** ``b_req_idx`` and ``b_seq_len`` are rebuilt
    from the host only when a request joins or leaves. While the running set holds
    steady a step just increments the lengths on-device, and ``cur_select_index``
    falls out of a gather against the slot table — so the metadata for a decode
    step costs two device kernels and no transfers.

    Args:
        runner: The executor whose cache, slot table and attention metadata this
            object drives. Constructing a :class:`SlotBatch` takes the cache over:
            the rows behind the slot table are claimed from the paged allocator.
    """

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self._atten = runner.atten_info
        self.device = runner.device
        self.max_seq_len = runner.max_seq_len

        table = runner.b_req_tokens_table
        total_slots, row_len = table.shape

        # The last slot is not handed to requests: it backs the filler rows that
        # pad a decode batch up to a captured CUDA-graph size. One slot is a cheap
        # price for keeping odd batch sizes on the graph path. With a single slot
        # there is nothing to spare, so padding is switched off instead.
        self._filler_slot: int | None = total_slots - 1 if total_slots > 1 else None
        self.num_slots = total_slots - 1 if total_slots > 1 else 1

        table.copy_(
            torch.arange(total_slots * row_len, dtype=table.dtype, device=self.device).view(
                total_slots, row_len
            )
        )
        if self._filler_slot is not None:
            # Filler rows attend over this region. Uninitialised fp16 can hold
            # NaN, and while a NaN there cannot reach a real sequence (no kernel
            # reduces across batch rows), a cache full of them makes any future
            # debugging session lie to you.
            start = self._filler_slot * row_len
            for layer in runner.kv_cache_manager.gpu_kv_buffer:
                layer[start : start + row_len].zero_()

        # Every row the table names now belongs to a slot; take them out of the
        # paged allocator so the two schemes cannot hand out the same row.
        runner.kv_cache_manager.claim(total_slots * row_len)

        # Offsets of each sequence in a flattened prefill grid, scaled per call.
        self._row_offsets = torch.arange(total_slots, dtype=torch.int32, device=self.device)

        # Device metadata plus the host mirror used to decide whether the next
        # step can reuse it.
        self._b_req_idx: torch.Tensor | None = None
        self._b_seq_len: torch.Tensor | None = None
        self._host_slots: list[int] = []
        self._host_lens: list[int] = []

    # ------------------------------------------------------------------ steps #
    def begin_prefill(self, slots: Sequence[int], prompt_lens: Sequence[int]) -> None:
        """Point the attention metadata at a padded prefill grid for ``slots``.

        The model flattens its ``[n, max_prompt_len]`` token grid row-major, so
        sequence ``i``'s token ``j`` must land in slot ``slots[i]``'s row ``j``.
        Positions past a sequence's own prompt are padding: they do write junk K/V
        into rows ``[prompt_len, max_prompt_len)`` of that slot, but attention
        never reads past ``b_seq_len``, and the sequence's own decode steps
        overwrite exactly those rows in order.

        Args:
            slots: Slot id per sequence, as handed out by the scheduler.
            prompt_lens: Real (unpadded) prompt length per sequence.
        """
        max_prompt_len = max(prompt_lens)
        if max_prompt_len > self.max_seq_len:
            raise ValueError(
                f"prompt length {max_prompt_len} exceeds max_seq_len {self.max_seq_len}"
            )

        n = len(slots)
        b_req_idx = self._to_device(slots)
        table = self._atten.b_req_tokens_table

        self._atten.b_req_idx = b_req_idx
        self._atten.b_seq_len = self._to_device(prompt_lens)
        self._atten.max_actual_seq_len = max_prompt_len
        self._atten.cur_select_index = table[b_req_idx, :max_prompt_len].reshape(-1)
        self._atten.b_start_loc = self._row_offsets[:n] * max_prompt_len

        # A prefill always changes the running set, so the decode after it has to
        # rebuild its own metadata rather than increment this.
        self._host_slots, self._host_lens = [], []

    def begin_decode(self, slots: Sequence[int], seq_lens: Sequence[int]) -> int:
        """Point the attention metadata at one decode step and report the batch size.

        Args:
            slots: Slot id per running request.
            seq_lens: Length each sequence will have *after* this step's token, i.e.
                the position the new K/V is written to is ``seq_lens[i] - 1``.

        Returns:
            The batch size actually submitted to the model. It exceeds
            ``len(slots)`` when the batch was padded up to a captured CUDA-graph
            size; the caller must discard the trailing logits rows.
        """
        padded_slots, padded_lens = self._pad(slots, seq_lens)

        if padded_slots == self._host_slots and padded_lens == [
            length + 1 for length in self._host_lens
        ]:
            # Same requests, one token further along: the device tensors already
            # hold the previous step's values, so advance them in place. This is
            # the steady state, and it moves nothing across the PCIe bus.
            self._b_seq_len += 1
        else:
            self._b_req_idx = self._to_device(padded_slots)
            self._b_seq_len = self._to_device(padded_lens)
        self._host_slots, self._host_lens = padded_slots, padded_lens

        table = self._atten.b_req_tokens_table
        self._atten.b_req_idx = self._b_req_idx
        self._atten.b_seq_len = self._b_seq_len
        self._atten.max_actual_seq_len = max(seq_lens)
        # Row `seq_len - 1` of each slot's region: where this step's K/V goes.
        self._atten.cur_select_index = table[self._b_req_idx, self._b_seq_len - 1]
        self._atten.b_start_loc = None
        return len(padded_slots)

    def reset(self) -> None:
        """Forget the running set so the next decode rebuilds its metadata."""
        self._host_slots, self._host_lens = [], []

    @property
    def seq_lens(self) -> torch.Tensor:
        """Cache length per row of the batch most recently submitted to the model.

        Includes any filler rows, so it is shaped for the model call rather than
        for the caller's request list. Decode positions fall straight out of it:
        a token written to cache row ``seq_len - 1`` sits at that same absolute
        position in the sequence.
        """
        if self._b_seq_len is None:
            raise RuntimeError("begin_decode() must run before seq_lens is read")
        return self._b_seq_len

    # -------------------------------------------------------------- internals #
    def _pad(self, slots: Sequence[int], seq_lens: Sequence[int]) -> tuple[list[int], list[int]]:
        """Grow the batch to the next captured CUDA-graph size, if there is one.

        Continuous batching produces whatever batch size the workload happens to
        have, while graphs are captured for a fixed grid, so an unpadded batch of
        7 would fall back to eager decode and give up most of the graph's win.
        Filler rows point at the reserved slot and carry the batch's own maximum
        length, which keeps every row advancing by exactly one token per step and
        so keeps the in-place fast path in :meth:`begin_decode` alive.
        """
        slots, seq_lens = list(slots), list(seq_lens)
        if self._filler_slot is None:
            return slots, seq_lens

        target = self._runner.graph_batch_size(len(slots))
        pad = target - len(slots)
        if pad <= 0:
            return slots, seq_lens

        filler_len = min(max(seq_lens), self.max_seq_len)
        return slots + [self._filler_slot] * pad, seq_lens + [filler_len] * pad

    def _to_device(self, values: Sequence[int]) -> torch.Tensor:
        """Upload a host list as a fresh int64 tensor.

        Deliberately a new allocation rather than a write into a reused staging
        buffer: the previous step's tensor may still be queued for the GPU, and
        overwriting it from the host would race with kernels that have not run
        yet. This only happens when the running set changes, so the copy is off
        the steady-state path.
        """
        return torch.tensor(list(values), dtype=torch.long, device=self.device)
