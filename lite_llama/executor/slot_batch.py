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
    def begin_prefill(
        self,
        slots: Sequence[int],
        seq_starts: Sequence[int],
        seq_lens: Sequence[int],
    ) -> None:
        """Point the attention metadata at a padded prefill grid for ``slots``.

        The model flattens its ``[n, width]`` token grid row-major, so sequence
        ``i``'s grid column ``j`` must land in slot ``slots[i]``'s cache row
        ``seq_starts[i] + j``: chunked prefill resumes mid-prompt, so each row
        starts at its own absolute position instead of at 0.

        ``seq_lens[i]`` is the sequence's *total* cached length after this chunk
        (start + chunk), which is what bounds attention — positions past a
        sequence's own real tokens are padding: they do write junk K/V into
        rows ``[seq_lens, start + width)`` of that slot, but attention never
        reads past ``b_seq_len``, and later steps overwrite exactly those rows
        in order.

        Args:
            slots: Slot id per sequence, as handed out by the scheduler.
            seq_starts: First cache row each sequence's chunk writes.
            seq_lens: Total cached length per sequence once the chunk lands
                (its prefix within the slot plus this chunk).
        """
        # Grid width is the widest *chunk* in the group, not the span between
        # the earliest start and the latest end — rows start at their own
        # positions, so the grid must be exactly as wide as the widest chunk
        # for cur_select_index to line up with the flattened token grid.
        max_prompt_len = max(
            end - start for start, end in zip(seq_starts, seq_lens, strict=True)
        )
        if max_prompt_len > self.max_seq_len:
            raise ValueError(
                f"prompt length {max_prompt_len} exceeds max_seq_len {self.max_seq_len}"
            )

        n = len(slots)
        b_req_idx = self._to_device(slots)
        starts = self._to_device(seq_starts)
        table = self._atten.b_req_tokens_table

        self._atten.b_req_idx = b_req_idx
        self._atten.b_seq_len = self._to_device(seq_lens)
        self._atten.max_actual_seq_len = max(seq_lens)
        self._atten.is_prefill = True
        # Grid column j of row i maps to cache row starts[i] + j — an outer
        # sum rather than the flat identity used when every chunk started at 0.
        cols = starts.unsqueeze(1) + torch.arange(
            max_prompt_len, device=self.device
        ).unsqueeze(0)
        self._atten.cur_select_index = table[b_req_idx.unsqueeze(1), cols].reshape(-1)
        self._atten.b_start_loc = self._row_offsets[:n] * max_prompt_len

        # A prefill always changes the running set, so the decode after it has to
        # rebuild its own metadata rather than increment this.
        self._host_slots, self._host_lens = [], []

    def begin_extend(
        self,
        slots: Sequence[int],
        seq_starts: Sequence[int],
        seq_lens: Sequence[int],
    ) -> int:
        """Point the attention metadata at one decode row per *token* of the chunks.

        The prefill kernel is pure self-attention over the current grid — it
        cannot see K/V that earlier chunks already wrote into the slot — so a
        chunk resuming mid-prompt runs through the decode kernel instead: each
        (request, position) pair becomes one row, its K/V lands at its cache row
        ``position``, and its query attends over the slot's rows ``[0, position +
        1)``. That is exactly causal extend semantics, paid at one row per token.
        Rows are one token wide, so batch padding keeps the whole pass on the
        captured decode graphs, exactly like a decode step.

        Args:
            slots: Slot id per chunk, as handed out by the scheduler.
            seq_starts: First cache row each chunk writes.
            seq_lens: Total cached length per sequence once the chunk lands.

        Returns:
            The row count actually submitted to the model, filler rows included;
            the caller discards the trailing rows' logits.
        """
        starts, ends = list(seq_starts), list(seq_lens)
        chunk_lens = [end - start for start, end in zip(starts, ends, strict=True)]
        if max(ends) > self.max_seq_len:
            raise ValueError(
                f"sequence length {max(ends)} exceeds max_seq_len {self.max_seq_len}"
            )

        rows_slot, rows_len = self._flatten_rows(slots, starts, chunk_lens)

        # Filler rows pad onto a captured decode graph, same trick as decode.
        # Their fake length tracks the longest real cache so the bucket choice
        # matches; the junk K/V they write stays inside the filler slot.
        filler_len = min(max(ends), self.max_seq_len)
        if self._filler_slot is not None:
            pad = self._runner.graph_batch_size(len(rows_slot)) - len(rows_slot)
            if pad > 0:
                rows_slot = torch.cat(
                    [
                        rows_slot,
                        torch.full((pad,), self._filler_slot, dtype=rows_slot.dtype, device=self.device),
                    ]
                )
                rows_len = torch.cat(
                    [
                        rows_len,
                        torch.full((pad,), filler_len, dtype=rows_len.dtype, device=self.device),
                    ]
                )

        table = self._atten.b_req_tokens_table
        self._b_req_idx = rows_slot
        self._b_seq_len = rows_len
        self._atten.b_req_idx = rows_slot
        self._atten.b_seq_len = rows_len
        self._atten.max_actual_seq_len = max(ends)
        self._atten.is_prefill = False
        # Row `seq_len - 1` is the cache row this row's fresh K/V lands in.
        self._atten.cur_select_index = table[rows_slot, rows_len - 1]
        self._atten.b_start_loc = None

        # A prefill always changes the running set, so the decode after it has
        # to rebuild its own metadata rather than increment this.
        self._host_slots, self._host_lens = [], []
        return len(rows_slot)

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
        self._atten.is_prefill = False
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
        for the caller's request list. Positions fall straight out of it — a
        token written to cache row ``seq_len - 1`` sits at that same absolute
        position in the sequence — which serves both a decode step and the
        one-row-per-token extend batch.
        """
        if self._b_seq_len is None:
            raise RuntimeError("begin_decode() must run before seq_lens is read")
        return self._b_seq_len

    # -------------------------------------------------------------- internals #
    def _flatten_rows(
        self,
        slots: Sequence[int],
        starts: Sequence[int],
        chunk_lens: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand ``(slot, start, len)`` triples into one row per token.

        Returns ``(row_slots, row_lens)`` where row ``r`` of request ``i``'s
        stretch has ``row_slots[r] == slots[i]`` and ``row_lens[r]`` equals its
        absolute position plus one — the cache length once its own K/V lands,
        which is both what the decode kernel reads and, minus one, the row it
        writes. Built with gathers rather than a per-token Python loop so the
        host cost stays flat in the chunk size.
        """
        lens = self._to_device(chunk_lens)
        # Host sum of a list the caller already held: no .item() round-trip.
        total = sum(chunk_lens)
        # Which request each flattened row belongs to, then the row's offset
        # from where that request's stretch begins in the flat index space.
        row_req = torch.repeat_interleave(
            torch.arange(len(slots), device=self.device), lens
        )
        stretch_starts = torch.cumsum(lens, 0) - lens
        within = torch.arange(total, device=self.device) - stretch_starts[row_req]
        row_lens = self._to_device(starts)[row_req] + within + 1
        row_slots = self._to_device(slots)[row_req]
        return row_slots, row_lens

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
