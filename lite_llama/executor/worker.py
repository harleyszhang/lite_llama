"""Model worker: runs one :class:`ModelInput` — forward *and* sampling — on this rank.

Core design
-----------
The seam between "decide what to run" and "run it" cannot sit between the forward
pass and the sampler. With a vocabulary-parallel head the sampler is itself a
collective (:mod:`lite_llama.engine.sampler` reduces and gathers to make the draw),
so every tensor-parallel rank must execute it. The unit of work is therefore
*forward + sample*, and what crosses the boundary is a :class:`ModelInput`: pure,
picklable data describing the work, never a device tensor.

Layout is *derived* here rather than shipped. Positions, grid width, graph padding
and the logits row of each sampled sequence all follow from
``(slots, seq_starts, seq_lens)``, identically on every rank — so the control plane
stays a few hundred bytes, and exactly one place decides how a plan becomes tensors.
That is what lets the single-process and multi-process executors share this class
instead of growing two copies of the same arithmetic that can drift apart.

Usage:
    worker = ModelWorker(llm_engine, max_num_seqs=32, max_seq_len=2048)
    tokens = worker.execute(plan)  # [len(plan.sampled)] sampled ids, on device
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch

from ..distributed.parallel_state import broadcast_tp, get_tp_world_size
from ..engine.sampler import BatchedSamplingParams, GeneratedSpan, SamplingParams

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..engine.llm_engine import LLMEngine


class PassKind(StrEnum):
    """Which kernel path a plan takes through attention.

    Attributes:
        PREFILL: A padded ``[sequences, width]`` grid through the prefill kernel.
            Pure self-attention over the grid, so it is only correct when nothing
            of the sequence is cached yet.
        EXTEND: One decode-style row per token, each query attending over its
            slot's whole cached history — what a prompt chunk resuming on top of
            a cached prefix needs, at one row per token.
        DECODE: One row per sequence, one token each.
    """

    PREFILL = "prefill"
    EXTEND = "extend"
    DECODE = "decode"


@dataclass(frozen=True)
class ModelInput:
    """One model pass, as data: what to run, not how to lay it out.

    Every field is a tuple of ints (or of frozen :class:`SamplingParams`), which
    makes a plan cheap to compare, safe to hold on to, and picklable — the last
    being what lets the same object be broadcast to tensor-parallel workers.

    Sequences are described by ``(slots[i], seq_starts[i], seq_lens[i])``: slot
    ``slots[i]`` receives cache rows ``[seq_starts[i], seq_lens[i])`` this pass, so
    each sequence's chunk length falls out as the difference. ``tokens`` is those
    chunks concatenated with no padding — padding is a property of the kernel the
    pass will use, and is added by :class:`ModelWorker`.

    Sampling is described separately because it covers a *subset*: a chunked
    prompt that did not finish has no next token to draw. ``sampled`` indexes into
    ``slots`` in ascending order, and ``sampling``/``gen_counts`` are parallel to
    it.

    Attributes:
        kind: Which kernel path the pass takes.
        slots: Cache slot per sequence.
        seq_starts: First cache row this pass writes, per sequence.
        seq_lens: Total cached length per sequence once this pass lands.
        tokens: Input ids, all sequences' chunks concatenated.
        sampling: Sampling knobs, one per sampled sequence.
        sampled: Indices into ``slots`` of the sequences to sample, ascending.
        gen_counts: Tokens already generated per sampled sequence. Doubles as the
            column its next token is written to and as the width of its
            repetition-penalty window.
        prefix_copies: ``(src_slot, dst_slot, start_token, num_tokens)`` runs of
            prefix-cache K/V to copy in before the pass runs, so its extend rows
            can attend over a prefix they never computed. Empty on a miss. Every
            tensor-parallel rank replays them against its own shard, which is why
            they travel with the plan rather than being applied by the driver.
    """

    kind: PassKind
    slots: tuple[int, ...]
    seq_starts: tuple[int, ...]
    seq_lens: tuple[int, ...]
    tokens: tuple[int, ...]
    sampling: tuple[SamplingParams, ...]
    sampled: tuple[int, ...]
    gen_counts: tuple[int, ...]
    prefix_copies: tuple[tuple[int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("a ModelInput must cover at least one sequence")
        if not len(self.slots) == len(self.seq_starts) == len(self.seq_lens):
            raise ValueError("slots, seq_starts and seq_lens must describe the same sequences")
        if len(self.tokens) != sum(self.chunk_lens):
            raise ValueError(f"got {len(self.tokens)} tokens for {sum(self.chunk_lens)} cache rows")
        if not len(self.sampled) == len(self.sampling) == len(self.gen_counts):
            raise ValueError("sampled, sampling and gen_counts must describe the same rows")

    @property
    def chunk_lens(self) -> tuple[int, ...]:
        """Tokens this pass feeds per sequence."""
        return tuple(end - start for start, end in zip(self.seq_starts, self.seq_lens, strict=True))


def _sync_tp(tokens: torch.Tensor) -> torch.Tensor:
    """Broadcast rank 0's sampled ids so every TP rank continues identically.

    Non-greedy sampling draws from a per-rank RNG, so without this the ranks would
    disagree about the token they just produced and every later step would compound
    the divergence.
    """
    if get_tp_world_size() > 1:
        return broadcast_tp(tokens)
    return tokens


class ModelWorker:
    """Executes plans against this rank's model shard and KV cache.

    Holds the only mutable execution state: the fixed-slot KV view and the
    ``[num_slots, max_seq_len]`` grid of generated tokens that the repetition
    penalty reads. Both are indexed by *slot*, never by position in a batch, so
    nothing here has to be invalidated when the running set changes — a plan
    always names the slots it means.

    Args:
        engine: A built :class:`~lite_llama.engine.llm_engine.LLMEngine`; the
            worker takes its KV cache over via the slot view.
        max_num_seqs: Concurrency ceiling, which caps the slots handed out and so
            the height of the generated-token grid.
        max_seq_len: Context bound, and the grid's width.
    """

    def __init__(self, engine: LLMEngine, max_num_seqs: int, max_seq_len: int) -> None:
        self._runner = engine.model_runner
        self._sampler = engine.sampler
        self._device = engine.device
        self._pad_id = engine.pad_id
        self._slot_batch = self._runner.enable_slot_kv_cache()

        # Slot ids stay below max_num_seqs, which keeps the generated-token grid
        # proportional to the concurrency the caller asked for rather than to
        # however many slots happen to fit in the cache.
        self.num_slots = min(self._slot_batch.num_slots, max_num_seqs)
        self._gen_grid = torch.zeros(
            (self.num_slots, max_seq_len), dtype=torch.long, device=self._device
        )
        self._columns = torch.arange(max_seq_len, device=self._device)
        self._no_tokens = torch.empty(0, dtype=torch.long, device=self._device)

        # Sampling knobs cost four small uploads to rebuild, so a steady batch
        # reuses them; the key is the plan's own rows, which needs no cooperation
        # from whoever mutates the running set.
        self._sampling_key: tuple[tuple[float, float, float, bool], ...] | None = None
        self._sampling: BatchedSamplingParams | None = None

    @torch.inference_mode()
    def execute(self, model_input: ModelInput) -> torch.Tensor:
        """Run one pass and return its sampled tokens.

        Returns:
            ``[len(model_input.sampled)]`` token ids on this rank's device,
            identical across tensor-parallel ranks. A pass whose sequences all
            still owe tokens (a prompt chunk that did not finish) runs the model —
            its K/V has to land — and returns an empty tensor.
        """
        # Before the forward, so extend rows resuming on a reused prefix find it
        # in their own slot. Same stream, so the ordering needs no synchronisation.
        self._slot_batch.copy_prefix(model_input.prefix_copies)
        logits = self._forward(model_input)
        if logits is None:
            return self._no_tokens
        return self._sample(model_input, logits)

    # ---------------------------------------------------------------- forwards #
    def _forward(self, plan: ModelInput) -> torch.Tensor | None:
        match plan.kind:
            case PassKind.PREFILL:
                return self._forward_grid(plan)
            case PassKind.EXTEND:
                return self._forward_extend(plan)
            case PassKind.DECODE:
                return self._forward_decode(plan)

    def _forward_grid(self, plan: ModelInput) -> torch.Tensor | None:
        """A padded token grid through the prefill kernel."""
        chunk_lens = plan.chunk_lens
        width = max(chunk_lens)
        grid, offset = [], 0
        for chunk in chunk_lens:
            grid.append(plan.tokens[offset : offset + chunk] + (self._pad_id,) * (width - chunk))
            offset += chunk

        input_ids = torch.tensor(grid, dtype=torch.long, device=self._device)
        # Padded columns run past a row's real position, but attention never reads
        # past that row's b_seq_len, so the junk positions are inert.
        positions = self._to_device(plan.seq_starts).unsqueeze(1) + torch.arange(
            width, device=self._device
        )

        self._slot_batch.begin_prefill(plan.slots, plan.seq_starts, plan.seq_lens)
        # A row's next-token logits sit at its own last real *column* rather than
        # at the end of the padded row. The model gathers one column per row, so
        # the whole grid is gathered here and the sampled subset selected after.
        logits = self._runner.forward(
            input_ids, positions, None, logits_positions=self._to_device(chunk_lens) - 1
        )
        return self._pick(logits, plan.sampled, len(plan.slots))

    def _forward_extend(self, plan: ModelInput) -> torch.Tensor | None:
        """Chunks resuming on a cached prefix: one decode-style row per token."""
        padded = self._slot_batch.begin_extend(plan.slots, plan.seq_starts, plan.seq_lens)
        input_ids = self._rows(plan.tokens, padded)
        # begin_extend set b_seq_len to each row's absolute position plus one, so
        # the position of the token it feeds is exactly that minus one.
        positions = (self._slot_batch.seq_lens - 1).view(-1, 1)

        logits = self._runner.forward(input_ids, positions, None)
        # One row per token: a sequence's next-token logits are on the last row of
        # its own stretch of the flattened batch.
        ends = list(itertools.accumulate(plan.chunk_lens))
        rows = tuple(ends[index] - 1 for index in plan.sampled)
        return self._pick(logits[:, -1, :], rows, padded)

    def _forward_decode(self, plan: ModelInput) -> torch.Tensor | None:
        """One token for every sequence in the plan."""
        rows = len(plan.slots)
        padded = self._slot_batch.begin_decode(plan.slots, plan.seq_lens)
        input_ids = self._rows(plan.tokens, padded)
        # The token being fed sits at its own cache row, i.e. length minus one.
        positions = self._slot_batch.seq_lens.view(-1, 1) - 1

        logits = self._runner.forward(input_ids, positions, None)
        return self._pick(logits[:rows, -1, :], plan.sampled, rows)

    # ---------------------------------------------------------------- sampling #
    def _sample(self, plan: ModelInput, logits: torch.Tensor) -> torch.Tensor:
        """Draw one token per sampled row and record it in the generated grid."""
        sampling = self._batched_sampling(plan.sampling)
        slots = self._to_device([plan.slots[index] for index in plan.sampled])
        # Where each row's new token goes, which is also how much history its
        # repetition penalty may look at.
        columns = self._to_device(plan.gen_counts)

        generated = None
        width = max(plan.gen_counts)
        if sampling.any_penalty and width:
            # The grid is far wider than any live sequence, so slice it down to
            # the longest history in the batch and mask the rest off per row.
            span = self._columns[:width].unsqueeze(0)
            generated = GeneratedSpan(
                self._gen_grid[slots.unsqueeze(1), span], span < columns.unsqueeze(1)
            )

        tokens = _sync_tp(self._sampler.sample_batched(logits, sampling, generated).reshape(-1))
        self._gen_grid[slots, columns] = tokens
        return tokens

    def _batched_sampling(self, params: tuple[SamplingParams, ...]) -> BatchedSamplingParams:
        """Device-side sampling knobs, rebuilt only when the sampled rows change.

        Four small uploads and a handful of comprehensions per build, which a
        steady decode batch — the same requests, step after step — should not pay
        every step. The key snapshots the four scalars that affect those
        tensors, so an in-place change to the user-facing parameters cannot
        leave stale values on the device.
        """
        # SamplingParams is intentionally a user-facing mutable dataclass. Do
        # not cache by object equality: the old key held the same objects, so an
        # in-place mutation compared equal to itself and left stale GPU knobs.
        key = tuple(
            (p.temperature, p.top_p, p.repetition_penalty, p.is_greedy) for p in params
        )
        if key != self._sampling_key:
            self._sampling = BatchedSamplingParams.build(params, self._device)
            self._sampling_key = key
        return self._sampling  # type: ignore[return-value]

    # --------------------------------------------------------------- internals #
    def _rows(self, tokens: tuple[int, ...], padded: int) -> torch.Tensor:
        """Shape one-token rows into the ``[padded, 1]`` batch the model will see.

        Filler rows exist only to reach a captured graph batch size; whatever id
        they carry is thrown away with their logits, so they take the pad token.
        The whole batch is uploaded in one copy — a few hundred bytes, against a
        step that moves the model's weights.
        """
        padding = (self._pad_id,) * (padded - len(tokens))
        return torch.tensor(tokens + padding, dtype=torch.long, device=self._device).view(padded, 1)

    def _pick(self, logits: torch.Tensor, rows: tuple[int, ...], total: int) -> torch.Tensor | None:
        """Narrow logits to the rows that will be sampled.

        Returns ``None`` when no row is sampled, which is the signal that the pass
        ran for its K/V alone.
        """
        if not rows:
            return None
        if len(rows) == total:
            return logits
        return logits[self._to_device(rows)]

    def _to_device(self, values: Sequence[int]) -> torch.Tensor:
        """Upload a host index list as a fresh int64 tensor.

        Deliberately a new allocation rather than a write into a reused staging
        buffer: the previous step's tensor may still be queued for the GPU, and
        overwriting it from the host would race with kernels that have not run yet.
        """
        return torch.tensor(values, dtype=torch.long, device=self._device)
