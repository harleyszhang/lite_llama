"""Model worker: runs one :class:`ModelInput` — forward *and* sampling — per rank.

:class:`ModelWorker` turns a plan into launch-ready rows (prefill grid,
extend or padded decode), runs the forward pass, samples the new tokens,
and applies the TP collectives so every rank agrees on the result.

Usage:
    worker = ModelWorker(engine, max_num_seqs, max_seq_len)
    tokens, logprobs = worker.execute(model_input)
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch

from ..distributed.parallel_state import tensor_model_parallel_broadcast
from ..engine.sampler import (
    BatchedSamplingParams,
    GeneratedSpan,
    PositionLogprobs,
    SamplingParams,
    rows_logprobs,
)
from .overlap import OverlapPolicy, StreamPool, Timeline

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..engine.llm_engine import LLMEngine

#: Environment variable enabling the launch/harvest engine pipeline (O2): the
#: engine plans and launches step N+1 while step N is still on the GPU, and
#: harvests N's tokens one step late. Off by default — it deliberately delays
#: stop handling by one token, so it is a deployment choice, not a default.
PIPELINE_ENV = "LITE_LLAMA_PIPELINE"


def pipeline_enabled() -> bool:
    """Read ``LITE_LLAMA_PIPELINE``; only ``1``/``true``/``on`` means on."""
    raw = os.environ.get(PIPELINE_ENV, "0").strip().lower()
    return raw in ("1", "true", "on")


class PassKind(StrEnum):
    """Which kernel path a plan takes through attention.

    Attributes:
        PREFILL: A padded ``[sequences, width]`` grid through a prefill kernel.
            First chunks (nothing cached yet) run pure self-attention over the
            grid; long resumed chunks — ``seq_starts > 0``, on a cache the
            chunked kernel can read verbatim — run the same grid with their keys
            and values coming from the slot's own cache rows instead.
        EXTEND: One decode-style row per token, each query attending over its
            slot's whole cached history. The route for resumed chunks too wide
            to replay a decode graph — short remainders, fp8 caches the chunked
            kernel cannot serve, and the ``LITE_LLAMA_FUSED_CHUNK_PREFILL=0``
            kill-switch's wholesale restore.
        DECODE: One row per sequence, one token each.
    """

    PREFILL = "prefill"
    EXTEND = "extend"
    DECODE = "decode"


@dataclass(frozen=True)
class ModelInput:
    """ModelInput: description of a single model forward pass (does not include data layout).

    All fields are tuples (or immutable :class:`SamplingParams`), making them comparable,
    persistent, picklable, and thus broadcastable to tensor-parallel workers.

    Sequences are described by (slots[i], seq_starts[i], seq_lens[i]):
        - slots[i]       : KV cache slot for sequence i
        - seq_starts[i]  : start row index within the cache for this sequence
        - seq_lens[i]    : end position (current chunk length) for this sequence
        This pass processes the cache span [seq_starts[i], seq_lens[i]) for each sequence.
        `tokens` is the concatenation of all token chunks from these spans (no padding).
        Padding, if needed, is added by :class:`ModelWorker` based on kernel requirements.

    Sampling fields apply only to a subset of sequences (those that have finished prefilling):
        - sampled          : indices into `slots` (in ascending order)
        - sampling / gen_counts : parallel to `sampled`, one per sequence to sample.

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
        block_writes: ``(slot, group_id, start_block, block_ids)`` block-table
            entries to install before the pass runs, so its rows address the
            physical pages the scheduler allocated them — a reused prefix points
            at the blocks that already hold its K/V, and no K/V is moved. Every
            tensor-parallel rank applies them to its own table, which is why they
            travel with the plan rather than being applied by the driver.
        prompt_logprobs: Per-sequence top-k width for prompt scoring, parallel
            to ``slots``; ``None`` for a sequence that did not ask. Empty (the
            default) when no sequence in the pass asked. Parallel to ``slots``
            rather than ``sampled`` because a partial chunk owes prompt records
            without having a row to sample.
        prompt_targets: The token id each input row is scored against, parallel
            to ``tokens``: row ``j`` of a chunk predicts position
            ``start + j + 1``, whose true token is known to the scheduler but
            is not necessarily part of this plan (a partial chunk's last row
            predicts the *next* chunk's first token). The sampled row's entry
            is never read; the engine fills it with ``0``.
    """

    kind: PassKind
    slots: tuple[int, ...]
    seq_starts: tuple[int, ...]
    seq_lens: tuple[int, ...]
    tokens: tuple[int, ...]
    sampling: tuple[SamplingParams, ...]
    sampled: tuple[int, ...]
    gen_counts: tuple[int, ...]
    block_writes: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    prompt_logprobs: tuple[int | None, ...] = ()
    prompt_targets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("a ModelInput must cover at least one sequence")
        if not len(self.slots) == len(self.seq_starts) == len(self.seq_lens):
            raise ValueError("slots, seq_starts and seq_lens must describe the same sequences")
        if len(self.tokens) != sum(self.chunk_lens):
            raise ValueError(f"got {len(self.tokens)} tokens for {sum(self.chunk_lens)} cache rows")
        if not len(self.sampled) == len(self.sampling) == len(self.gen_counts):
            raise ValueError("sampled, sampling and gen_counts must describe the same rows")
        if self.prompt_logprobs and len(self.prompt_logprobs) != len(self.slots):
            raise ValueError("prompt_logprobs must be empty or parallel to slots")
        if self.prompt_targets and len(self.prompt_targets) != len(self.tokens):
            raise ValueError("prompt_targets must be empty or parallel to tokens")
        if any(k is not None for k in self.prompt_logprobs) and not self.prompt_targets:
            raise ValueError("prompt_logprobs needs prompt_targets to score against")

    @property
    def chunk_lens(self) -> tuple[int, ...]:
        """Tokens this pass feeds per sequence."""
        return tuple(end - start for start, end in zip(self.seq_starts, self.seq_lens, strict=True))


@dataclass(frozen=True)
class PassLogprobs:
    """Logprob records one pass produced, beside the tokens it drew.

    Attributes:
        sampled: One entry per row of ``ModelInput.sampled`` — the record of
            the distribution that row's token was drawn from, ``None`` where
            the request did not ask.
        prompt: One entry per sequence of ``ModelInput.slots``: the chunk's
            positions scored against the prompt's own tokens, or ``None`` where
            the request did not ask. Entry ``j`` of sequence ``i`` describes
            position ``seq_starts[i] + j + 1`` — the row that consumed token
            ``seq_starts[i] + j`` produced the distribution that position's
            own token was scored on. A final chunk's last row is the sampled
            row, so it appears in ``sampled`` and not here.
    """

    sampled: tuple[PositionLogprobs | None, ...] = ()
    prompt: tuple[tuple[PositionLogprobs, ...] | None, ...] = ()


@dataclass
class _PreparedPass:
    """One pass's device-bound inputs, built on the host and uploaded.

    With the overlap policy on, the upload rides the pool's copy stream and
    ``event`` marks its completion: the consumer stream waits the event instead
    of the host waiting the copy. With the policy off the upload was the legacy
    blocking one and ``event`` is ``None``, which the pool treats as a no-op —
    the forward paths below read the same either way. Either way the tensors the
    model is fed are identical, which is the equivalence the tests pin.

    Attributes:
        input_ids: The pass's token grid — ``[n, width]`` for a prefill,
            ``[padded, 1]`` for extend/decode, filler rows included.
        positions: Absolute positions, prefill only; extend and decode derive
            theirs from the slot metadata instead (no upload at all).
        logits_positions: Per-row gather index of the next-token logits,
            prefill only.
        event: Completion of this pass's copies on the copy stream; the copies
            are stream-ordered, so the last event covers them all.
        padded: Rows actually submitted, extend/decode only — the caller
            discards the trailing filler rows' logits.
    """

    input_ids: torch.Tensor
    event: torch.cuda.Event | None
    positions: torch.Tensor | None = None
    logits_positions: torch.Tensor | None = None
    padded: int = 0


class ModelWorker:
    """Executes plans against this rank's model shard and KV cache.

    Holds the only mutable execution state: the fixed-slot KV view and the
    ``[num_slots, max_seq_len]`` grid of generated tokens that the repetition
    penalty reads. Both are indexed by *slot*, never by position in a batch, so
    nothing here has to be invalidated when the running set changes — a plan
    always names the slots it means.

    With the launch/harvest pipeline on, the worker also keeps the *next*
    decode input per slot on the device (:attr:`_next_tokens`): every pass's
    sampler writes the tokens it drew straight into that grid, and the next
    decode pass gathers its inputs out of it — the token the engine would
    otherwise have to read back, detokenise-adjacent bookkeeping aside, never
    crosses to the host and back.

    Args:
        engine: A built :class:`~lite_llama.engine.llm_engine.LLMEngine`; the
            worker takes its KV cache over via the slot view.
        max_num_seqs: Concurrency ceiling, which caps the slots handed out and so
            the height of the generated-token grid.
        max_seq_len: Context bound, and the grid's width.
        pipeline: Whether decode inputs come from the device-side
            next-token grid (the O2 launch/harvest engine). ``None`` reads
            :data:`PIPELINE_ENV`, which is also how a tensor-parallel
            follower learns the driver's choice.
    """

    def __init__(
        self,
        engine: LLMEngine,
        max_num_seqs: int,
        max_seq_len: int,
        *,
        pipeline: bool | None = None,
    ) -> None:
        self._runner = engine.model_runner
        self._sampler = engine.sampler
        self._device = engine.device
        self._pad_id = engine.pad_id
        self._slot_batch = self._runner.enable_slot_kv_cache()
        self._pipeline = pipeline_enabled() if pipeline is None else pipeline

        # Slot ids stay below max_num_seqs, which keeps the generated-token grid
        # proportional to the concurrency the caller asked for rather than to
        # however many slots happen to fit in the cache.
        self.num_slots = min(self._slot_batch.num_slots, max_num_seqs)
        # Cache rows, expressed in the scheduler's block size: the block pool is
        # what admits requests now, so it has to be sized by the memory that
        # actually exists rather than by the table's geometry.
        self.num_kv_blocks = (
            self._runner.kv_cache_manager.gpu_num_blocks // self._slot_batch.block_size
        )
        self._gen_grid = torch.zeros(
            (self.num_slots, max_seq_len), dtype=torch.long, device=self._device
        )
        self._columns = torch.arange(max_seq_len, device=self._device)
        self._no_tokens = torch.empty(0, dtype=torch.long, device=self._device)
        # The O2 feedback lane: whatever each slot's most recent pass sampled,
        # kept on the device so the next decode pass feeds it back without a
        # host round-trip. A prefill's first token lands here too — the engine
        # has not harvested it yet when it plans the first decode step.
        self._next_tokens = torch.full(
            (self.num_slots,), self._pad_id, dtype=torch.long, device=self._device
        )

        # Sampling knobs cost four small uploads to rebuild, so a steady batch
        # reuses them; the key is the plan's own rows, which needs no cooperation
        # from whoever mutates the running set.
        self._sampling_key: tuple[tuple[float, float, float, bool, int | None], ...] | None = None
        self._sampling: BatchedSamplingParams | None = None

        # L1 cross-stream overlap: input uploads ride a copy stream so the host
        # never stalls on a pageable H2D between kernel launches, and — the engine
        # deferring its readback to the end of the step — a later pass's copies
        # overlap an earlier pass's forward on the device. On a non-CUDA device
        # the pool is built with the policy off, which collapses every site back
        # to the inline blocking upload.
        on_cuda = torch.device(self._device).type == "cuda"
        self._policy = OverlapPolicy.from_env() if on_cuda else OverlapPolicy(enabled=False)
        self.timeline = Timeline.from_env(str(self._device)) if on_cuda else Timeline(enabled=False)
        self._pool = StreamPool(str(self._device), self._policy, self.timeline)

    @torch.inference_mode()
    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        """Run one pass and return its sampled tokens plus any logprob records.

        Returns:
            ``(tokens, records)``: ``[len(model_input.sampled)]`` token ids on
            this rank's device, identical across tensor-parallel ranks, and the
            pass's :class:`PassLogprobs` — ``None`` when no request asked for
            logprobs, which costs nothing extra. A pass whose sequences all
            still owe tokens (a prompt chunk that did not finish) runs the
            model — its K/V has to land — and returns an empty tensor; its
            records, when asked for, are entirely prompt records.
        """
        prepared = self.prepare(model_input)
        # Before the forward, so every row this pass reads or writes already has
        # a page behind it. After the prepare, because prepare only gathers table
        # entries for rows the plan names, and those are exactly the ones these
        # writes install -- the gather happens on the device, after this.
        self._slot_batch.write_block_tables(model_input.block_writes)
        logits, prompt = self._forward(model_input, prepared)
        sampled_records = None
        if logits is None:
            tokens = self._no_tokens
        else:
            tokens, sampled_records = self._sample(model_input, logits)
        if not any(prompt) and sampled_records is None:
            return tokens, None
        return tokens, PassLogprobs(
            sampled=tuple(sampled_records) if sampled_records is not None else (),
            prompt=prompt,
        )

    # -------------------------------------------------------------- preparing #
    def prepare(self, plan: ModelInput) -> _PreparedPass:
        """Build one pass's inputs on the host and start their upload.

        Everything the model is fed follows from ``(slots, seq_starts,
        seq_lens)``, so the layout arithmetic runs on the host — row expansion
        and graph padding through the slot view's plan helpers — and the upload
        leaves on the pool's copy stream. The returned pass carries the event
        the forward must consume before launching kernels; with the policy off
        the upload already happened inline and the event is ``None``.
        """
        match plan.kind:
            case PassKind.PREFILL:
                return self._prepare_grid(plan)
            case PassKind.EXTEND:
                return self._prepare_rows(plan, PassKind.EXTEND)
            case PassKind.DECODE:
                return self._prepare_rows(plan, PassKind.DECODE)

    def _prepare_grid(self, plan: ModelInput) -> _PreparedPass:
        """Pad the prompt chunks into a grid and upload it with its positions."""
        chunk_lens = plan.chunk_lens
        width = max(chunk_lens)
        grid, offset = [], 0
        for chunk in chunk_lens:
            grid.append(plan.tokens[offset : offset + chunk] + (self._pad_id,) * (width - chunk))
            offset += chunk
        # Padded columns run past a row's real position, but attention never
        # reads past that row's b_seq_len, so the junk positions are inert.
        positions = [list(range(start, start + width)) for start in plan.seq_starts]

        input_ids, _ = self._pool.upload_async(
            grid, dtype=torch.long, label="upload.prefill.tokens"
        )
        positions_t, _ = self._pool.upload_async(
            positions, dtype=torch.long, label="upload.prefill.positions"
        )
        # A row's next-token logits sit at its own last real *column*.
        logits_pos, event = self._pool.upload_async(
            [chunk - 1 for chunk in chunk_lens], dtype=torch.long, label="upload.prefill.logits"
        )
        return _PreparedPass(
            input_ids=input_ids,
            event=event,
            positions=positions_t,
            logits_positions=logits_pos,
        )

    def _prepare_rows(self, plan: ModelInput, kind: PassKind) -> _PreparedPass:
        """Upload one-token rows padded onto the captured graph width.

        Extend and decode share the shape — one row per token, filler rows up to
        the graph width carrying the pad id — and differ only in whose helper
        plans the rows: extend flattens chunks, decode pads the running set.
        Filler rows exist only to reach a captured graph batch size; whatever id
        they carry is thrown away with their logits.
        """
        if kind is PassKind.EXTEND:
            rows_slot, _ = self._slot_batch.plan_extend_rows(
                plan.slots, plan.seq_starts, plan.seq_lens
            )
            padded = len(rows_slot)
        else:
            padded_slots, _ = self._slot_batch.pad_decode_rows(plan.slots, plan.seq_lens)
            padded = len(padded_slots)

        if kind is PassKind.DECODE and self._pipeline:
            # The launch/harvest pipeline: the plan's token entries are
            # placeholders (``-1``), because the engine has not harvested the
            # tokens it would name. Gather the real inputs off the device grid
            # the last pass's sampler wrote, pad with inert ids up to the graph
            # width, and skip the upload entirely — there is nothing to upload.
            real = len(plan.slots)
            gathered = self._next_tokens[self._to_device(plan.slots)]
            pad = padded - real
            if pad > 0:
                gathered = torch.cat(
                    [
                        gathered,
                        torch.full((pad,), self._pad_id, dtype=torch.long, device=self._device),
                    ]
                )
            return _PreparedPass(
                input_ids=gathered.view(padded, 1), event=None, padded=padded
            )

        rows = plan.tokens + (self._pad_id,) * (padded - len(plan.tokens))
        input_ids, event = self._pool.upload_async(
            rows, dtype=torch.long, label=f"upload.{kind}.tokens"
        )
        return _PreparedPass(input_ids=input_ids.view(padded, 1), event=event, padded=padded)

    # ---------------------------------------------------------------- forwards #
    def _forward(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """The pass's sampled-row logits and its per-sequence prompt records.

        The second element is parallel to ``plan.slots`` and all-``None`` for a
        pass nobody asked prompt logprobs of — decode passes always, chunk
        passes by request.
        """
        match plan.kind:
            case PassKind.PREFILL:
                return self._forward_grid(plan, prepared)
            case PassKind.EXTEND:
                return self._forward_extend(plan, prepared)
            case PassKind.DECODE:
                return self._forward_decode(plan, prepared)

    def _forward_grid(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """A padded token grid through the prefill kernel."""
        self._slot_batch.begin_prefill(plan.slots, plan.seq_starts, plan.seq_lens)
        # The grid, its positions and the gather index all left on the copy
        # stream; one stream-ordered wait covers them, then the kernels launch.
        self._pool.consume(
            prepared.event, prepared.input_ids, prepared.positions, prepared.logits_positions
        )
        wants_prompt = any(k is not None for k in plan.prompt_logprobs)
        with self.timeline.region("forward.prefill", "compute"):
            # The model gathers one column per row, so the whole grid is gathered
            # here and the sampled subset selected after. A pass that scores
            # prompt positions needs every column's logits instead: the gather
            # is skipped and the lm_head projects the whole grid — the price of
            # prompt_logprobs, paid in GEMM width rather than a second forward.
            logits = self._runner.forward(
                prepared.input_ids,
                prepared.positions,
                None,
                logits_positions=None if wants_prompt else prepared.logits_positions,
            )
        if not wants_prompt:
            return self._pick(logits, plan.sampled, len(plan.slots)), (None,) * len(plan.slots)
        prompt = self._prompt_records(plan, logits, grid=True)
        rows = torch.arange(len(plan.slots), device=logits.device)
        gathered = logits[rows, prepared.logits_positions]
        return self._pick(gathered, plan.sampled, len(plan.slots)), prompt

    def _forward_extend(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """Chunks resuming on a cached prefix: one decode-style row per token."""
        padded = self._slot_batch.begin_extend(plan.slots, plan.seq_starts, plan.seq_lens)
        self._pool.consume(prepared.event, prepared.input_ids)
        # begin_extend set b_seq_len to each row's absolute position plus one, so
        # the position of the token it feeds is exactly that minus one.
        positions = (self._slot_batch.seq_lens - 1).view(-1, 1)

        with self.timeline.region("forward.extend", "compute"):
            # No logits_positions: the pass projects every row anyway (one row
            # per token), so prompt scoring costs nothing extra here — the
            # stretch of rows a sequence owns *is* its prompt's distributions.
            logits = self._runner.forward(prepared.input_ids, positions, None)
        # One row per token: a sequence's next-token logits are on the last row of
        # its own stretch of the flattened batch.
        ends = list(itertools.accumulate(plan.chunk_lens))
        rows = tuple(ends[index] - 1 for index in plan.sampled)
        flat = logits[:, -1, :]
        prompt = (None,) * len(plan.slots)
        if any(k is not None for k in plan.prompt_logprobs):
            prompt = self._prompt_records(plan, flat, grid=False)
        return self._pick(flat, rows, padded), prompt

    def _forward_decode(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """One token for every sequence in the plan."""
        rows = len(plan.slots)
        self._slot_batch.begin_decode(plan.slots, plan.seq_lens)
        self._pool.consume(prepared.event, prepared.input_ids)
        # The token being fed sits at its own cache row, i.e. length minus one.
        positions = self._slot_batch.seq_lens.view(-1, 1) - 1

        with self.timeline.region("forward.decode", "compute"):
            logits = self._runner.forward(prepared.input_ids, positions, None)
        # Decode never has prompt positions to score: they were all covered
        # during prefill, so the second element is uniformly empty.
        return self._pick(logits[:rows, -1, :], plan.sampled, rows), (None,) * len(plan.slots)

    # ---------------------------------------------------------------- sampling #
    def _sample(
        self, plan: ModelInput, logits: torch.Tensor
    ) -> tuple[torch.Tensor, list[PositionLogprobs | None] | None]:
        """Draw one token per sampled row and record it in the generated grid.

        The second return is the per-row logprob records, ``None`` when no
        sampled row asked — the common case, which costs nothing extra.
        """
        if not plan.sampled:
            return torch.empty(0, dtype=torch.long, device=logits.device), None

        sampling = self._batched_sampling(plan.sampling)
        slots = self._to_device([plan.slots[index] for index in plan.sampled])
        columns = self._to_device(plan.gen_counts)

        generated = None
        width = max(plan.gen_counts)
        if sampling.any_penalty and width:
            span = self._columns[:width].unsqueeze(0)
            generated = GeneratedSpan(
                self._gen_grid[slots.unsqueeze(1), span], span < columns.unsqueeze(1)
            )

        ids, records = self._sampler.sample_batched_with_logprobs(logits, sampling, generated)
        # Every TP rank must hold the same ids: non-greedy sampling draws from a
        # per-rank RNG, so without the broadcast the ranks would disagree about
        # the token they just produced and every later step would compound the
        # divergence. A world of one is the collective's own early return.
        tokens = tensor_model_parallel_broadcast(ids.reshape(-1))
        self._gen_grid[slots, columns] = tokens
        if self._pipeline:
            # Device-side feedback: the next pass that feeds this slot reads
            # its input here, never through the host.
            self._next_tokens[slots] = tokens
        return tokens, records

    def readback(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Stage a pass's sampled tokens for the host, without waiting.

        The D2H copy rides the pool's copy stream behind the pass that produced
        the tokens; the caller synchronises on the returned event one engine
        step later, by which time the copy has landed under other work. On a
        pool with the policy off (or a CPU device) this is a plain ``.cpu()``
        and the event is ``None`` — same contract, no overlap.
        """
        return self._pool.readback_async(tokens, label="readback.tokens")

    def _batched_sampling(self, params: tuple[SamplingParams, ...]) -> BatchedSamplingParams:
        """Device-side sampling knobs, rebuilt only when the sampled rows change.

        Four small uploads and a handful of comprehensions per build, which a
        steady decode batch — the same requests, step after step — should not pay
        every step. The key snapshots the five values that feed those tensors,
        so an in-place change to the user-facing parameters cannot leave stale
        values on the device.
        """
        # SamplingParams is intentionally a user-facing mutable dataclass. Do
        # not cache by object equality: the old key held the same objects, so an
        # in-place mutation compared equal to itself and left stale GPU knobs.
        key = tuple(
            (p.temperature, p.top_p, p.repetition_penalty, p.is_greedy, p.logprobs) for p in params
        )
        if key != self._sampling_key:
            self._sampling = BatchedSamplingParams.build(params, self._device)
            self._sampling_key = key
        return self._sampling  # type: ignore[return-value]

    # --------------------------------------------------------------- internals #
    def _prompt_records(
        self, plan: ModelInput, logits: torch.Tensor, *, grid: bool
    ) -> tuple[tuple[PositionLogprobs, ...] | None, ...]:
        """Score each asking sequence's chunk against the prompt's own tokens.

        ``grid`` selects the logits layout: a prefill grid is ``[n, width, V]``,
        where sequence ``i``'s rows are row ``i``'s leading columns; an extend
        pass is the flattened ``[total_tokens, V]``, where they are the
        sequence's stretch. Either way row ``j`` of sequence ``i`` predicted
        position ``seq_starts[i] + j + 1``, so the score target is
        ``prompt_targets`` at the same token offset. A final chunk's last row
        is excluded — it is the sampled row, whose record the sampler produces.
        """
        records: list[tuple[PositionLogprobs, ...] | None] = []
        sampled = set(plan.sampled)
        offset = 0
        for index, k in enumerate(plan.prompt_logprobs):
            chunk = plan.chunk_lens[index]
            if k is None:
                records.append(None)
            else:
                rows = chunk - 1 if index in sampled else chunk
                if rows <= 0:
                    # A one-token final chunk covers no prompt position: its
                    # only row is the one being sampled.
                    records.append(())
                else:
                    rows_t = logits[index, :rows] if grid else logits[offset : offset + rows]
                    targets = self._to_device(plan.prompt_targets[offset : offset + rows])
                    records.append(tuple(rows_logprobs(rows_t, targets, k)))
            offset += chunk
        return tuple(records)

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
