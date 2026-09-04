"""Token sampling: turn a step's logits into the next token id per sequence.

:class:`Sampler` applies temperature, top-p nucleus sampling and repetition
penalty to a batch of logits; greedy decoding short-circuits to argmax. The
pure helpers below stay unit-testable on CPU without a model.

Usage:
    probs = torch.softmax(logits / temperature, -1)
    token = sample_top_p(probs, top_p, k)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_max,
    tensor_model_parallel_broadcast,
)


def local_vocab_offset(local_width: int) -> int | None:
    """First global token id of this rank's logits slice, or ``None`` when TP is off.

    The shard is derived from the ambient parallel state and the width of the tensor
    itself rather than threaded down from the model, for the same reason
    :func:`~rapid_llm.distributed.parallel_state.tensor_model_parallel_all_reduce` is: there is exactly one
    vocabulary split per process, and a sampler that has to be *told* about it is a
    sampler that can be told wrong. It holds because every logits producer in the
    package is a :class:`~rapid_llm.modules.vocab_parallel.ParallelLMHead`.
    """
    if get_tensor_model_parallel_world_size() == 1:
        return None
    return get_tensor_model_parallel_rank() * local_width


@dataclass
class SamplingParams:
    """Decoding hyper-parameters for one generation request.

    Attributes:
        temperature: Softmax temperature; ``0`` means greedy (argmax).
        top_p: Nucleus sampling threshold in ``(0, 1]``.
        max_gen_len: Maximum number of tokens to generate; ``None`` lets the engine
            fill the remaining context window.
        repetition_penalty: Penalty on logits of already-generated tokens
            (``1.0`` disables it). Values > 1 discourage repetition. Enabled by
            default because small base models fall into repetition loops under
            both greedy and low-temperature sampling (verified: HuggingFace's
            own implementation loops on the same prompts).
        stop_on_repeat: Circuit breaker in the engine — stop a sequence whose
            generated text degenerates into a repeating template.
        logprobs: Report the ``k`` most likely tokens' log-probabilities per
            generated token (the sampled token's own logprob always comes
            along; ``0`` reports it alone). ``None`` — the default — skips the
            work entirely: the top-k over the vocabulary is the only part of
            sampling whose cost scales with the vocabulary.
        prompt_logprobs: Same reporting for the prompt positions, computed
            during prefill from the same forward pass. Position 0 has no
            predictor and is reported as ``None``, and so is any position
            served from the prefix cache, whose forward never ran.
    """

    temperature: float = 0.6
    top_p: float = 0.9
    max_gen_len: int | None = None
    repetition_penalty: float = 1.1
    stop_on_repeat: bool = True
    logprobs: int | None = None
    prompt_logprobs: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_gen_len is not None and self.max_gen_len < 1:
            raise ValueError(f"max_gen_len must be >= 1 or None, got {self.max_gen_len}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        for field, value in (
            ("logprobs", self.logprobs),
            ("prompt_logprobs", self.prompt_logprobs),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{field} must be >= 0, got {value}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass(frozen=True)
class GeneratedSpan:
    """The tokens each sequence has generated so far, as one padded tensor.

    Attributes:
        token_ids: ``[batch, span]`` ids; entries where ``mask`` is ``False``
            are padding and must not influence sampling.
        mask: ``[batch, span]`` ``True`` at real generated positions.
    """

    token_ids: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class PositionLogprobs:
    """One position's logprob record: the token there, and the best alternatives.

    Attributes:
        token_id: The token actually at this position (sampled, or the prompt's
            own token for prompt logprobs).
        logprob: Its log-probability under the distribution this position
            produced. Not guaranteed to appear in the top-k lists: a sampled
            token is not necessarily one of the ``k`` most likely.
        top_token_ids: The ``k`` most likely token ids, descending.
        top_logprobs: Their log-probabilities, parallel to ``top_token_ids``.
    """

    token_id: int
    logprob: float
    top_token_ids: tuple[int, ...]
    top_logprobs: tuple[float, ...]


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: GeneratedSpan,
    penalty: float | torch.Tensor,
    vocab_offset: int = 0,
) -> torch.Tensor:
    """HuggingFace-style repetition penalty over the generated span.

    For every token that already occurred, its logit is divided by ``penalty``
    when positive (de-moted) and multiplied when negative, exactly as
    ``transformers``' ``RepetitionPenaltyLogitsProcessor``. Only the *generated*
    tokens are penalised — the prompt is user-supplied context.

    The whole batch is handled by one scatter plus two selects. The obvious
    alternative — looping over rows and calling ``torch.unique`` — launches a
    handful of kernels per sequence per decode step, which at batch 8 costs more
    than the penalty itself. Marking hits in a boolean table also makes the
    penalty idempotent for repeated tokens, matching ``unique`` semantics.

    Padded positions are redirected to a scratch column instead of being
    scattered with ``False``: two positions can carry the *same* token id, one
    padded and one real, and the ``False`` write would then clear the real hit
    and silently drop that token's penalty.

    Args:
        logits: ``[batch, vocab]``, or this rank's ``[batch, vocab / tp]`` slice.
        generated: Padded generated-token view for the batch.
        penalty: Penalty factor; ``1.0`` is a no-op. A ``[batch, 1]`` tensor
            applies a different factor per row, which is what an online batch of
            independently configured requests needs.
        vocab_offset: First global token id covered by ``logits``. Non-zero under
            vocabulary parallelism, where a token generated by another rank's slice is
            simply not this rank's to penalise — it falls out of range and is redirected
            to the same scratch column as padding.

    Returns:
        New logits of the same shape (input is left untouched).
    """
    batch, width = logits.shape
    seen = torch.zeros(batch, width + 1, dtype=torch.bool, device=logits.device)
    local = generated.token_ids - vocab_offset
    mine = generated.mask & (local >= 0) & (local < width)
    columns = torch.where(mine, local, width)
    seen.scatter_(1, columns, True)
    penalised = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen[:, :width], penalised, logits)


# Nucleus sampling never touches most of the vocabulary: real model
# distributions put far more than ``top_p`` mass on the top few dozen tokens,
# so a 1024-wide pool makes the draw identical to a full-vocabulary sort in
# every non-degenerate case while skipping the sort over the rest.
_TOP_P_CANDIDATES = 1024


def sample_top_p(
    probs: torch.Tensor, top_p: float | torch.Tensor, k: int | None = _TOP_P_CANDIDATES
) -> torch.Tensor:
    """Nucleus (top-p) sampling.

    Keeps the smallest set of highest-probability tokens whose cumulative mass
    just exceeds ``top_p``, renormalises, and samples one token from it.

    The candidates come from ``torch.topk`` over a ``k``-wide pool, which
    returns probabilities already in descending order — no sort over the whole
    vocabulary. Whenever the pool's total mass exceeds ``top_p`` the nucleus is
    provably contained in it and the draw matches a full sort exactly; a
    flatter distribution than that (extreme temperature) keeps the entire
    pool, capping the nucleus at ``k`` tokens.

    Args:
        probs: ``[batch, vocab]`` probability distribution.
        top_p: Cumulative probability threshold; a ``[batch, 1]`` tensor gives
            each row its own threshold.
        k: Candidate pool size, clamped to the vocabulary. ``None`` uses the
            full vocabulary for exact sampling of flat distributions.

    Returns:
        ``[batch, 1]`` sampled token ids.
    """
    # top_p=1 is ordinary categorical sampling. Avoiding topk here is both exact
    # (the fixed candidate pool otherwise drops the long tail) and considerably
    # cheaper for the common OpenAI-compatible default.
    if not isinstance(top_p, torch.Tensor) and top_p == 1.0:
        return torch.multinomial(probs, num_samples=1)
    k = probs.shape[-1] if k is None else min(k, probs.shape[-1])
    top_probs, top_idx = torch.topk(probs, k, dim=-1)  # already descending
    return _draw_from_nucleus(top_probs, top_idx, top_p)


def _draw_from_nucleus(
    top_probs: torch.Tensor, top_idx: torch.Tensor, top_p: float | torch.Tensor
) -> torch.Tensor:
    """Draw one token from an already descending ``(probability, id)`` candidate pool.

    Split out because vocabulary-parallel sampling builds its pool differently — by
    gathering each rank's local top-k — but must make the *same* nucleus decision from
    it, and a second copy of this arithmetic is a second place to get it wrong.
    Both tensors are consumed in place.
    """
    cumulative = torch.cumsum(top_probs, dim=-1)
    # Drop tokens once the mass *before* them already exceeds top_p.
    drop_mask = cumulative - top_probs > top_p
    # A row whose pool never accumulates top_p has no droppable tail inside
    # it; keeping every candidate is the closest the pool can get to its
    # nucleus.
    drop_mask &= cumulative[:, -1:].gt(top_p)
    top_probs.masked_fill_(drop_mask, 0.0)
    top_probs.div_(top_probs.sum(dim=-1, keepdim=True))
    choice = torch.multinomial(top_probs, num_samples=1)
    return torch.gather(top_idx, -1, choice)


def global_argmax(local_logits: torch.Tensor, vocab_offset: int) -> torch.Tensor:
    """Greedy pick across a vocabulary split over the TP ranks.

    Each rank offers its own best ``(logit, global id)``; the winner is the lowest id
    among the ranks holding the maximum, which keeps the choice deterministic when two
    slices tie — ``argmax`` over a gathered pool would leave that to kernel ordering.
    Two values per row cross the wire.

    Returns:
        ``[batch, 1]`` global token ids.
    """
    local_best, local_id = local_logits.max(dim=-1, keepdim=True)
    best = tensor_model_parallel_all_reduce_max(local_best.clone())
    ids = local_id + vocab_offset
    losers = torch.full_like(ids, torch.iinfo(ids.dtype).max)
    candidates = torch.where(local_best == best, ids, losers)
    return tensor_model_parallel_all_gather(candidates).amin(dim=-1, keepdim=True)


def greedy_ids(logits: torch.Tensor, offset: int | None) -> torch.Tensor:
    """Argmax per row, over the whole vocabulary or across its TP split.

    Split out so :meth:`Sampler._draw` calls it only where a row will actually
    use the result. Computing it unconditionally costs a full-vocabulary argmax
    per step, and under TP two collectives, for a batch that turns out to be
    wholly stochastic and throws it away.

    Args:
        logits: ``[batch, vocab]``, or this rank's slice.
        offset: This rank's first global token id, ``None`` when TP is off.

    Returns:
        ``[batch, 1]`` next-token ids.
    """
    if offset is None:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return global_argmax(logits, offset)


def vocab_logsumexp(scaled: torch.Tensor) -> torch.Tensor:
    """``logsumexp`` over a vocabulary split across TP ranks: two scalars per row.

    The identity that makes vocabulary-parallel sampling cheap is
    ``log_softmax(x)_i = x_i - logsumexp(x)``: the normaliser is one number per row, so a
    rank holding only a slice of ``x`` needs nothing but that number to turn its slice
    into exact global log-probabilities. Assembling it takes a MAX reduce (the shift that
    keeps ``exp`` from overflowing must be the maximum over *all* slices, not this one)
    and a SUM reduce. vLLM instead all-gathers the logits, moving ``batch x vocab``
    values where this moves ``2 * batch``.

    Args:
        scaled: ``[batch, vocab / tp]`` logits, already divided by the temperature.

    Returns:
        ``[batch, 1]`` global ``logsumexp``, identical on every rank.
    """
    row_max = tensor_model_parallel_all_reduce_max(scaled.amax(dim=-1, keepdim=True))
    row_sum = tensor_model_parallel_all_reduce((scaled - row_max).exp().sum(dim=-1, keepdim=True))
    return row_max + row_sum.log()


def sharded_top_p(
    local_logits: torch.Tensor,
    temperature: float | torch.Tensor,
    top_p: float | torch.Tensor,
    vocab_offset: int,
    k: int | None = _TOP_P_CANDIDATES,
) -> torch.Tensor:
    """Nucleus sampling over a vocabulary split across TP ranks.

    Two collectives, both on tensors whose size is set by the batch and ``k`` rather
    than by the vocabulary:

    1. a **decentralised log_softmax** (:func:`vocab_logsumexp`) — the normaliser arrives
       as two scalars per row, so subtracting it from the local logits yields exactly the
       global probabilities without gathering any logits;
    2. a **candidate gather** — each rank's top ``k`` ``(logit, id)`` pairs. The union
       contains the global top ``k``, because a token in the global top ``k`` has fewer
       than ``k`` tokens above it anywhere, let alone on its own rank.

    Returns:
        ``[batch, 1]`` global token ids.
    """
    scaled = local_logits.float() / temperature
    log_z = vocab_logsumexp(scaled)

    local_k = scaled.shape[-1] if k is None else min(k, scaled.shape[-1])
    local_top, local_ids = torch.topk(scaled, local_k, dim=-1)
    pool = tensor_model_parallel_all_gather(local_top)
    pool_ids = tensor_model_parallel_all_gather(local_ids + vocab_offset)
    global_k = pool.shape[-1] if k is None else min(k, pool.shape[-1])
    top_probs, order = torch.topk((pool - log_z).exp(), global_k, dim=-1)
    return _draw_from_nucleus(top_probs, pool_ids.gather(-1, order), top_p)


def _distribution_records(
    scaled: torch.Tensor, ids: torch.Tensor, offset: int | None, k: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Chosen-token logprobs and the top-k of the distribution, TP-aware.

    Args:
        scaled: ``[rows, vocab]`` (or this rank's slice) logits, already in
            their final form — penalty applied, temperature divided through.
        ids: ``[rows, 1]`` global ids whose logprobs are wanted alongside the
            top-k (the tokens actually sitting at these positions).
        offset: This rank's first global token id, ``None`` when TP is off.
        k: Top-k width; ``0`` skips the top-k entirely.

    Returns:
        ``(chosen_logprobs [rows, 1], top_values [rows, k] | None,
        top_ids [rows, k] | None)`` — identical on every rank, because the
        normaliser is the two-scalar ``vocab_logsumexp`` reduction and the
        chosen-token gather is a masked sum: the rank owning the id contributes
        its logit, every other rank contributes zero.
    """
    if offset is None:
        log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
        logprobs = scaled - log_z
        chosen = logprobs.gather(-1, ids)
        if k <= 0:
            return chosen, None, None
        top_values, top_ids = logprobs.topk(min(k, scaled.shape[-1]), dim=-1)
        return chosen, top_values, top_ids

    log_z = vocab_logsumexp(scaled)
    width = scaled.shape[-1]
    local = ids - offset
    valid = (local >= 0) & (local < width)
    gathered = scaled.gather(-1, local.clamp(0, width - 1))
    chosen = (
        tensor_model_parallel_all_reduce(torch.where(valid, gathered, torch.zeros_like(gathered)))
        - log_z
    )
    if k <= 0:
        return chosen, None, None
    local_values, local_ids = (scaled - log_z).topk(min(k, width), dim=-1)
    pool_values = tensor_model_parallel_all_gather(local_values)
    pool_ids = tensor_model_parallel_all_gather(local_ids + offset)
    top_values, order = pool_values.topk(min(k, pool_values.shape[-1]), dim=-1)
    return chosen, top_values, pool_ids.gather(-1, order)


def _to_records(
    ids: torch.Tensor,
    ks: Sequence[int | None],
    chosen: torch.Tensor,
    top_values: torch.Tensor | None,
    top_ids: torch.Tensor | None,
) -> list[PositionLogprobs | None] | None:
    """Turn the device tensors into per-row host records; ``None`` where unasked.

    One readback covers the whole batch — the decode step synchronises once for
    detokenisation anyway, so the records ride that same sync rather than
    adding one of their own.
    """
    if all(k is None for k in ks):
        return None
    ids_l = ids.reshape(-1).tolist()
    chosen_l = chosen.reshape(-1).tolist()
    top_ids_l = top_ids.tolist() if top_ids is not None else None
    top_values_l = top_values.tolist() if top_values is not None else None
    records: list[PositionLogprobs | None] = []
    for row, k in enumerate(ks):
        if k is None:
            records.append(None)
            continue
        width = 0 if top_ids_l is None else min(k, len(top_ids_l[row]))
        records.append(
            PositionLogprobs(
                token_id=ids_l[row],
                logprob=chosen_l[row],
                top_token_ids=tuple(top_ids_l[row][:width]) if top_ids_l else (),
                top_logprobs=tuple(top_values_l[row][:width]) if top_values_l else (),
            )
        )
    return records


def rows_logprobs(logits: torch.Tensor, target_ids: torch.Tensor, k: int) -> list[PositionLogprobs]:
    """One :class:`PositionLogprobs` per row: the target token's, plus the top-k.

    This is the prompt-logprobs arithmetic — a block of positions whose
    *target* tokens are already known (the prompt's own), so there is no draw,
    only the distribution. Computed on the raw logits: temperature and
    penalties are sampling-time notions and do not apply to scoring a prompt.

    Args:
        logits: ``[rows, vocab]`` — this rank's slice under TP.
        target_ids: ``[rows]`` global ids the rows' distributions are scored on.
        k: Top-k width.
    """
    offset = local_vocab_offset(logits.shape[-1])
    chosen, top_values, top_ids = _distribution_records(
        logits.float(), target_ids.view(-1, 1), offset, k
    )
    records = _to_records(
        target_ids.view(-1, 1), [k] * logits.shape[0], chosen, top_values, top_ids
    )
    return records  # type: ignore[return-value]


@dataclass(frozen=True)
class BatchedSamplingParams:
    """One row of sampling knobs per sequence, as device tensors.

    Online serving mixes requests that disagree about temperature, top-p and
    repetition penalty. Splitting the batch by configuration and sampling each
    group separately would multiply the kernel launches by the number of distinct
    configurations, so the knobs become ``[batch, 1]`` tensors instead and every
    row is sampled in the same pass.

    Build these with :meth:`build`, and rebuild only when the running set changes
    — the values are fixed for the lifetime of a request, so a stable batch can
    reuse the same tensors and keep the decode step free of host-device traffic.

    Attributes:
        temperature: ``[batch, 1]``, clamped away from zero; greedy rows are
            overwritten afterwards, so their value here is irrelevant.
        top_p: ``[batch, 1]`` nucleus threshold.
        repetition_penalty: ``[batch, 1]`` penalty factor.
        greedy: ``[batch, 1]`` bool, ``True`` where ``temperature == 0``.
        all_greedy: Whether every row is greedy, letting the sampler skip the
            softmax and the sort entirely.
        any_penalty: Whether any row has a penalty other than ``1.0``; when not,
            the generated-span gather can be skipped.
        all_top_p_one: Whether every stochastic row is plain categorical
            sampling, enabling a sort-free single-device path.
        needs_full_vocab: Whether any stochastic row needs the probability tail
            that the normal bounded candidate pool omits.
        logprobs_ks: Per-row top-k widths for logprob reporting, ``None`` for a
            row that opted out. Host-side: it is control flow (how wide a
            top-k to run), not a kernel input, so it never becomes a tensor.
    """

    temperature: torch.Tensor
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    greedy: torch.Tensor
    all_greedy: bool
    any_penalty: bool
    all_top_p_one: bool
    needs_full_vocab: bool
    logprobs_ks: tuple[int | None, ...] = ()

    @classmethod
    def build(
        cls, params: Sequence[SamplingParams], device: str | torch.device
    ) -> BatchedSamplingParams:
        """Stack one :class:`SamplingParams` per sequence into device tensors."""
        if not params:
            raise ValueError("BatchedSamplingParams needs at least one row")

        def column(values: list[float]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(-1)

        greedy_flags = [p.is_greedy for p in params]
        return cls(
            # A greedy row would divide the logits by zero here. Its value never
            # reaches the output (torch.where picks the argmax for that row), but
            # the division still runs, and inf/NaN in an unused row is the kind of
            # thing that later shows up as an unexplained NaN somewhere else.
            temperature=column([p.temperature if not p.is_greedy else 1.0 for p in params]),
            top_p=column([p.top_p for p in params]),
            repetition_penalty=column([p.repetition_penalty for p in params]),
            greedy=torch.tensor(greedy_flags, dtype=torch.bool, device=device).unsqueeze(-1),
            all_greedy=all(greedy_flags),
            any_penalty=any(p.repetition_penalty != 1.0 for p in params),
            all_top_p_one=all(p.is_greedy or p.top_p == 1.0 for p in params),
            # A stochastic top_p=1 row must retain the whole distribution. In a
            # mixed batch the other rows still need sorting, so use the exact
            # full-vocabulary pool for the shared pass.
            needs_full_vocab=any(not p.is_greedy and p.top_p == 1.0 for p in params),
            logprobs_ks=tuple(p.logprobs for p in params),
        )


class Sampler:
    """Applies :class:`SamplingParams` to per-step logits.

    The two public methods differ only in whether the knobs are Python scalars (one
    configuration for the whole batch) or ``[batch, 1]`` tensors (one per request);
    both delegate to :meth:`_draw`, which is also the single place that knows whether
    the logits are the full vocabulary or this rank's slice of it.
    """

    @torch.inference_mode()
    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generated: GeneratedSpan | None = None,
    ) -> torch.Tensor:
        """Select the next token for each sequence.

        Args:
            logits: ``[batch, seq_len, vocab]`` or ``[batch, vocab]`` — this rank's
                slice of ``vocab`` under tensor parallelism. When a sequence dimension
                is present, only the last position is used.
            params: Sampling configuration.
            generated: Optional padded generated-token view used by
                ``repetition_penalty``; pass ``None`` when the penalty is off.

        Returns:
            ``[batch, 1]`` next-token ids.
        """
        ids, _ = self.sample_with_logprobs(logits, params, generated)
        return ids

    @torch.inference_mode()
    def sample_with_logprobs(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generated: GeneratedSpan | None = None,
    ) -> tuple[torch.Tensor, list[PositionLogprobs | None] | None]:
        """:meth:`sample` plus per-row logprob records when ``params.logprobs`` is set.

        The records describe the distribution actually drawn from: penalised
        logits divided by the temperature. Greedy rows divide by the clamped
        1.0, i.e. they report the raw model distribution — the same numbers
        HuggingFace's ``compute_transition_scores`` produces under
        ``do_sample=False``.

        Returns:
            ``(ids [batch, 1], records)``; ``records`` is ``None`` when
            ``params.logprobs`` is ``None``, which costs nothing extra.
        """
        return self._draw(
            logits,
            generated,
            penalty=params.repetition_penalty if params.repetition_penalty != 1.0 else None,
            temperature=params.temperature,
            top_p=params.top_p,
            all_greedy=params.is_greedy,
            greedy=None,
            all_top_p_one=params.top_p == 1.0,
            needs_full_vocab=params.top_p == 1.0,
            logprobs_ks=params.logprobs,
        )

    @torch.inference_mode()
    def sample_batched(
        self,
        logits: torch.Tensor,
        params: BatchedSamplingParams,
        generated: GeneratedSpan | None = None,
    ) -> torch.Tensor:
        """Select the next token for a batch whose rows are configured independently.

        Greedy and stochastic rows coexist: both branches are evaluated for the
        whole batch and combined with a select. Running them as two masked passes
        instead would need a device-side row count to size each sub-batch, and
        reading that back is exactly the per-step synchronisation the engine is
        built to avoid.

        Args:
            logits: ``[batch, seq_len, vocab]`` or ``[batch, vocab]``. When a
                sequence dimension is present, only the last position is used.
            params: Per-row sampling knobs for this batch.
            generated: Optional padded generated-token view used by the
                repetition penalty; pass ``None`` when no row enables it.

        Returns:
            ``[batch, 1]`` next-token ids.
        """
        ids, _ = self.sample_batched_with_logprobs(logits, params, generated)
        return ids

    @torch.inference_mode()
    def sample_batched_with_logprobs(
        self,
        logits: torch.Tensor,
        params: BatchedSamplingParams,
        generated: GeneratedSpan | None = None,
    ) -> tuple[torch.Tensor, list[PositionLogprobs | None] | None]:
        """:meth:`sample_batched` plus per-row logprob records.

        Rows are independent: those whose :class:`SamplingParams` set
        ``logprobs`` get a record, the others get ``None`` entries, and when no
        row asked the whole computation is skipped (``records is None``).
        """
        return self._draw(
            logits,
            generated,
            penalty=params.repetition_penalty if params.any_penalty else None,
            temperature=params.temperature,
            top_p=params.top_p,
            all_greedy=params.all_greedy,
            greedy=params.greedy,
            all_top_p_one=params.all_top_p_one,
            needs_full_vocab=params.needs_full_vocab,
            logprobs_ks=params.logprobs_ks or None,
        )

    def _draw(
        self,
        logits: torch.Tensor,
        generated: GeneratedSpan | None,
        *,
        penalty: float | torch.Tensor | None,
        temperature: float | torch.Tensor,
        top_p: float | torch.Tensor,
        all_greedy: bool,
        greedy: torch.Tensor | None,
        all_top_p_one: bool = False,
        needs_full_vocab: bool = False,
        logprobs_ks: int | Sequence[int | None] | None = None,
    ) -> tuple[torch.Tensor, list[PositionLogprobs | None] | None]:
        """Shared body of the four public sampling methods.

        ``greedy`` is ``None`` when the whole batch shares one configuration, in which
        case ``all_greedy`` already decided which branch to take and no select is
        needed. ``logprobs_ks`` is a per-row top-k width, or one scalar for the
        whole batch; ``None`` skips the logprob arithmetic entirely.
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        offset = local_vocab_offset(logits.shape[-1])

        if penalty is not None and generated is not None:
            logits = apply_repetition_penalty(logits, generated, penalty, offset or 0)

        sampled_ids = None
        if not all_greedy:
            if offset is None:
                probs = torch.softmax(logits / temperature, dim=-1)
                sampled_ids = (
                    torch.multinomial(probs, num_samples=1)
                    if all_top_p_one
                    else sample_top_p(
                        probs, top_p, k=None if needs_full_vocab else _TOP_P_CANDIDATES
                    )
                )
            else:
                sampled_ids = sharded_top_p(
                    logits,
                    temperature,
                    top_p,
                    offset,
                    k=None if needs_full_vocab or all_top_p_one else _TOP_P_CANDIDATES,
                )

        # The greedy branch is drawn only where a row will use it: a wholly
        # stochastic batch skips the argmax entirely (see ``greedy_ids``).
        if sampled_ids is None:
            ids = greedy_ids(logits, offset)
        elif greedy is None:
            ids = sampled_ids
        else:
            ids = torch.where(greedy, greedy_ids(logits, offset), sampled_ids)

        if logprobs_ks is None:
            return ids, None
        ks = [logprobs_ks] * logits.shape[0] if isinstance(logprobs_ks, int) else list(logprobs_ks)
        k_max = max((k for k in ks if k is not None), default=None)
        if k_max is None:
            return ids, None
        # TP: a non-greedy draw is per-rank (each rank's own RNG) until the
        # worker broadcasts the winner. A record must describe the token the
        # caller will actually see, so synchronise first; the worker's own
        # broadcast of the returned ids is then an idempotent no-op.
        if offset is not None and not all_greedy:
            ids = tensor_model_parallel_broadcast(ids)
        # A greedy whole-batch call arrives with temperature == 0.0, which the
        # draw itself never divides by. The records must describe the raw
        # distribution, so substitute the clamp BatchedSamplingParams applies.
        safe_temperature = temperature
        if not isinstance(temperature, torch.Tensor) and all_greedy:
            safe_temperature = 1.0
        scaled = logits.float() / safe_temperature
        chosen, top_values, top_ids = _distribution_records(scaled, ids, offset, k_max)
        return ids, _to_records(ids, ks, chosen, top_values, top_ids)
