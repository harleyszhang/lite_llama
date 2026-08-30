"""Token sampling: turns a step's logits into the next token id per sequence.

The single implementation of the temperature + top-p logic the legacy generators
each used to inline: :class:`Sampler` does the work, :class:`SamplingParams` carries
the knobs, and ``temperature == 0`` is greedy decoding.

A one-shot batch shares one :class:`SamplingParams` across every sequence, but an
online batch does not: each request arrives with its own temperature, top-p and
penalty. :class:`BatchedSamplingParams` holds those knobs as per-row tensors so
:meth:`Sampler.sample_batched` still samples the whole batch in one pass instead
of looping over requests.

Under tensor parallelism the logits arriving here are a *slice* of the vocabulary
(:class:`~lite_llama.modules.vocab_parallel.ParallelLMHead` does not gather), and the
sampler reconstructs the global distribution without ever assembling one:
``log_softmax(x)_i = x_i - logsumexp(x)``, so two **scalars per row** — the maximum and
the sum of exponentials — are all that must cross the wire. Candidates then come from a
local top-k, because the union of per-rank top-k provably contains the global top-k, so
the gather is ``O(k * tp)`` and independent of the vocabulary size.

Usage:
    next_ids = Sampler().sample(logits, SamplingParams(temperature=0.0))
    next_ids = Sampler().sample_batched(logits, BatchedSamplingParams.build(...))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..distributed.parallel_state import (
    all_gather_tp,
    all_reduce_max_tp,
    all_reduce_tp,
    get_tp_rank,
    get_tp_world_size,
)


def local_vocab_offset(local_width: int) -> int | None:
    """First global token id of this rank's logits slice, or ``None`` when TP is off.

    The shard is derived from the ambient parallel state and the width of the tensor
    itself rather than threaded down from the model, for the same reason
    :func:`~lite_llama.distributed.parallel_state.all_reduce_tp` is: there is exactly one
    vocabulary split per process, and a sampler that has to be *told* about it is a
    sampler that can be told wrong. It holds because every logits producer in the
    package is a :class:`~lite_llama.modules.vocab_parallel.ParallelLMHead`.
    """
    if get_tp_world_size() == 1:
        return None
    return get_tp_rank() * local_width


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
    """

    temperature: float = 0.6
    top_p: float = 0.9
    max_gen_len: int | None = None
    repetition_penalty: float = 1.1
    stop_on_repeat: bool = True

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_gen_len is not None and self.max_gen_len < 1:
            raise ValueError(f"max_gen_len must be >= 1 or None, got {self.max_gen_len}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")

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
    best = all_reduce_max_tp(local_best.clone())
    ids = local_id + vocab_offset
    losers = torch.full_like(ids, torch.iinfo(ids.dtype).max)
    candidates = torch.where(local_best == best, ids, losers)
    return all_gather_tp(candidates).amin(dim=-1, keepdim=True)


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
    row_max = all_reduce_max_tp(scaled.amax(dim=-1, keepdim=True))
    row_sum = all_reduce_tp((scaled - row_max).exp().sum(dim=-1, keepdim=True))
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
    pool = all_gather_tp(local_top)
    pool_ids = all_gather_tp(local_ids + vocab_offset)
    global_k = pool.shape[-1] if k is None else min(k, pool.shape[-1])
    top_probs, order = torch.topk((pool - log_z).exp(), global_k, dim=-1)
    return _draw_from_nucleus(top_probs, pool_ids.gather(-1, order), top_p)


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
    """

    temperature: torch.Tensor
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    greedy: torch.Tensor
    all_greedy: bool
    any_penalty: bool
    all_top_p_one: bool
    needs_full_vocab: bool

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
    ) -> torch.Tensor:
        """Shared body of :meth:`sample` and :meth:`sample_batched`.

        ``greedy`` is ``None`` when the whole batch shares one configuration, in which
        case ``all_greedy`` already decided which branch to take and no select is
        needed.
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        offset = local_vocab_offset(logits.shape[-1])

        if penalty is not None and generated is not None:
            logits = apply_repetition_penalty(logits, generated, penalty, offset or 0)

        if offset is None:
            greedy_ids = torch.argmax(logits, dim=-1, keepdim=True)
            if all_greedy:
                return greedy_ids
            probs = torch.softmax(logits / temperature, dim=-1)
            if all_top_p_one:
                sampled_ids = torch.multinomial(probs, num_samples=1)
            else:
                sampled_ids = sample_top_p(
                    probs,
                    top_p,
                    k=None if needs_full_vocab else _TOP_P_CANDIDATES,
                )
        else:
            greedy_ids = global_argmax(logits, offset)
            if all_greedy:
                return greedy_ids
            sampled_ids = sharded_top_p(
                logits,
                temperature,
                top_p,
                offset,
                k=None if needs_full_vocab or all_top_p_one else _TOP_P_CANDIDATES,
            )

        if greedy is None:
            return sampled_ids
        return torch.where(greedy, greedy_ids, sampled_ids)
