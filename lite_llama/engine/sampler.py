"""Token sampling: turns a step's logits into the next token id per sequence.

The single implementation of the temperature + top-p logic the legacy generators
each used to inline: :class:`Sampler` does the work, :class:`SamplingParams` carries
the knobs, and ``temperature == 0`` is greedy decoding.

A one-shot batch shares one :class:`SamplingParams` across every sequence, but an
online batch does not: each request arrives with its own temperature, top-p and
penalty. :class:`BatchedSamplingParams` holds those knobs as per-row tensors so
:meth:`Sampler.sample_batched` still samples the whole batch in one pass instead
of looping over requests.

Usage:
    next_ids = Sampler().sample(logits, SamplingParams(temperature=0.0))
    next_ids = Sampler().sample_batched(logits, BatchedSamplingParams.build(...))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


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
    logits: torch.Tensor, generated: GeneratedSpan, penalty: float | torch.Tensor
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
        logits: ``[batch, vocab]``.
        generated: Padded generated-token view for the batch.
        penalty: Penalty factor; ``1.0`` is a no-op. A ``[batch, 1]`` tensor
            applies a different factor per row, which is what an online batch of
            independently configured requests needs.

    Returns:
        New ``[batch, vocab]`` logits (input is left untouched).
    """
    batch, vocab = logits.shape
    seen = torch.zeros(batch, vocab + 1, dtype=torch.bool, device=logits.device)
    columns = torch.where(generated.mask, generated.token_ids, vocab)
    seen.scatter_(1, columns, True)
    penalised = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen[:, :vocab], penalised, logits)


# Nucleus sampling never touches most of the vocabulary: real model
# distributions put far more than ``top_p`` mass on the top few dozen tokens,
# so a 1024-wide pool makes the draw identical to a full-vocabulary sort in
# every non-degenerate case while skipping the sort over the rest.
_TOP_P_CANDIDATES = 1024


def sample_top_p(
    probs: torch.Tensor, top_p: float | torch.Tensor, k: int = _TOP_P_CANDIDATES
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
        k: Candidate pool size, clamped to the vocabulary.

    Returns:
        ``[batch, 1]`` sampled token ids.
    """
    k = min(k, probs.shape[-1])
    top_probs, top_idx = torch.topk(probs, k, dim=-1)  # already descending
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
    """

    temperature: torch.Tensor
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    greedy: torch.Tensor
    all_greedy: bool
    any_penalty: bool

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
        )


class Sampler:
    """Applies :class:`SamplingParams` to per-step logits."""

    @torch.inference_mode()
    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generated: GeneratedSpan | None = None,
    ) -> torch.Tensor:
        """Select the next token for each sequence.

        Args:
            logits: ``[batch, seq_len, vocab]`` or ``[batch, vocab]``. When a sequence
                dimension is present, only the last position is used.
            params: Sampling configuration.
            generated: Optional padded generated-token view used by
                ``repetition_penalty``; pass ``None`` when the penalty is off.

        Returns:
            ``[batch, 1]`` next-token ids.
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if params.repetition_penalty != 1.0 and generated is not None:
            logits = apply_repetition_penalty(logits, generated, params.repetition_penalty)

        if params.is_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)

        probs = torch.softmax(logits / params.temperature, dim=-1)
        return sample_top_p(probs, params.top_p)

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
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if params.any_penalty and generated is not None:
            logits = apply_repetition_penalty(logits, generated, params.repetition_penalty)

        greedy_ids = torch.argmax(logits, dim=-1, keepdim=True)
        if params.all_greedy:
            return greedy_ids

        probs = torch.softmax(logits / params.temperature, dim=-1)
        sampled_ids = sample_top_p(probs, params.top_p)
        return torch.where(params.greedy, greedy_ids, sampled_ids)
