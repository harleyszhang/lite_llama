"""Token sampling: turns a step's logits into the next token id per sequence.

The single implementation of the temperature + top-p logic the legacy generators
each used to inline: :class:`Sampler` does the work, :class:`SamplingParams` carries
the knobs, and ``temperature == 0`` is greedy decoding.

Usage:
    next_ids = Sampler().sample(logits, SamplingParams(temperature=0.0))
"""

from __future__ import annotations

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
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )

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
    logits: torch.Tensor, generated: GeneratedSpan, penalty: float
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
        penalty: Penalty factor; ``1.0`` is a no-op.

    Returns:
        New ``[batch, vocab]`` logits (input is left untouched).
    """
    batch, vocab = logits.shape
    seen = torch.zeros(batch, vocab + 1, dtype=torch.bool, device=logits.device)
    columns = torch.where(generated.mask, generated.token_ids, vocab)
    seen.scatter_(1, columns, True)
    penalised = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen[:, :vocab], penalised, logits)


def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling.

    Keeps the smallest set of highest-probability tokens whose cumulative mass just
    exceeds ``top_p``, renormalises, and samples one token from it.

    Args:
        probs: ``[batch, vocab]`` probability distribution.
        top_p: Cumulative probability threshold.

    Returns:
        ``[batch, 1]`` sampled token ids.
    """
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Drop tokens once the mass *before* them already exceeds top_p.
    drop_mask = cumulative - sorted_probs > top_p
    sorted_probs[drop_mask] = 0.0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
    choice = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_idx, -1, choice)


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
