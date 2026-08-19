"""Token sampling: turns a step's logits into the next token id per sequence.

The four legacy generators each inlined their own copy of the temperature +
top-p logic (and one shipped a second copy of ``sample_top_p`` at module scope).
:class:`Sampler` is the single implementation; :class:`SamplingParams` carries the
knobs. ``temperature == 0`` is treated as greedy decoding.
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


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: list[torch.Tensor], penalty: float
) -> torch.Tensor:
    """HuggingFace-style repetition penalty over the generated span.

    For every token that already occurred, its logit is divided by ``penalty``
    when positive (de-moted) and multiplied when negative, exactly as
    ``transformers``' ``RepetitionPenaltyLogitsProcessor``. Only the *generated*
    tokens are penalised — the prompt is user-supplied context.

    Args:
        logits: ``[batch, vocab]``.
        token_ids: One 1-D tensor of generated token ids per batch row.
        penalty: Penalty factor; ``1.0`` is a no-op.

    Returns:
        New ``[batch, vocab]`` logits (input is left untouched).
    """
    out = logits.clone()
    for i, ids in enumerate(token_ids):
        if ids.numel() == 0:
            continue
        seen = torch.unique(ids)
        scores = out[i, seen]
        out[i, seen] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return out


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
        generated_tokens: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Select the next token for each sequence.

        Args:
            logits: ``[batch, seq_len, vocab]`` or ``[batch, vocab]``. When a sequence
                dimension is present, only the last position is used.
            params: Sampling configuration.
            generated_tokens: Optional per-sequence generated token ids used by
                ``repetition_penalty``; pass ``None`` when the penalty is off.

        Returns:
            ``[batch, 1]`` next-token ids.
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if params.repetition_penalty != 1.0 and generated_tokens is not None:
            logits = apply_repetition_penalty(logits, generated_tokens, params.repetition_penalty)

        if params.is_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)

        probs = torch.softmax(logits / params.temperature, dim=-1)
        return sample_top_p(probs, params.top_p)
