"""Device-resident stop bookkeeping for a decode batch (mirrors vLLM).

:class:`StopCriteria` holds one writable flag per sequence plus the batch's
stop-token rows, and after each step marks exactly the sequences whose new
token is a stop id. :func:`detect_repetition` is the CPU-side loop breaker.

Usage:
    criteria = StopCriteria(batch_size, stop_ids, vocab_size)
    detect_repetition(text, window, min_reps)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

# How often the engine may ask "is the whole batch done?". Answering requires a
# device-to-host read, so it is amortised over this many decode steps.
POLL_INTERVAL = 8

_REASON_NAMES = ("length", "eos", "repeat")
_REASON_EOS = 1
_REASON_REPEAT = 2


def load_stop_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids from the tokenizer *and* ``generation_config.json``.

    HuggingFace checkpoints declare the full stop set — possibly a list,
    e.g. Qwen's ``[151645, 151643]`` — in ``generation_config.json``, while the
    tokenizer only knows one ``eos_token_id``. Instruct models usually terminate
    on a different special token than the tokenizer default (``<|im_end|>`` vs
    ``<|endoftext|>``), so without the generation config a finished request
    would keep decoding until ``max_gen_len``.
    """
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    gen_cfg_path = Path(model_path) / "generation_config.json"
    if gen_cfg_path.is_file():
        try:
            with open(gen_cfg_path) as f:
                gen_cfg = json.load(f)
            eos = gen_cfg.get("eos_token_id")
            if isinstance(eos, int):
                ids.add(eos)
            elif isinstance(eos, list):
                ids.update(int(e) for e in eos)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # a broken generation_config must not block inference
    return ids


# Repetition detection: loops that only vary in numbers ("... for 13 years ... "
# "... for 11 years ...") are still loops, so digits are normalised away before
# comparing the trailing window against earlier text.
_DIGIT_RE = re.compile(r"\d+")


def detect_repetition(text: str, window: int = 128, min_reps: int = 3) -> bool:
    """Whether ``text`` degenerated into a repeating template loop.

    Takes the last ``window`` characters (digits normalised to a placeholder)
    and reports ``True`` when that exact block already occurred at least
    ``min_reps`` times earlier in the text. Normal prose essentially never
    repeats a 128-character block verbatim three times, while base-model
    repetition loops do it within one or two loop cycles — including loops
    whose only variation is a counting number, thanks to the normalisation.
    """
    if len(text) < window * (min_reps + 1):
        return False
    norm = _DIGIT_RE.sub("<n>", text)
    recent = norm[-window:]
    return norm[:-window].count(recent) >= min_reps


class StopCriteria:
    """Tracks which sequences finished, and why, without per-step syncs.

    Args:
        batch_size: Number of sequences in the batch.
        stop_token_ids: Token ids that terminate a sequence.
        vocab_size: Vocabulary size, used to size the stop-token lookup table.
        device: Torch device string.
    """

    def __init__(
        self,
        batch_size: int,
        stop_token_ids: set[int],
        vocab_size: int,
        device: str = "cuda",
    ) -> None:
        self.batch_size = batch_size
        self._finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # 0 = ran to the length cap, 1 = stop token, 2 = repetition breaker.
        self._reason = torch.zeros(batch_size, dtype=torch.uint8, device=device)
        # A boolean lookup table beats ``torch.isin`` in the loop: membership
        # becomes a single gather with no temporary allocation, and it captures
        # cleanly into a CUDA graph.
        self._is_stop = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if stop_token_ids:
            ids = torch.tensor(sorted(stop_token_ids), dtype=torch.long, device=device)
            self._is_stop[ids] = True

    @property
    def finished(self) -> torch.Tensor:
        """Device bool tensor, ``True`` where the sequence has stopped."""
        return self._finished

    def update(self, next_token: torch.Tensor, writable: torch.Tensor) -> None:
        """Record stop tokens for this step. Pure device work, never syncs.

        Args:
            next_token: ``[batch]`` ids just sampled.
            writable: ``[batch]`` bool, ``False`` where the position belongs to
                the original prompt and the sampled token is discarded.
        """
        hit = writable & self._is_stop[next_token] & ~self._finished
        self._reason = torch.where(hit, torch.full_like(self._reason, _REASON_EOS), self._reason)
        self._finished |= hit

    def mark_repeat(self, index: int) -> None:
        """Flag one sequence as stopped by the repetition breaker."""
        self._finished[index] = True
        self._reason[index] = _REASON_REPEAT

    def all_finished(self) -> bool:
        """Whether every sequence has stopped. Costs one device-to-host read."""
        return bool(self._finished.all())

    def reasons(self) -> list[str]:
        """Per-sequence stop reason. Costs one device-to-host read; call once."""
        return [_REASON_NAMES[int(code)] for code in self._reason.tolist()]
