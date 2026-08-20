"""Canonical golden-regression cases, shared by the test and the update script.

Kept in one module so ``tests/golden/test_token_parity.py`` and
``scripts/golden_tokens.py`` cannot drift: if the script recorded a different
set of prompts than the test replays, the committed baseline would be checked
against cases it was never generated from.

Each case is ``(name, prompts, max_gen_len)``. Between them they cover the
layouts that have historically broken independently:

* ``single`` -- no batching at all,
* ``batch_uniform`` -- equal-length prompts, so padding is a no-op,
* ``batch_mixed`` -- a very short prompt beside a long one, which is what the
  packed-vs-padded prefill bug corrupted,
* ``batch8`` -- eight sequences, enough to cross the CUDA-graph capture buckets.
"""

from __future__ import annotations

CASES: list[tuple[str, list[str], int]] = [
    ("single", ["The future of artificial intelligence is"], 48),
    ("batch_uniform", ["How to learn python?", "How to learn c++?"], 32),
    ("batch_mixed", ["Hi", "The history of the Roman Empire spans many centuries, and"], 32),
    (
        "batch8",
        [
            "I believe the meaning of life is",
            "VGG is a very important cnn backbone,",
            "Can you introduce the American Civil War.",
            "who is the first president of the United States?",
            "How to learn c++, give me some code example.",
            "How to learn python, give me some code examples.",
            "How to learn llm, please introduce transformer",
            "How to learn cnn, please introduce resnet",
        ],
        32,
    ),
]

#: Repetition-penalty settings swept for every case. 1.0 is the plain path; 1.1
#: exercises ``apply_repetition_penalty``, whose vectorised rewrite is exactly
#: the kind of change that must not move a single token.
PENALTIES: tuple[float, ...] = (1.0, 1.1)

#: KV pool size used when collecting and when checking. Pinned because the
#: allocator's bump fast path depends on capacity, so a different pool could in
#: principle produce a different row layout.
MAX_GPU_NUM_BLOCKS = 8192
MAX_SEQ_LEN = 1024


def case_key(name: str, penalty: float) -> str:
    """Stable key for one (case, penalty) pair in the golden JSON."""
    return name if penalty == 1.0 else f"{name}_rp{penalty}"
