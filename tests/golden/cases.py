"""Canonical golden-regression cases, shared by the test and the update script.

One row per (model, penalty, scheme) with fixed prompts and params, so
``test_token_parity`` and the golden-update script always agree on
what "the baseline" means.

Usage:
    from tests.golden.cases import case_key
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard text cases (offline batch)
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Continuous-batching cases (online engine path)
# --------------------------------------------------------------------------- #
CB_CASES: list[tuple[str, list[str], int]] = [
    ("cb_single", ["Explain what a GPU is in one sentence."], 32),
    (
        "cb_interleaved",
        [
            "Name the capital of Japan.",
            "Write a haiku about rain.",
            "Summarise the theory of relativity in three sentences.",
        ],
        48,
    ),
]

# --------------------------------------------------------------------------- #
# Quantisation path cases (runtime int8/fp8/smoothquant)
# --------------------------------------------------------------------------- #
#: (scheme_name, prompts, max_gen_len). The test records separate baselines
#: per scheme so a kernel change in one path does not invalidate the others.
QUANT_CASES: list[tuple[str, list[str], int]] = [
    ("quant_single", ["The capital of France is"], 32),
    ("quant_batch", ["What is deep learning?", "Describe a neural network."], 32),
]

#: Runtime quantisation schemes exercised by the golden gate.
QUANT_SCHEMES: tuple[str, ...] = ("int8", "fp8", "smoothquant")

#: Repetition-penalty settings swept for every case. 1.0 is the plain path; 1.1
#: exercises ``apply_repetition_penalty``, whose vectorised rewrite is exactly
#: the kind of change that must not move a single token.
PENALTIES: tuple[float, ...] = (1.0, 1.1)

#: KV pool size used when collecting and when checking. Pinned because the
#: allocator's bump fast path depends on capacity, so a different pool could in
#: principle produce a different row layout.
MAX_GPU_NUM_BLOCKS = 8192
MAX_SEQ_LEN = 1024


def case_key(name: str, penalty: float, scheme: str = "") -> str:
    """Stable key for one (case, penalty[, scheme]) combination in the golden JSON."""
    key = name if penalty == 1.0 else f"{name}_rp{penalty}"
    return f"{key}_{scheme}" if scheme else key
