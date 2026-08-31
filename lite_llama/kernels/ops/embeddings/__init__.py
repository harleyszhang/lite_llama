"""Embedding domain: the vocab-parallel embedding lookup.

No registration rows: the lookup is bandwidth over a sharded table and the
module layer (``modules/vocab_parallel.py``) calls
:func:`~lite_llama.kernels.ops.embeddings.vocab_embedding.vocab_parallel_embedding`
directly — the dispatch question "which implementation?" has no meaningful
second answer until an external backend ships a fused variant worth ranking.

Usage:
    from lite_llama.kernels.ops.embeddings.vocab_embedding import (
        vocab_parallel_embedding,
    )
"""
