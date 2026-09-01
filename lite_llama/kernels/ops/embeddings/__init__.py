"""Embedding domain: the vocab-parallel embedding lookup.

Registers the domain's spec row and re-exports the gather kernel
:func:`~lite_llama.kernels.ops.embeddings.vocab_embedding.vocab_parallel_embedding`
behind the sharded embedding layer.

Usage:
    from lite_llama.kernels import vocab_parallel_embedding
"""
