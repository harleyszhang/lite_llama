"""Execution layer: model building/loading, KV cache, CUDA graphs, forward steps.

Turns a resolved config into a running model and drives each forward pass:
:class:`~rapid_llm.executor.model_runner.ModelRunner` builds via the loader, sizes
the paged KV cache, and dispatches prefill/decode (optionally CUDA-graphed).

Usage:
    runner = ModelRunner.build(config, ...); logits = runner.forward(ids, pos, mm)
"""
