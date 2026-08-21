"""Accuracy evaluation harness.

Mirrors vLLM's ``tests/evals/``: a benchmark dataset is turned into prompts, run
through the engine, scored, and compared against a per-checkpoint threshold
declared in ``configs/*.yaml``. The difference is the execution path — vLLM
launches a server and talks OpenAI HTTP, while lite_llama has no server, so
:mod:`tests.evals.runner` drives the offline :class:`~lite_llama.LLM` directly.
"""
