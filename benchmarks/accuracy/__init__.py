"""Accuracy benches: parity against reference implementations, plus the
GSM8K vLLM comparison arm. One module per evaluated model family; the
analysis steps are subcommands of the module that produced the JSONs they
consume.

* :mod:`benchmarks.accuracy.deepseek` — V3-4layers / V4-Flash trimmed checkpoints: reference-vs-lite parity, the vLLM vote and the agreement analyses
* :mod:`benchmarks.accuracy.dspark_to_hf` — the DSpark -> transformers loader the V4 sides share
* :mod:`benchmarks.accuracy.convert_v4_hf` — one-shot DSpark -> bf16 HF safetensors on disk
* :mod:`benchmarks.accuracy.gsm8k_vllm` — the vLLM side of tests/evals/gsm8k.py
"""
