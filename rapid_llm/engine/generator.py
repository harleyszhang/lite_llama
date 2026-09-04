"""Backward-compatible ``TextGenerator`` / ``VisionGenerator`` wrappers.

Both classes build and hold an
:class:`~rapid_llm.engine.llm_engine.LLMEngine` and forward ``generate``
to it — thin facades kept so older scripts keep working while new code
uses :class:`~rapid_llm.engine.llm.LLM`.

Usage:
    gen = TextGenerator(checkpoints_dir, tokenizer_path)
    texts = gen.generate(prompts, params)
"""

from __future__ import annotations

from collections.abc import Iterator

from PIL import Image

from .llm import LLM
from .sampler import SamplingParams


class TextGenerator:
    """Text-only batch/stream generation. Prefer :class:`LLM` for new code.

    ``use_cuda_graph`` defaults to ``True``: an eager decode step launches ~300
    tiny kernels whose launch latency dominates the few milliseconds of actual
    GPU work, so replaying a captured graph is several times faster for the same
    arithmetic. Capture falls back to eager automatically if it cannot fit.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str | None = None,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = True,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
        cuda_graph_lazy: bool = False,
        hf_overrides: dict[str, object] | None = None,
    ) -> None:
        self._llm = LLM(
            model=checkpoints_dir,
            tokenizer=tokenizer_path,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            kv_cache_dtype=kv_cache_dtype,
            cuda_graph_lazy=cuda_graph_lazy,
            hf_overrides=hf_overrides,
        )
        # Legacy attribute: callers (e.g. the CLI) read ``engine.last_stop_reasons``.
        self.engine = self._llm
        self.tokenizer = self._llm.tokenizer

    def generate(self, prompts: list[str], params: SamplingParams | None = None) -> list[str]:
        """Return a full completion for each prompt."""
        return [output.text for output in self._llm.generate(prompts, params)]

    def stream(
        self, prompts: list[str], params: SamplingParams | None = None
    ) -> Iterator[list[str]]:
        """Yield incremental text per step for each prompt."""
        yield from self._llm.stream(prompts, params)


class VisionGenerator:
    """Image-conditioned generation for LLaVA and Qwen3-VL. Prefer :class:`LLM`.

    ``use_cuda_graph`` defaults to ``True`` as with text models: only the decode
    step is captured, and by then the vision tokens are ordinary KV-cache rows,
    so the vision tower and the DeepStack hooks never run inside a capture.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = True,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
        cuda_graph_lazy: bool = False,
    ) -> None:
        self._llm = LLM(
            model=checkpoints_dir,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            kv_cache_dtype=kv_cache_dtype,
            cuda_graph_lazy=cuda_graph_lazy,
        )
        self.engine = self._llm
        self.device = device
        self.is_qwen3_vl = self._llm.is_qwen3_vl

    def generate(
        self,
        prompt: str,
        images: list[Image.Image],
        params: SamplingParams | None = None,
    ) -> str:
        """Return a full completion for one image-conditioned prompt."""
        return self._llm.generate([prompt], params, images=images)[0].text

    def stream(
        self,
        prompt: str,
        images: list[Image.Image],
        params: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Yield incremental text for one image-conditioned prompt."""
        for step in self._llm.stream([prompt], params, images=images):
            yield step[0]
