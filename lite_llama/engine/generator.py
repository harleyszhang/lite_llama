"""Backward-compatible wrappers around :class:`~lite_llama.engine.llm.LLM`.

``TextGenerator`` / ``VisionGenerator`` predate the unified ``LLM`` entry
point and are kept as thin delegating shells so existing callers (CLI,
examples, benchmarks, tests) keep working unchanged. **New code should use
:class:`~lite_llama.engine.llm.LLM` directly** — it subsumes both wrappers
with a single ``generate``/``stream`` API.

All multimodal preparation lives in
:class:`~lite_llama.engine.multimodal.MultimodalPreparer`; the generation loop
lives in :class:`~lite_llama.engine.llm_engine.LLMEngine`. Nothing is
implemented here except argument/result adaptation.
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
    ) -> None:
        self._llm = LLM(
            model=checkpoints_dir,
            tokenizer=tokenizer_path,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
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

    CUDA graphs stay disabled for these models: the vision tower and the
    DeepStack hook change control flow per prefill.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
    ) -> None:
        self._llm = LLM(
            model=checkpoints_dir,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=False,
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
