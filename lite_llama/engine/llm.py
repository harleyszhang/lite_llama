"""User-facing ``LLM`` entry point — a vLLM-style facade over the engine.

``LLM`` *is* an :class:`~lite_llama.engine.llm_engine.LLMEngine` (inheritance): it
adds prompt normalisation, multimodal preparation and ``RequestOutput`` packaging,
while the engine owns the prefill/decode loop and ``ModelRunner`` the single-device
forward. One ``LLM`` == one engine; request routing (DP) and tensor parallel (TP)
grow in the respective layers without touching this API.

Usage:
    llm = LLM(model="my_weight/Qwen2.5-0.5B")
    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0))
    print(out[0].outputs[0].text)
"""

from __future__ import annotations

from collections.abc import Iterator

from PIL import Image

from ..models.config import read_model_type
from ..models.registry import ModelRegistry, ModelSpec
from .llm_engine import LLMEngine
from .multimodal import MultimodalPreparer
from .outputs import CompletionOutput, RequestOutput
from .sampler import SamplingParams


def _resolve_spec(model: str) -> ModelSpec:
    """Resolve the checkpoint's architecture before the engine is built.

    The decision whether to build a multimodal preparer (and whether CUDA
    graphs are safe) must happen *before* ``LLMEngine.__init__`` runs, so the
    engine's own config load cannot be reused; the extra read costs a few KB.
    """
    return ModelRegistry.resolve(read_model_type(model))


class LLM(LLMEngine):
    """Generate completions for text or vision-language prompts.

    Args:
        model: HuggingFace checkpoint directory (``config.json`` plus ``*.safetensors``).
        tokenizer: Tokenizer location; defaults to ``model``.
        max_seq_len: Context bound; also caps the KV cache.
        max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
        device: Torch device string.
        use_cuda_graph: Capture decode CUDA graphs. ``None`` (default) enables
            them for text-only models and disables them for multimodal ones,
            whose vision tower changes control flow per prefill.
        quantization: Runtime weight quantisation (``"int8"``); ``None`` keeps
            the checkpoint's native format (fp16 or auto-detected fp8).
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        data_parallel_size: Reserved for DP support; must be 1 today.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool | None = None,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
    ) -> None:
        if data_parallel_size != 1:
            raise NotImplementedError(
                "data_parallel_size > 1 is not implemented yet; DP will route "
                "requests across engine workers at this entry layer"
            )

        spec = _resolve_spec(model)
        if use_cuda_graph is None:
            use_cuda_graph = not spec.is_multimodal
        # CUDA graphs are incompatible with TP (NCCL collectives inside the graph)
        if tensor_parallel_size > 1:
            use_cuda_graph = False

        super().__init__(
            checkpoints_dir=model,
            tokenizer_path=tokenizer,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
        )

        # Strategy: only multimodal checkpoints get a preparer (and pay for the
        # AutoProcessor import/load).
        self.multimodal = MultimodalPreparer(self, model) if spec.is_multimodal else None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def is_qwen3_vl(self) -> bool:
        """Whether the loaded model is Qwen3-VL (mrope position-id path)."""
        return self.multimodal is not None and self.multimodal.is_qwen3_vl

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        images: list[Image.Image] | None = None,
    ) -> list[RequestOutput]:
        """Generate one completion per prompt; returns vLLM-shaped outputs.

        Args:
            prompts: A single prompt or a batch.
            sampling_params: Defaults to :class:`SamplingParams` defaults.
            images: Images shared by every prompt (multimodal models only).
        """
        params = sampling_params or SamplingParams()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)

        if images is not None:
            completions, reasons = self._generate_multimodal(prompts, images, params)
        else:
            token_ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
            completions = LLMEngine.generate_text(self, token_ids, params)
            reasons = self.last_stop_reasons or ["length"] * len(prompts)

        return [
            RequestOutput(prompt=p, outputs=[CompletionOutput(0, text, reason)])
            for p, text, reason in zip(prompts, completions, reasons, strict=True)
        ]

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        images: list[Image.Image] | None = None,
    ) -> Iterator[list[str]]:
        """Yield incremental text per prompt at each decode step.

        Multimodal streaming serves one prompt per call (the processor path is
        single-request); text streaming is batched.
        """
        params = sampling_params or SamplingParams()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)

        if images is not None:
            preparer = self._require_multimodal()
            if len(prompts) != 1:
                raise ValueError("multimodal streaming accepts exactly one prompt")
            token_ids, mm_inputs, position_ids = preparer.prepare(prompts[0], images)
            yield from LLMEngine.generate(
                self, [token_ids], params, position_ids=position_ids, multi_modal_inputs=mm_inputs
            )
            return

        token_ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        yield from LLMEngine.generate(self, token_ids, params)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _require_multimodal(self) -> MultimodalPreparer:
        if self.multimodal is None:
            raise ValueError(
                "images were provided but the loaded model is text-only; "
                "use a vision-language checkpoint (llava / qwen3_vl)"
            )
        return self.multimodal

    def _generate_multimodal(
        self, prompts: list[str], images: list[Image.Image], params: SamplingParams
    ) -> tuple[list[str], list[str | None]]:
        """Generate per prompt with shared images (the processor path is single-request)."""
        preparer = self._require_multimodal()
        completions: list[str] = []
        reasons: list[str | None] = []
        for prompt in prompts:
            token_ids, mm_inputs, position_ids = preparer.prepare(prompt, images)
            completions.extend(
                LLMEngine.generate_text(
                    self, [token_ids], params,
                    position_ids=position_ids, multi_modal_inputs=mm_inputs,
                )
            )
            # last_stop_reasons is overwritten per engine call — collect as we go.
            reasons.append(self.last_stop_reasons[0] if self.last_stop_reasons else None)
        return completions, reasons
