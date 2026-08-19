"""High-level, user-facing generation entry points.

:class:`TextGenerator` wraps :class:`~lite_llama.engine.llm_engine.LLMEngine` with
prompt tokenisation; :class:`VisionGenerator` adds the HuggingFace processor path
for LLaVA and Qwen3-VL, including mrope position ids for Qwen3-VL.

Both reuse the single generation loop in the engine; they only differ in how a
prompt (and its images) become token ids + optional multimodal tensors.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from .llm_engine import LLMEngine
from .sampler import SamplingParams


class TextGenerator:
    """Text-only generation from string prompts."""

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str | None = None,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = False,
    ) -> None:
        self.engine = LLMEngine(
            checkpoints_dir,
            tokenizer_path,
            max_seq_len,
            max_gpu_num_blocks,
            device,
            use_cuda_graph=use_cuda_graph,
        )
        self.tokenizer = self.engine.tokenizer

    def _encode(self, prompts: list[str]) -> list[list[int]]:
        return [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]

    def generate(self, prompts: list[str], params: SamplingParams | None = None) -> list[str]:
        """Return a full completion for each prompt."""
        params = params or SamplingParams()
        return self.engine.generate_text(self._encode(prompts), params)

    def stream(
        self, prompts: list[str], params: SamplingParams | None = None
    ) -> Iterator[list[str]]:
        """Yield incremental text per step for each prompt."""
        params = params or SamplingParams()
        yield from self.engine.generate(self._encode(prompts), params)


class VisionGenerator:
    """Image/video conditioned generation for LLaVA and Qwen3-VL.

    The HuggingFace processor for the checkpoint does all prompt+image preparation,
    including expanding the ``<image>`` marker to the right number of placeholder
    tokens. That keeps this class model-agnostic: the only model-specific step is
    computing mrope position ids for Qwen3-VL, which is delegated to the model's own
    ``get_rope_index`` when present.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
    ) -> None:
        self.engine = LLMEngine(
            checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
        )
        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoints_dir, trust_remote_code=True)
        self.is_qwen3_vl = self.engine.executor.spec.model_type == "qwen3_vl"

    def _prepare(
        self, prompt: str, images: list[Image.Image]
    ) -> tuple[list[int], dict[str, Any], torch.Tensor | None]:
        """Run the processor and package engine inputs.

        Returns:
            ``(token_ids, multi_modal_inputs, position_ids)`` where ``position_ids``
            is the mrope tensor for Qwen3-VL or ``None`` for LLaVA.
        """
        if self.is_qwen3_vl:
            prompt = self._wrap_qwen3_vl_prompt(prompt, len(images))
        batch = self.processor(text=prompt, images=images, return_tensors="pt")
        input_ids = batch["input_ids"]
        token_ids = input_ids[0].tolist()

        multi_modal_inputs: dict[str, Any] = {}
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in batch and batch[key] is not None:
                multi_modal_inputs[key] = batch[key].to(self.device)

        position_ids = self._mrope_positions(input_ids, batch) if self.is_qwen3_vl else None
        return token_ids, multi_modal_inputs, position_ids

    def _wrap_qwen3_vl_prompt(self, prompt: str, num_images: int) -> str:
        """Wrap a plain user prompt in the Qwen3-VL chat template.

        Unlike LLaVA there is no ``<image>`` marker to expand: the processor only
        grows ``<|image_pad|>`` placeholders that the chat template inserts between
        ``<|vision_start|>``/``<|vision_end|>``. A prompt that already carries the
        vision markers is passed through untouched.
        """
        if "<|vision_start|>" in prompt:
            return prompt
        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _mrope_positions(self, input_ids: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """Compute Qwen3-VL 3D position ids, reusing the HF reference implementation."""
        hf_model = self.engine.executor.model
        # Qwen3-VL's config-only get_rope_index is the vetted source of the (t, h, w)
        # index maths; reimplementing its ~120 lines here would only add risk. It is
        # an *unbound* method that reads only ``config`` and ``get_vision_position_ids``
        # off its host, so a tiny adapter exposes both from the lite_llama model.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        class _RopeIndexHost:
            """Duck-typed ``self`` for Qwen3VLModel.get_rope_index."""

            def __init__(self, config: Any) -> None:
                self.config = config

            get_vision_position_ids = Qwen3VLModel.get_vision_position_ids

        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "image_grid_thw": batch.get("image_grid_thw"),
            "video_grid_thw": batch.get("video_grid_thw"),
            "attention_mask": batch.get("attention_mask"),
        }
        # transformers >= 5 groups tokens by modality through mm_token_type_ids
        # (returned by the processor) instead of scanning for vision markers.
        if "mm_token_type_ids" in inspect.signature(Qwen3VLModel.get_rope_index).parameters:
            kwargs["mm_token_type_ids"] = batch.get("mm_token_type_ids")

        position_ids, _ = Qwen3VLModel.get_rope_index(_RopeIndexHost(hf_model.config), **kwargs)
        return position_ids.to(self.device)

    def generate(
        self,
        prompt: str,
        images: list[Image.Image],
        params: SamplingParams | None = None,
    ) -> str:
        """Return a full completion for one image-conditioned prompt."""
        params = params or SamplingParams()
        token_ids, mm_inputs, position_ids = self._prepare(prompt, images)
        return self.engine.generate_text(
            [token_ids], params, position_ids=position_ids, multi_modal_inputs=mm_inputs
        )[0]

    def stream(
        self,
        prompt: str,
        images: list[Image.Image],
        params: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Yield incremental text for one image-conditioned prompt."""
        params = params or SamplingParams()
        token_ids, mm_inputs, position_ids = self._prepare(prompt, images)
        for step in self.engine.generate(
            [token_ids], params, position_ids=position_ids, multi_modal_inputs=mm_inputs
        ):
            yield step[0]
