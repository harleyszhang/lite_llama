"""Multimodal prompt preparation (Strategy seam for the LLM entry point).

:class:`MultimodalPreparer` owns everything that turns ``(prompt, images)``
into engine-ready ``(token_ids, multi_modal_inputs, position_ids)``:

* the HuggingFace ``AutoProcessor`` call (expands image placeholders),
* the Qwen3-VL chat-template wrapping (its processor grows ``<|image_pad|>``
  placeholders only inside a chat template),
* mrope 3D position ids for Qwen3-VL (delegated to the vetted HF reference
  implementation rather than reimplemented).

Extracted from the legacy ``VisionGenerator`` so the logic lives in exactly
one place, consumed by both :class:`~lite_llama.engine.llm.LLM` and the
backward-compatible generator wrappers.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from transformers import AutoProcessor

if TYPE_CHECKING:
    from .llm_engine import LLMEngine


class MultimodalPreparer:
    """Prepares image-conditioned prompts for LLaVA-style and Qwen3-VL models."""

    def __init__(self, engine: LLMEngine, model_dir: str) -> None:
        self._engine = engine
        self.device = engine.device
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.is_qwen3_vl = engine.model_runner.spec.model_type == "qwen3_vl"

    def prepare(
        self, prompt: str, images: list[Image.Image]
    ) -> tuple[list[int], dict[str, Any], torch.Tensor | None]:
        """Run the processor and package engine inputs.

        Returns:
            ``(token_ids, multi_modal_inputs, position_ids)`` where
            ``position_ids`` is the mrope tensor for Qwen3-VL or ``None``.
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
        hf_model = self._engine.model_runner.model
        # Qwen3-VL's config-only get_rope_index is the vetted source of the (t, h, w)
        # index maths; reimplementing its ~120 lines here would only add risk. It is
        # an *unbound* method that reads only ``config`` and ``get_vision_position_ids``
        # off its host, so a tiny adapter exposes both from the lite_llama model. The
        # config it wants is the raw HF one, not lite_llama's wrapper.
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

        position_ids, _ = Qwen3VLModel.get_rope_index(
            _RopeIndexHost(hf_model.config.hf_config), **kwargs
        )
        return position_ids.to(self.device)
