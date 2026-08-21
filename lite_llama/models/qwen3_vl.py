"""Qwen3-VL: SigLIP-style vision tower + Qwen3 language model with mrope and DeepStack.

Two mechanisms make Qwen3-VL more than "encode image, scatter embeddings":

**mrope.** Each vision token carries a ``(t, h, w)`` position instead of one index,
and the rotary dimensions are split across those components in an interleaved
layout. :class:`~lite_llama.models.rotary_embedding.MRotaryEmbedding` handles that;
the ``[3, batch, seq_len]`` position ids are produced by the input processor.

**DeepStack.** The vision tower emits, in addition to the merged patch embeddings,
one extra feature map per entry of ``vision_config.deepstack_visual_indexes``. Those
are *added into the language model's hidden states at the vision token positions*
after the first few decoder layers (see arXiv:2406.04334). Skipping this does not
crash — it silently degrades quality — so it is wired in through the
:meth:`~lite_llama.models.base.CausalLM._after_layer` hook.

Parameter layout, and the HF checkpoint keys it is filled from::

    vision_tower.*        <- model.visual.*            (Qwen3VLVisionModel, HF names)
    language_model.*      <- model.language_model.*    (lite_llama names)
    language_model.lm_head_weight  <- the tied embedding table (absent from the
                                      checkpoint, which sets tie_word_embeddings)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from .base import CausalLM
from .config import ModelConfig
from .interfaces import (
    LANGUAGE_MODEL_PREFIX,
    MultiModalCausalLM,
    merge_multimodal_embeddings,
)
from .rotary_embedding import MRotaryEmbedding


class Qwen3VLTextModel(CausalLM):
    """Qwen3 decoder stack with mrope and DeepStack feature injection."""

    qkv_bias = False
    use_qk_norm = True
    rotary_class = MRotaryEmbedding

    def _after_layer(
        self,
        hidden_states: torch.Tensor,
        layer_index: int,
        layer_context: dict[str, Any],
    ) -> torch.Tensor:
        """Add the DeepStack feature for this layer at the vision token positions."""
        embeds = layer_context.get("deepstack_visual_embeds")
        if embeds is None or layer_index >= len(embeds):
            return hidden_states

        mask = layer_context["visual_pos_mask"].to(hidden_states.device)
        visual = embeds[layer_index].to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states[mask] = hidden_states[mask] + visual
        return hidden_states


class Qwen3VLForCausalLM(MultiModalCausalLM):
    """Qwen3-VL for image/video conditioned generation.

    Args:
        config: Parsed configuration wrapping a HuggingFace ``Qwen3VLConfig``.
    """

    weight_prefixes = (
        ("model.language_model.", LANGUAGE_MODEL_PREFIX),
        ("model.visual.", "vision_tower."),
        ("", LANGUAGE_MODEL_PREFIX),
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hf_config = config.hf_config

        self.vision_tower = Qwen3VLVisionModel(hf_config.vision_config)
        self.language_model = Qwen3VLTextModel(config)

        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id

        # Populated by encode_vision and consumed by forward in the same step.
        self._deepstack_embeds: list[torch.Tensor] | None = None

    @property
    def placeholder_token_ids(self) -> tuple[int, ...]:
        return (self.image_token_id, self.video_token_id)

    def encode_vision(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode images and/or videos into language-model embedding space.

        Args:
            pixel_values: Flattened image patches ``[num_patches, patch_features]``.
            image_grid_thw: ``[num_images, 3]`` temporal/height/width patch grid.
            pixel_values_videos: Flattened video patches.
            video_grid_thw: ``[num_videos, 3]`` patch grid for videos.

        Returns:
            ``[num_vision_tokens, text_hidden_size]`` in prompt order (images first,
            then videos, matching how the processor lays out the placeholders).
        """
        target_dtype = next(self.vision_tower.parameters()).dtype
        target_device = next(self.vision_tower.parameters()).device

        embeds: list[torch.Tensor] = []
        deepstack_parts: list[list[torch.Tensor]] = []

        for patches, grid in (
            (pixel_values, image_grid_thw),
            (pixel_values_videos, video_grid_thw),
        ):
            if patches is None:
                continue
            patches = patches.to(device=target_device, dtype=target_dtype)
            grid = grid.to(device=target_device)
            # transformers < 5 returned a plain (merged, deepstack) tuple; newer
            # versions wrap the same two tensors in a ModelOutput.
            out = self.vision_tower(patches, grid_thw=grid)
            if isinstance(out, Mapping):
                merged, deepstack = out["pooler_output"], out["deepstack_features"]
            else:
                merged, deepstack = out
            embeds.append(merged)
            deepstack_parts.append(deepstack)

        if not embeds:
            raise ValueError("encode_vision called without any pixel values")

        # Concatenate images and videos per DeepStack level so the level count stays
        # equal to len(deepstack_visual_indexes).
        self._deepstack_embeds = [
            torch.cat(level, dim=0) if len(level) > 1 else level[0]
            for level in zip(*deepstack_parts, strict=False)
        ]
        return torch.cat(embeds, dim=0) if len(embeds) > 1 else embeds[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        multi_modal_inputs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run one prefill or decode step, wiring DeepStack into the prefill pass."""
        inputs_embeds = None
        layer_context = None
        is_prefill = input_ids.shape[1] > 1

        if is_prefill and multi_modal_inputs:
            self._deepstack_embeds = None
            vision_embeds = self.encode_vision(**multi_modal_inputs)
            inputs_embeds = self.get_input_embeddings(input_ids)

            visual_pos_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in self.placeholder_token_ids:
                visual_pos_mask |= input_ids == token_id

            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, vision_embeds, self.placeholder_token_ids
            )
            if self._deepstack_embeds is not None:
                layer_context = {
                    "visual_pos_mask": visual_pos_mask,
                    "deepstack_visual_embeds": self._deepstack_embeds,
                }

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            atten_info=atten_info,
            inputs_embeds=inputs_embeds,
            layer_context=layer_context,
        )
