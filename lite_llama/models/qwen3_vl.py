"""Qwen3-VL: Vision-Language model combining Qwen3VLVisionModel with Qwen3Model.

Architecture:
    - Vision encoder: Qwen3VLVisionModel (from transformers, ViT + PatchMerger)
    - Language model: Qwen3Model (custom Triton-kernel implementation)
    - Merge strategy: replace image placeholder tokens with vision embeddings
"""
from typing import Optional

import torch
import torch.nn as nn

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig as HFQwen3VLConfig

from .qwen3 import Qwen3Model
from .model_config import Qwen3Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Qwen3VLForCausalLM(nn.Module):
    """Qwen3-VL model for multimodal inference (vision + language).

    State-dict key layout (after weight conversion):
        vision_tower.*          – Qwen3VLVisionModel parameters (HF format)
        language_model.*        – Qwen3Model parameters (lite_llama format)
        lm_head_weight          – (vocab_size, hidden_size)
    """

    def __init__(self, config: HFQwen3VLConfig):
        super().__init__()
        self.config = config
        self.device = "cuda"

        # Vision encoder (use HF implementation – complex ViT + merger)
        self.vision_tower = Qwen3VLVisionModel(config.vision_config)

        # Language model (custom Qwen3 with Triton kernels)
        text_cfg_dict = config.text_config.to_dict()
        self.qwen3_config = Qwen3Config.from_dict(text_cfg_dict)
        self.language_model = Qwen3Model(self.qwen3_config)

        # LM head (shared or separate)
        self.lm_head_weight = nn.Parameter(
            torch.rand(
                self.qwen3_config.vocab_size,
                self.qwen3_config.hidden_size,
                dtype=torch.float16,
            )
        )

        # Token IDs for vision placeholders
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        self.vision_end_token_id = config.vision_end_token_id

    def vision_encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images/videos through the vision tower.

        Args:
            pixel_values: (seq_len, channels * temporal_patch * patch * patch)
            grid_thw: (num_images_or_videos, 3) – temporal, height, width grids

        Returns:
            image_embeds: (num_merged_patches, text_hidden_size)
        """
        pixel_values = pixel_values.to(dtype=torch.float16, device=self.device)
        grid_thw = grid_thw.to(device=self.device)

        # Qwen3VLVisionModel returns merged hidden states
        vision_outputs = self.vision_tower(
            hidden_states=pixel_values,
            grid_thw=grid_thw,
        )
        # vision_outputs is a tuple; first element is hidden_states
        if isinstance(vision_outputs, tuple):
            image_embeds = vision_outputs[0]
        else:
            image_embeds = vision_outputs.last_hidden_state

        return image_embeds

    def _merge_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Replace image/video placeholder positions with vision embeddings."""
        # Find positions of image and video tokens
        image_mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
        image_mask = image_mask.to(inputs_embeds.device)

        # Flatten and assign
        num_vision_tokens = image_mask.sum().item()
        if num_vision_tokens > 0:
            vision_flat = image_embeds.view(-1, image_embeds.shape[-1])
            if vision_flat.shape[0] >= num_vision_tokens:
                inputs_embeds[image_mask] = vision_flat[:num_vision_tokens].to(inputs_embeds.dtype)
            else:
                logger.warning(
                    f"Vision tokens ({num_vision_tokens}) > vision embeddings ({vision_flat.shape[0]}), padding with zeros"
                )
                inputs_embeds[image_mask] = torch.nn.functional.pad(
                    vision_flat, (0, 0, 0, num_vision_tokens - vision_flat.shape[0])
                ).to(inputs_embeds.dtype)

        return inputs_embeds

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        """Forward pass supporting both text-only and multimodal inputs.

        For prefill with images: pixel_values and grid_thw must be provided.
        For decode (seq_len=1): only input_ids are needed.
        """
        input_ids = input_ids.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        batch_size, seq_len = input_ids.shape

        if inputs_embeds is None:
            if seq_len > 1 and pixel_values is not None:
                # Prefill with vision: encode images and merge
                image_embeds = self.vision_encode(pixel_values, grid_thw)
                inputs_embeds = self.get_input_embeddings(input_ids)
                inputs_embeds = self._merge_vision_embeddings(
                    input_ids, inputs_embeds, image_embeds
                )
            else:
                # Text-only or decode phase
                inputs_embeds = None  # Let language_model handle embedding

        # Use language model's lm_head_weight for output projection
        # Temporarily set it if the language model uses its own
        hidden_states = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            atten_info=atten_info,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
