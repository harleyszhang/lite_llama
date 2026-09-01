"""LLaVA-1.5: CLIP vision tower + 2-layer MLP projector + LLaMA language model.

:class:`LlavaLlama` implements :class:`MultiModalCausalLM`: image features
pass through the projector and splice into the token embeddings at the
image-token placeholder positions.

Usage:
    model = LlavaLlama(config)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

from .config import ModelConfig
from .interfaces import LANGUAGE_MODEL_PREFIX, MultiModalCausalLM
from .llama import LlamaModel


class LlavaMultiModalProjector(nn.Module):
    """Projects CLIP patch features into the language model's embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)
        if projector_hidden_act != "gelu":
            raise ValueError(
                f"unsupported projector_hidden_act {projector_hidden_act!r}; "
                "only 'gelu' is implemented"
            )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(image_features)))


class LlavaLlama(MultiModalCausalLM):
    """LLaVA-1.5 for single-image (or multi-image) conditioned generation.

    Args:
        config: Parsed configuration wrapping a HuggingFace ``LlavaConfig``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hf_config = config.hf_config

        self.select_layer = hf_config.vision_feature_layer
        self.select_feature = hf_config.vision_feature_select_strategy
        self.image_token_index = hf_config.image_token_index

        self.vision_tower = CLIPVisionModel(hf_config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=hf_config.vision_config.hidden_size,
            text_hidden_size=config.hidden_size,
            projector_hidden_act=hf_config.projector_hidden_act,
        )
        self.language_model = LlamaModel(config)

        nested = any(key.startswith("vision_model.") for key in self.vision_tower.state_dict())
        vision = "vision_tower.vision_model." if nested else "vision_tower."

        self.weight_prefixes = (
            (LANGUAGE_MODEL_PREFIX, LANGUAGE_MODEL_PREFIX),
            ("vision_tower.vision_model.", vision),
            ("vision_tower.", vision),
            ("multi_modal_projector.", "multi_modal_projector."),
            ("model.language_model.", LANGUAGE_MODEL_PREFIX),
            ("model.vision_tower.", vision),
            ("model.multi_modal_projector.", "multi_modal_projector."),
            ("", LANGUAGE_MODEL_PREFIX),
        )

    @property
    def placeholder_token_ids(self) -> tuple[int, ...]:
        return (self.image_token_index,)

    def _select_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Apply the configured patch-selection strategy.

        ``default`` drops the leading CLS token (576 of 577 patches survive for
        LLaVA-1.5), ``full`` keeps everything. The processor expands the ``<image>``
        placeholder using the same rule, so the two must stay in sync.
        """
        if self.select_feature in ("default", "patch"):
            return image_features[:, 1:].contiguous()
        if self.select_feature == "full":
            return image_features
        raise ValueError(f"unexpected vision_feature_select_strategy: {self.select_feature}")

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into projected patch embeddings.

        Args:
            pixel_values: ``[num_images, 3, H, W]`` preprocessed pixels.

        Returns:
            ``[num_images * num_patches, text_hidden_size]``.
        """
        target_dtype = self.multi_modal_projector.linear_1.weight.dtype
        target_device = self.multi_modal_projector.linear_1.weight.device
        pixel_values = pixel_values.to(device=target_device, dtype=target_dtype)

        outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states[self.select_layer]
        hidden_states = self._select_image_features(hidden_states)

        image_features = self.multi_modal_projector(hidden_states)
        if not torch.isfinite(image_features).all():
            raise RuntimeError("vision tower produced non-finite image features")
        return image_features
