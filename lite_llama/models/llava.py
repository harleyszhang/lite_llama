"""LLaVA-1.5: CLIP vision tower + 2-layer MLP projector + LLaMA language model.

The vision tower is HuggingFace's ``CLIPVisionModel`` (a faithful ViT is not what
this project is about), while the language model is lite_llama's own Triton-kernel
:class:`~lite_llama.models.llama.LlamaModel`.

Checkpoint key layout::

    vision_tower.vision_model.*        – CLIPVisionModel parameters (HF names)
    multi_modal_projector.linear_{1,2}.{weight,bias}
    language_model.*                   – LlamaModel parameters (lite_llama names)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
from transformers import LlavaConfig as HFLlavaConfig

from .llama import LlamaModel
from .model_config import LlamaConfig
from .multimodal import MultiModalCausalLM


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
        config: A HuggingFace :class:`~transformers.LlavaConfig`. ``max_seq_len`` may
            be attached to it by the executor to bound the KV cache.
    """

    def __init__(self, config: HFLlavaConfig) -> None:
        super().__init__()
        self.config = config

        self.select_layer = config.vision_feature_layer
        self.select_feature = config.vision_feature_select_strategy
        self.image_token_index = config.image_token_index

        self.vision_tower = CLIPVisionModel(config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
        )

        self.text_config = LlamaConfig.from_dict(
            config.text_config.to_dict(),
            max_seq_len=getattr(config, "max_seq_len", 2048),
        )
        self.language_model = LlamaModel(self.text_config)

    @property
    def placeholder_token_ids(self) -> tuple[int, ...]:
        return (self.image_token_index,)

    def remap_checkpoint_keys(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reconcile the vision tower's key layout with the installed transformers.

        ``CLIPVisionModel`` nests its encoder under a ``vision_model`` submodule in
        transformers 4.x but exposes it directly in 5.x. A checkpoint converted
        under one major version therefore fails ``load_state_dict`` under the other
        with hundreds of missing/unexpected ``vision_tower.*`` keys. Rather than
        pinning transformers, adapt the checkpoint to whatever layout the live
        module tree actually has.
        """
        expects_nested = any(
            key.startswith("vision_tower.vision_model.") for key in self.state_dict()
        )
        has_nested = any(key.startswith("vision_tower.vision_model.") for key in state_dict)
        if expects_nested == has_nested:
            return state_dict

        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if expects_nested and key.startswith("vision_tower."):
                # 5.x-style checkpoint, 4.x-style module tree: insert vision_model.
                suffix = key[len("vision_tower.") :]
                remapped[f"vision_tower.vision_model.{suffix}"] = value
            elif has_nested and key.startswith("vision_tower.vision_model."):
                # 4.x-style checkpoint, 5.x-style module tree: drop vision_model.
                suffix = key[len("vision_tower.vision_model.") :]
                remapped[f"vision_tower.{suffix}"] = value
            else:
                remapped[key] = value
        return remapped

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
        # `vision_feature_layer` is negative and indexes the hidden-state tuple,
        # e.g. -2 picks the penultimate layer as LLaVA-1.5 does.
        hidden_states = outputs.hidden_states[self.select_layer]
        hidden_states = self._select_image_features(hidden_states)

        image_features = self.multi_modal_projector(hidden_states)
        if not torch.isfinite(image_features).all():
            raise RuntimeError("vision tower produced non-finite image features")
        return image_features
