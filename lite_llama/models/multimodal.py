"""Shared scaffolding for vision-language models.

Design note — why merging is a plain scatter:

The HuggingFace processors for both LLaVA and Qwen3-VL already expand the single
``<image>`` marker in the prompt into exactly as many placeholder tokens as the
vision tower will emit patches (576 for LLaVA-1.5 at 336x336). By reusing those
processors, ``input_ids`` arrives at the model with its final length, so:

* merging vision embeddings is a masked assignment, not a sequence-expanding
  rewrite (the old ``merge_input_ids_with_image_features`` built new tensors and
  recomputed positions);
* ``position_ids`` are a plain ``arange`` over the already-correct length;
* KV-cache reservation needs no vision-specific arithmetic, because the prompt
  length the engine sees already accounts for every vision token.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn as nn

from .base import CausalLM


def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: torch.Tensor,
    placeholder_token_ids: int | tuple[int, ...],
) -> torch.Tensor:
    """Scatter vision embeddings onto the placeholder positions of a text sequence.

    Args:
        input_ids: ``[batch, seq_len]`` token ids containing the placeholders.
        inputs_embeds: ``[batch, seq_len, hidden]`` text embeddings, updated in place.
        multimodal_embeddings: ``[num_vision_tokens, hidden]`` (or any shape that
            flattens to it) vision embeddings in prompt order.
        placeholder_token_ids: Token id(s) marking positions to overwrite.

    Returns:
        ``inputs_embeds`` with the placeholder positions replaced.

    Raises:
        ValueError: If the number of vision embeddings does not match the number of
            placeholders. Silently padding or truncating here hides upstream bugs
            (a wrong ``patch_size``, or a processor/model config mismatch), so the
            mismatch is surfaced instead.
    """
    if isinstance(placeholder_token_ids, int):
        placeholder_token_ids = (placeholder_token_ids,)

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in placeholder_token_ids:
        mask |= input_ids == token_id

    num_placeholders = int(mask.sum())
    flat_vision = multimodal_embeddings.reshape(-1, inputs_embeds.shape[-1])
    if flat_vision.shape[0] != num_placeholders:
        raise ValueError(
            f"vision embedding count ({flat_vision.shape[0]}) does not match the "
            f"number of placeholder tokens ({num_placeholders}); check that the "
            "processor and the model config agree on patch_size and the vision "
            "feature selection strategy"
        )

    inputs_embeds[mask.to(inputs_embeds.device)] = flat_vision.to(inputs_embeds.dtype)
    return inputs_embeds


class MultiModalCausalLM(nn.Module):
    """Vision encoder + language model with placeholder-based embedding merge.

    Subclasses build ``self.language_model`` (a :class:`CausalLM`) plus whatever
    vision modules they need, and implement :meth:`encode_vision`. The prefill /
    decode split and the merge itself are handled here.

    Class attributes:
        placeholder_token_attrs: Config attribute names holding the placeholder
            token ids (for example ``("image_token_index",)``).
    """

    placeholder_token_attrs: ClassVar[tuple[str, ...]] = ()

    language_model: CausalLM

    @abstractmethod
    def encode_vision(self, **multi_modal_inputs: Any) -> torch.Tensor:
        """Encode raw vision inputs into language-model embedding space.

        Returns:
            ``[num_vision_tokens, text_hidden_size]`` in prompt order.
        """

    @property
    @abstractmethod
    def placeholder_token_ids(self) -> tuple[int, ...]:
        """Token ids that vision embeddings are scattered onto."""

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        multi_modal_inputs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run one prefill or decode step.

        Args:
            input_ids: ``[batch, seq_len]``; ``seq_len == 1`` selects the decode path.
            position_ids: ``[batch, seq_len]`` absolute positions.
            atten_info: KV-cache bookkeeping for this step.
            multi_modal_inputs: Processor outputs (``pixel_values`` and friends).
                Only consumed during prefill; ignored while decoding because the
                vision tokens are already in the KV cache.
        """
        inputs_embeds = None
        is_prefill = input_ids.shape[1] > 1
        if is_prefill and multi_modal_inputs:
            vision_embeds = self.encode_vision(**multi_modal_inputs)
            inputs_embeds = self.get_input_embeddings(input_ids)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, vision_embeds, self.placeholder_token_ids
            )

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            atten_info=atten_info,
            inputs_embeds=inputs_embeds,
        )
