"""Capability interfaces a model class opts into, plus their shared helpers.

Today there is exactly one: :class:`MultiModalCausalLM`, which a model declares by
inheriting it in order to accept ``multi_modal_inputs``. Mirrors vLLM's
``model_executor/models/interfaces.py``, where ``SupportsMultiModal`` and friends
sit apart from the concrete architectures.

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
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn

from . import weights
from .base import CausalLM

#: lite_llama parameter prefix of the decoder stack inside a vision-language model.
#: Also the marker in :attr:`MultiModalCausalLM.weight_prefixes` that says "hand the
#: rest of this key to the text model's own translation".
LANGUAGE_MODEL_PREFIX = "language_model."


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
        weight_prefixes: ``(checkpoint prefix, lite_llama prefix)`` pairs covering the
            whole checkpoint, tried in order. A pair targeting
            :data:`LANGUAGE_MODEL_PREFIX` hands the remainder to the text model's own
            key translation; every other pair is a plain rename, because the vision
            tower and projector *are* HF modules and keep HF parameter names.
    """

    weight_prefixes: ClassVar[tuple[tuple[str, str], ...]] = ()

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

    # ---- weight loading --------------------------------------------------- #
    def translate_weight_key(self, key: str) -> weights.Target:
        """Route a checkpoint key to the right submodule via :attr:`weight_prefixes`."""
        for hf_prefix, lite_prefix in self.weight_prefixes:
            rest = weights.strip_prefix(key, hf_prefix)
            if rest is None:
                continue
            if lite_prefix != LANGUAGE_MODEL_PREFIX:
                return lite_prefix + rest, weights.whole
            target = self.language_model.translate_weight_key(rest)
            if target is None:
                return None
            name, destination = target
            return lite_prefix + name, destination
        return None

    def load_weights(self, checkpoint: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Fill vision tower, projector and language model from one checkpoint stream."""
        tied = (
            {
                f"{LANGUAGE_MODEL_PREFIX}lm_head_weight": (
                    f"{LANGUAGE_MODEL_PREFIX}embed_tokens.weight"
                )
            }
            if self.language_model.config.tie_word_embeddings
            else None
        )
        weights.load_weights(self, checkpoint, self.translate_weight_key, tied=tied)

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
