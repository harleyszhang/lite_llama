"""Capability interfaces a model opts into, plus their shared helpers.

Today just :class:`MultiModalCausalLM`, inherited by a model to accept
``multi_modal_inputs`` (mirrors vLLM's ``interfaces.py``). Because the HF
processors already expand ``<image>`` into the exact number of placeholder tokens
the vision tower emits, merging vision embeddings is a plain masked scatter
(:func:`merge_multimodal_embeddings`) — no sequence-expanding rewrite, positions
stay a plain ``arange``, and KV reservation needs no vision-specific arithmetic.

Usage:
    class MyVLM(MultiModalCausalLM): ...
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn

from ..modules.quantization import adapt_int4_checkpoint
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
        """Fill vision tower, projector and language model from one checkpoint stream.

        The sharder is passed through so tensor parallelism works for the language
        model; vision-tower parameters do not match any ``_SHARD_DIM`` entry and
        are therefore returned unchanged (replicated across ranks).
        """
        tied = (
            {
                f"{LANGUAGE_MODEL_PREFIX}lm_head.weight": (
                    f"{LANGUAGE_MODEL_PREFIX}embed_tokens.weight"
                )
            }
            if self.language_model.config.tie_word_embeddings
            else None
        )
        quant = self.language_model.quant
        if quant is not None and quant.is_int4:
            # Same canonical-layout rewrite as CausalLM.load_weights.
            checkpoint = adapt_int4_checkpoint(checkpoint, quant)
        weights.load_weights(
            self,
            checkpoint,
            self.translate_weight_key,
            tied=tied,
            shard=weights.tp_shard,
        )

    @torch.no_grad()
    def quantize_(self, quant) -> None:
        """Delegate to the language model; vision modules stay in their native dtype."""
        self.language_model.quantize_(quant)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        multi_modal_inputs: dict[str, Any] | None = None,
        logits_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one prefill or decode step.

        Args:
            input_ids: ``[batch, seq_len]``; ``seq_len == 1`` selects the decode path.
            position_ids: ``[batch, seq_len]`` absolute positions.
            atten_info: KV-cache bookkeeping for this step; ``atten_info.is_prefill``
                decides prefill vs decode, so a single-token prompt is still a
                prefill.
            multi_modal_inputs: Processor outputs (``pixel_values`` and friends).
                Only consumed during prefill; ignored while decoding because the
                vision tokens are already in the KV cache.
            logits_positions: Optional ``[batch]`` positions whose logits the
                caller wants; forwarded to the language model so the lm_head
                GEMM runs only on the requested rows.
        """
        inputs_embeds = None
        if atten_info.is_prefill and multi_modal_inputs:
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
            logits_positions=logits_positions,
        )
