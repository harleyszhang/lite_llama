"""The same layer as HuggingFace builds it, for the harness to diff against.

:class:`HFLayerReference` instantiates the checkpoint's own decoder layer
via :func:`hf_decoder_layer_class`, loads the matching weights, and runs
the same inputs so a diff is meaningful.

Usage:
    ref = HFLayerReference(config, layer_index)
"""

from __future__ import annotations

import copy
import importlib

import torch
import torch.nn as nn

from ...models.config import ModelConfig
from .single_layer import LayerReference


def hf_decoder_layer_class(hf_config: object) -> type[nn.Module]:
    """The ``transformers`` decoder-layer class for ``hf_config``.

    Resolved through the auto mapping and the model class's own ``_no_split_modules``,
    which names the one module transformers refuses to shard — that is, the decoder
    layer. Going through the declared name rather than a hard-coded ``f"{Family}
    DecoderLayer"`` string means a family that renames its layer still resolves.

    Raises:
        LookupError: If transformers has no causal-LM model for this config, or its
            model class does not declare a single no-split module.
    """
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    try:
        model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(hf_config)]
    except KeyError:
        raise LookupError(
            f"transformers has no causal-LM implementation for {type(hf_config).__name__}"
        ) from None
    names = getattr(model_cls, "_no_split_modules", None) or ()
    if len(names) != 1:
        raise LookupError(
            f"{model_cls.__name__} declares _no_split_modules={list(names)}; "
            "cannot tell which one is the decoder layer"
        )
    return getattr(importlib.import_module(model_cls.__module__), names[0])


class HFLayerReference(LayerReference):
    """One ``transformers`` decoder layer, run against the harness's two shapes.

    The layer is built at its real index, not at zero: models with mixed layer types
    (sliding vs full attention, dense vs routed MLP) decide which one this is from the
    index, and a reference for the wrong kind of layer is worse than no reference.

    Weights are whatever ``transformers`` initialises — the harness mirrors them into
    its own layer before comparing, so no checkpoint is needed to check that the two
    implementations agree. Build order is fp32-then-cast, matching how the published
    checkpoints were produced rather than initialising in bf16.

    Args:
        config: The parsed lite_llama config; its ``text_config`` drives the layer.
        layer_index: Index within the decoder stack.
        device: Where the layer runs.
        dtype: Compute dtype; defaults to the checkpoint's.
        seed: Seed for the initialisation, so two runs mirror identical weights.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_index: int,
        *,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        seed: int = 0,
    ) -> None:
        # A copy, because pinning the attention implementation must not reach back into
        # the config the harness built its own layer from.
        hf_config = copy.deepcopy(config.text_config)
        hf_config._attn_implementation = "eager"

        layer_cls = hf_decoder_layer_class(hf_config)
        torch.manual_seed(seed)
        self.layer = layer_cls(hf_config, layer_index)
        self.dtype = dtype or config.dtype
        self.layer = self.layer.to(device=device, dtype=self.dtype).eval()
        self.device = device
        self.layer_index = layer_index
        self.name = f"transformers {layer_cls.__name__}"
        self._cache: object | None = None
        self._hf_config = hf_config
        # The Deepseek family's layer rotates with a polar ``freqs_cis`` tensor
        # rather than the (cos, sin) tuple every other family takes (its
        # ``apply_rotary_emb`` is the original view_as_complex spelling).
        self._freqs_cis = str(getattr(hf_config, "model_type", "")).startswith("deepseek")

    def state_dict(self) -> dict[str, torch.Tensor]:
        """This layer's parameters, under HuggingFace's own names.

        Parameters only, not ``nn.Module.state_dict()``: a buffer that happens to be
        persistent has no counterpart in the lite_llama layer, and the strict coverage
        check would reject it as an unknown key.
        """
        return dict(self.layer.named_parameters())

    def prefill(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Causal forward over the whole prompt, from an empty cache."""
        from transformers.cache_utils import DynamicCache

        # A config-shaped cache, so the per-layer slot this layer's index names exists.
        self._cache = DynamicCache(config=self._hf_config)
        seq_len = hidden_states.shape[1]
        mask = self._causal_mask(hidden_states.shape[0], seq_len)
        return self._forward(hidden_states, position_embeddings, mask)

    def decode(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """One token over the cached history; no mask, because nothing is masked.

        A single query attends to every key before it, which is exactly what an absent
        mask means in ``eager_attention_forward``.

        Raises:
            RuntimeError: If :meth:`prefill` has not run, since there would be no
                history for the token to attend to.
        """
        if self._cache is None:
            raise RuntimeError("call prefill() before decode(): the cache is empty")
        return self._forward(hidden_states, position_embeddings, None)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._freqs_cis:
            # cos/sin arrive doubled-width (first half == second half); the
            # complex form wants one entry per frequency pair. Both halves of
            # the tables already carry the YaRN attention scaling.
            cos, sin = position_embeddings
            half = cos.shape[-1] // 2
            position_embeddings = torch.complex(
                cos[..., :half].float(), sin[..., :half].float()
            )
        with torch.no_grad():
            out = self.layer(
                hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
                past_key_values=self._cache,
                use_cache=True,
            )
        # transformers 5.x returns the tensor; earlier releases wrapped it in a tuple.
        return out[0] if isinstance(out, tuple) else out

    def _causal_mask(self, batch: int, seq_len: int) -> torch.Tensor:
        """``[batch, 1, seq_len, seq_len]`` additive mask, zero below the diagonal.

        ``finfo.min`` rather than ``-inf``: the mask is added to attention logits in the
        compute dtype, and ``-inf + finite`` in a row that is entirely masked would make
        the softmax produce NaN instead of a defined (if meaningless) distribution.
        """
        blocked = torch.full(
            (seq_len, seq_len),
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
            device=self.device,
        ).triu(1)
        return blocked.expand(batch, 1, seq_len, seq_len)
