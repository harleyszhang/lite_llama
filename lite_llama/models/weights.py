"""HuggingFace checkpoint keys -> lite_llama parameters.

:func:`translate_text_key` maps one HF key to a target parameter (fusing
gate/up or q/k/v where the skeleton merged them); :func:`load_weights`
streams ``(key, tensor)`` pairs through the translator and verifies every
parameter was covered.

Usage:
    translate = partial(translate_text_key, packed=True)
    load_weights(model, weights, translate, tied=False)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn

#: Which block of a packed parameter one checkpoint tensor fills: ``0/1/2`` for the
#: fused ``[q | k | v]``, ``0/1`` for the fused gate/up pair, ``(expert, projection)``
#: for the stacked MoE experts, ``None`` when the parameter is not packed.
ShardId = Any

#: ``(parameter name, shard id)``, or ``None`` for a key the model ignores.
Target = tuple[str, ShardId] | None

#: Maps a checkpoint key to its :data:`Target`.
Translator = Callable[[str], Target]


def default_weight_loader(
    param: torch.Tensor, loaded: torch.Tensor, shard_id: ShardId = None
) -> torch.Tensor:
    """Fill a parameter that carries no loader of its own; return the view written.

    The fallback for parameters no layer claimed: norms, the MoE router and the
    whole vision tower are replicated on every rank and never packed, so the whole
    rule is "the shapes must match, then copy".
    """
    if param.shape != loaded.shape:
        raise ValueError(
            f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
            f"parameter of shape {tuple(param.shape)}"
        )
    param.data.copy_(loaded)
    return param.data


# --------------------------------------------------------------------------- #
# Text-model key translation
# --------------------------------------------------------------------------- #

#: HF module paths whose ``weight``/``bias`` leaf was folded into the parent's
#: parameter name. Matched as a suffix of the checkpoint key's module path, so
#: ``layers.7.self_attn.q_norm.weight`` becomes ``layers.7.self_attn.q_norm_weight``.
#: The projections are absent because they are real submodules
#: (:class:`~lite_llama.modules.linear.LinearBase`) whose parameter names already
#: match HF's.
_FLATTENED: tuple[str, ...] = (
    "self_attn.q_norm",
    "self_attn.k_norm",
    # MLA: the latent/query layernorms are bare parameters on the attention
    # module, folded the same way. ``q_norm`` above stays unmatched here because
    # endswith is exact — ``q_a_layernorm`` is not ``q_norm``.
    "self_attn.kv_a_layernorm",
    "self_attn.q_a_layernorm",
    "input_layernorm",
    "post_attention_layernorm",
    # MoE router. The dense SwiGLU gate is ``mlp.gate_proj`` and therefore does
    # not match this suffix.
    "mlp.gate",
)

#: Keys outside the decoder stack, matched exactly rather than by suffix.
#: ``lm_head.weight`` is absent because :class:`~lite_llama.modules.vocab_parallel.
#: ParallelLMHead` is a real submodule whose parameter is already called that.
_TOP_LEVEL: dict[str, str] = {
    "norm.weight": "norm_weight",
}

#: ``layers.N.mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale_inv}`` in an
#: MoE checkpoint. The scales of an fp8 checkpoint are stacked exactly like the
#: weights they belong to, one coarse row per 128 fine ones.
_EXPERT_KEY = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate|up|down)_proj"
    r"\.(?P<leaf>weight|weight_scale_inv)$"
)

#: gate/up/down as projection ids within one expert's slice: gate and up are the
#: two halves of the fused ``gate_up_proj`` (0 and 1), down is its own tensor (2).
_EXPERT_PROJ_ID = {"gate": 0, "up": 1, "down": 2}


def translate_text_key(key: str, packed: Mapping[str, tuple[str, ...]]) -> Target:
    """Map one decoder-stack checkpoint key onto ``(parameter, shard id)``.

    Args:
        key: Checkpoint key with the model's own prefix already stripped, e.g.
            ``layers.3.self_attn.v_proj.weight``.
        packed: The model's ``packed_modules_mapping``: ``{fused module path:
            (checkpoint module paths, in block order)}``. A key under one of the
            source modules maps onto the fused parameter with the source's index
            as its shard id.

    Returns:
        The parameter the tensor belongs to and which block of it the tensor
        fills. Keys that already match a lite_llama parameter name
        (``embed_tokens.weight``, ``layers.N.mlp.down_proj.weight``) map to
        themselves with no shard id.
    """
    if key in _TOP_LEVEL:
        return _TOP_LEVEL[key], None

    experts = _EXPERT_KEY.match(key)
    if experts is not None:
        prefix, index = experts["prefix"], int(experts["expert"])
        # The scale grid is stacked alongside the weight under its own name,
        # because a ParameterDict entry cannot carry a second leaf.
        suffix = "_scale_inv" if experts["leaf"].endswith("_scale_inv") else ""
        proj = _EXPERT_PROJ_ID[experts["proj"]]
        name = f"{prefix}.gate_up_proj{suffix}" if proj < 2 else f"{prefix}.down_proj{suffix}"
        return name, (index, proj)

    module, _, leaf = key.rpartition(".")
    for fused, sources in packed.items():
        for shard_id, source in enumerate(sources):
            if module.endswith(source):
                return f"{module[: -len(source)]}{fused}.{leaf}", shard_id
    for suffix in _FLATTENED:
        if module.endswith(suffix):
            return f"{module}_{leaf}", None
    return key, None


def strip_prefix(key: str, prefix: str) -> str | None:
    """Return ``key`` without ``prefix``, or ``None`` when it does not match."""
    return key[len(prefix) :] if key.startswith(prefix) else None


# --------------------------------------------------------------------------- #
# The copy loop
# --------------------------------------------------------------------------- #


def load_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    translate: Translator,
    tied: Mapping[str, str] | None = None,
) -> None:
    """Copy a HuggingFace checkpoint into ``model``'s already-allocated parameters.

    Args:
        model: Model whose parameters have real storage (see
            :func:`lite_llama.executor.loader.materialise_parameters`).
        weights: ``(checkpoint key, tensor)`` pairs, in any order.
        translate: Maps a checkpoint key to ``(parameter name, shard id)``, or
            ``None`` to skip it (HF bookkeeping tensors, keys belonging to
            another submodule). The shard id is handed to the parameter's
            ``weight_loader``, which owns both the destination view and the
            tensor-parallel narrow; parameters without one (norms, the MoE
            router, vision towers) fall back to :func:`default_weight_loader`.
        tied: ``{target parameter: source parameter}`` pairs to satisfy by *aliasing*
            when the checkpoint omits the target. Checkpoints with
            ``tie_word_embeddings: true`` ship no ``lm_head.weight`` at all, and
            lite_llama keeps it as its own parameter, so it is pointed at the
            embedding table rather than given a copy of it — which is both half the
            memory and the only way ``lm_head.weight is embed_tokens.weight`` can hold
            once the two are sharded. A checkpoint that *does* ship the target wins:
            the tie is a fallback, not an override.

    Raises:
        ValueError: If a parameter ends up unfilled, partially filled or written
            more than once, or if a key maps to a parameter that does not exist.
    """
    params = dict(model.named_parameters())
    filled: dict[str, int] = dict.fromkeys(params, 0)

    for key, tensor in weights:
        target = translate(key)
        if target is None:
            continue
        name, shard_id = target
        param = params.get(name)
        if param is None:
            raise ValueError(f"checkpoint key {key!r} maps to unknown parameter {name!r}")

        loader = getattr(param, "weight_loader", None) or default_weight_loader
        try:
            view = loader(param, tensor, shard_id)
        except ValueError as e:
            raise ValueError(f"checkpoint key {key!r} -> {name!r}: {e}") from None
        filled[name] += view.numel()

    for target_name, source_name in (tied or {}).items():
        if filled.get(target_name) == 0:
            _alias_parameter(model, target_name, params[source_name])
            filled[target_name] = params[target_name].numel()

    _verify_coverage(params, filled)


def _alias_parameter(model: nn.Module, dotted_name: str, source: nn.Parameter) -> None:
    """Rebind ``model.<dotted_name>`` to ``source``, freeing what was there.

    ``nn.Module`` deduplicates shared parameters, so after this the aliased name no
    longer appears in ``named_parameters()`` — which is the point: there is one tensor,
    and reporting it twice would double every memory total computed from that iterator.
    """
    path, _, leaf = dotted_name.rpartition(".")
    owner = model.get_submodule(path) if path else model
    setattr(owner, leaf, source)


def _verify_coverage(params: Mapping[str, nn.Parameter], filled: Mapping[str, int]) -> None:
    """Fail naming the offending parameters rather than with a bare count.

    Three distinguishable failures: nothing wrote a parameter (a rename rule stopped
    matching), something wrote only part of it (one block of a fused QKV arrived), or
    something wrote more than fits (two keys competing for the same destination).
    """
    missing = sorted(name for name, count in filled.items() if count == 0)
    mismatched = sorted(
        f"{name} ({count} of {params[name].numel()} elements)"
        for name, count in filled.items()
        if 0 < count != params[name].numel()
    )
    if not missing and not mismatched:
        return

    problems = []
    if missing:
        problems.append(f"{len(missing)} never written, e.g. {', '.join(missing[:5])}")
    if mismatched:
        problems.append(f"{len(mismatched)} partially written, e.g. {', '.join(mismatched[:5])}")
    raise ValueError("checkpoint does not cover every parameter — " + "; ".join(problems))
