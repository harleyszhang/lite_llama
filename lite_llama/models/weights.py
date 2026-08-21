"""HuggingFace checkpoint keys -> lite_llama parameters.

lite_llama consumes HF checkpoints as they ship: no offline conversion step, no
private file format. What it *does* need is a translation layer, because two
deliberate structural choices make its parameter tree differ from HF's:

* **Fused K/V.** ``k_proj`` and ``v_proj`` are concatenated along dim 0 into one
  ``kv_proj_weight`` so a decode step writes both halves of the KV cache with a
  single kernel launch.
* **Stacked MoE experts.** HF stores ``3 * num_experts`` matrices per MoE layer;
  lite_llama stacks them into three tensors so the expert FFN runs as two grouped
  GEMMs instead of a Python loop over experts.

Everything else is a naming difference: lite_llama holds projection weights as
bare ``nn.Parameter``s (``self_attn.q_proj_weight``) rather than ``nn.Linear``
submodules, which keeps the ``F.linear`` calls explicit in the model code, so the
checkpoint's ``self_attn.q_proj.weight`` loses one level of nesting.

The mapping is expressed as *destinations*: a checkpoint tensor names the
parameter it belongs to plus the view inside that parameter which it fills.
Whole-parameter loads use :func:`whole`; the fused parameters use one of the
shard selectors, so two (or ``2 * num_experts``) checkpoint tensors add up to one
parameter. :func:`load_weights` then verifies that every parameter was covered
exactly once, element for element — a mapping rule that silently misses a key
would otherwise leave a model that runs and returns nonsense.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

import torch
import torch.nn as nn

#: Selects the region of a parameter that one checkpoint tensor fills.
Destination = Callable[[torch.Tensor], torch.Tensor]

#: ``(parameter name, destination)``, or ``None`` for a key the model ignores.
Target = tuple[str, Destination] | None

#: Maps a checkpoint key to its :data:`Target`.
Translator = Callable[[str], Target]


def whole(param: torch.Tensor) -> torch.Tensor:
    """The whole parameter: the common case."""
    return param


def half(index: int) -> Destination:
    """Half ``index`` of a parameter fused along dim 0 (the K/V pair)."""

    def select(param: torch.Tensor) -> torch.Tensor:
        size = param.shape[0] // 2
        return param.narrow(0, index * size, size)

    return select


def expert(index: int) -> Destination:
    """The slice of a stacked ``[num_experts, ...]`` parameter owned by one expert."""

    def select(param: torch.Tensor) -> torch.Tensor:
        return param[index]

    return select


def expert_half(index: int, half_index: int) -> Destination:
    """Half ``half_index`` of expert ``index``'s slice of a stacked gate/up parameter."""

    def select(param: torch.Tensor) -> torch.Tensor:
        rows = param.shape[1] // 2
        return param[index].narrow(0, half_index * rows, rows)

    return select


# --------------------------------------------------------------------------- #
# Text-model key translation
# --------------------------------------------------------------------------- #

#: HF module paths whose ``weight``/``bias`` leaf was folded into the parent's
#: parameter name. Matched as a suffix of the checkpoint key's module path, so
#: ``layers.7.self_attn.q_proj.weight`` becomes ``layers.7.self_attn.q_proj_weight``.
_FLATTENED: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "input_layernorm",
    "post_attention_layernorm",
    # MoE router. The dense SwiGLU gate is ``mlp.gate_proj`` and therefore does
    # not match this suffix.
    "mlp.gate",
)

#: HF module path suffix -> which half of ``kv_proj_{weight,bias}`` it fills.
_FUSED_KV: dict[str, int] = {"self_attn.k_proj": 0, "self_attn.v_proj": 1}

#: Keys outside the decoder stack, matched exactly rather than by suffix.
_TOP_LEVEL: dict[str, str] = {
    "norm.weight": "norm_weight",
    "lm_head.weight": "lm_head_weight",
}

#: ``layers.N.mlp.experts.E.{gate,up,down}_proj.weight`` in an MoE checkpoint.
_EXPERT_KEY = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate|up|down)_proj\.weight$"
)


def translate_text_key(key: str) -> Target:
    """Map one decoder-stack checkpoint key onto ``(parameter, destination)``.

    Args:
        key: Checkpoint key with the model's own prefix already stripped, e.g.
            ``layers.3.self_attn.v_proj.weight``.

    Returns:
        The parameter the tensor belongs to and the view inside it to fill. Keys
        that already match a lite_llama parameter name (``embed_tokens.weight``,
        ``layers.N.mlp.up_proj.weight``) map to themselves.
    """
    if key in _TOP_LEVEL:
        return _TOP_LEVEL[key], whole

    experts = _EXPERT_KEY.match(key)
    if experts is not None:
        prefix, index = experts["prefix"], int(experts["expert"])
        if experts["proj"] == "down":
            return f"{prefix}.down_proj", expert(index)
        # gate and up are fused along dim 0 inside each expert's slice.
        return f"{prefix}.gate_up_proj", expert_half(index, 0 if experts["proj"] == "gate" else 1)

    module, _, leaf = key.rpartition(".")
    for suffix, index in _FUSED_KV.items():
        if module.endswith(suffix):
            return f"{module[: -len(suffix)]}self_attn.kv_proj_{leaf}", half(index)
    for suffix in _FLATTENED:
        if module.endswith(suffix):
            return f"{module}_{leaf}", whole
    return key, whole


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
        translate: Maps a checkpoint key to its destination, or ``None`` to skip
            it (HF bookkeeping tensors, keys belonging to another submodule).
        tied: ``{target parameter: source parameter}`` pairs to fill by copy when
            the checkpoint omits the target. Checkpoints with
            ``tie_word_embeddings: true`` ship no ``lm_head.weight`` at all, and
            lite_llama keeps it as its own parameter.

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
        name, destination = target
        param = params.get(name)
        if param is None:
            raise ValueError(f"checkpoint key {key!r} maps to unknown parameter {name!r}")

        view = destination(param.data)
        if view.shape != tensor.shape:
            raise ValueError(
                f"checkpoint key {key!r} has shape {tuple(tensor.shape)} but "
                f"{name!r} expects {tuple(view.shape)}"
            )
        view.copy_(tensor)
        filled[name] += view.numel()

    for target_name, source_name in (tied or {}).items():
        if filled.get(target_name) == 0:
            params[target_name].data.copy_(params[source_name].data)
            filled[target_name] = params[target_name].numel()

    _verify_coverage(params, filled)


def _verify_coverage(params: Mapping[str, nn.Parameter], filled: Mapping[str, int]) -> None:
    """Fail naming the offending parameters rather than with a bare count.

    Three distinguishable failures: nothing wrote a parameter (a rename rule stopped
    matching), something wrote only part of it (one half of a fused K/V arrived), or
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
