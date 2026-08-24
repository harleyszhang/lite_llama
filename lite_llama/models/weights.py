"""HuggingFace checkpoint keys -> lite_llama parameters.

lite_llama loads HF checkpoints as-is; only a key translation is needed because three
structural choices differ from HF: **fused QKV** (``q_proj``+``k_proj``+``v_proj``
concatenated into ``qkv_proj.weight``, one GEMM per attention block), **fused gate/up**
(the dense MLP's ``gate_proj``+``up_proj`` concatenated into ``gate_up_proj.weight`` so
the forward pass is one GEMM) and **stacked MoE experts** (``3*num_experts`` matrices
packed into three tensors for grouped-GEMM experts).
The rest is naming (bare ``nn.Parameter`` vs ``nn.Linear``). Rules are expressed as
*destinations* (parameter + the view to fill); :func:`load_weights` then verifies
every parameter is covered exactly once, so a missed key fails loudly, not silently.

A destination is handed the arriving tensor as well as the parameter, which is what
makes the unequal ``[q | k | v]`` blocks expressible: the parameter alone only fixes
``q = total - 2 * kv``, and the arriving tensor supplies the missing width. No head
geometry has to be threaded into the translator.

A second table, :data:`_SHARD_DIM`, says how each weight is cut for tensor
parallelism. It is written in terms of the *incoming* tensor rather than the
parameter, which is what makes one entry cover the weight and its quantisation
scales at once (a scale grid is the same matrix at a coarser resolution) and also
the fused/stacked parameters, whose own axes are shifted by the fusing — q, k and v
arrive separately, so ``qkv_proj``'s one entry splits each of them by its own head
count and a rank's blocks land next to each other rather than next to other ranks'.

Usage:
    load_weights(model, hf_weights_iterator(path), model.translate_weight_key)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

import torch
import torch.nn as nn

from ..distributed.parallel_state import get_tp_rank, get_tp_world_size

#: Selects the region of a parameter that one checkpoint tensor fills, given both the
#: parameter and that tensor.
Destination = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

#: ``(parameter name, destination)``, or ``None`` for a key the model ignores.
Target = tuple[str, Destination] | None

#: Maps a checkpoint key to its :data:`Target`.
Translator = Callable[[str], Target]

#: Narrows an incoming tensor to the slice this rank owns.
Sharder = Callable[[str, torch.Tensor], torch.Tensor]


def whole(param: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
    """The whole parameter: the common case."""
    return param


def half(index: int) -> Destination:
    """Half ``index`` of a parameter fused along dim 0 (the gate/up pair)."""

    def select(param: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        size = param.shape[0] // 2
        return param.narrow(0, index * size, size)

    return select


def qkv_segment(index: int) -> Destination:
    """Block ``index`` of the fused ``[q | k | v]`` parameter, ``0`` being q.

    The three blocks are not the same width — grouped-query attention gives q more
    heads — and the parameter alone does not say where the boundaries are: a total of
    ``T`` only fixes ``q = T - 2 * kv``. The arriving tensor closes the gap, since its
    own row count *is* the width of the block it fills. That keeps the head geometry in
    :class:`~lite_llama.modules.linear.QKVParallelLinear`, where it belongs, instead of
    duplicating it in this table.

    It works unchanged for the quantisation scale grids, whose rows are the same blocks
    counted in groups of ``group_n`` instead of channels.
    """

    def select(param: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        rows = incoming.shape[0]
        total = param.shape[0]
        offset = (0, total - 2 * rows, total - rows)[index]
        return param.narrow(0, offset, rows)

    return select


def expert(index: int) -> Destination:
    """The slice of a stacked ``[num_experts, ...]`` parameter owned by one expert."""

    def select(param: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        return param[index]

    return select


def expert_half(index: int, half_index: int) -> Destination:
    """Half ``half_index`` of expert ``index``'s slice of a stacked gate/up parameter."""

    def select(param: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        rows = param.shape[1] // 2
        return param[index].narrow(0, half_index * rows, rows)

    return select


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
    "input_layernorm",
    "post_attention_layernorm",
    # MoE router. The dense SwiGLU gate is ``mlp.gate_proj`` and therefore does
    # not match this suffix.
    "mlp.gate",
)

#: HF module path suffix -> which block of the fused ``qkv_proj`` it fills.
_FUSED_QKV: dict[str, int] = {
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
}

#: HF module path suffix -> which half of the dense MLP's fused ``gate_up_proj``
#: it fills. The MoE experts' gate/up are fused separately (see ``_EXPERT_KEY``)
#: and never reach this table.
_FUSED_GATE_UP: dict[str, int] = {"mlp.gate_proj": 0, "mlp.up_proj": 1}

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

#: Suffixes that separate a parameter from the module owning it. Ordered so the
#: longest match wins: ``q_proj.weight_scale_inv`` belongs to ``q_proj``, not to
#: ``q_proj.weight``. ``weight_scale``/``weight_zeros`` are the int4 grids.
_PARAM_SUFFIXES: tuple[str, ...] = (
    ".weight_scale_inv",
    "_scale_inv",
    ".weight_scale",
    ".weight_zeros",
    ".weight",
    ".bias",
)

#: Module path suffix -> the dimension of the *incoming* checkpoint tensor that
#: tensor parallelism splits. Column-parallel weights are cut along their output
#: rows (dim 0), row-parallel ones along the contracted columns (dim 1); a stacked
#: expert parameter is fed one expert at a time, so its own leading expert axis
#: does not appear here. The two vocabulary tensors are ``[vocab, hidden]``, so
#: splitting the vocabulary is also a dim-0 cut.
_SHARD_DIM: tuple[tuple[str, int], ...] = (
    ("embed_tokens", 0),
    ("lm_head", 0),
    ("self_attn.qkv_proj", 0),
    ("self_attn.o_proj", 1),
    ("mlp.experts.gate_up_proj", 0),
    ("mlp.experts.down_proj", 1),
    ("mlp.gate_up_proj", 0),
    ("mlp.down_proj", 1),
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
        # The scale grid is stacked alongside the weight under its own name,
        # because a ParameterDict entry cannot carry a second leaf.
        suffix = "_scale_inv" if experts["leaf"].endswith("_scale_inv") else ""
        if experts["proj"] == "down":
            return f"{prefix}.down_proj{suffix}", expert(index)
        # gate and up are fused along dim 0 inside each expert's slice.
        return (
            f"{prefix}.gate_up_proj{suffix}",
            expert_half(index, 0 if experts["proj"] == "gate" else 1),
        )

    module, _, leaf = key.rpartition(".")
    for suffix, index in _FUSED_QKV.items():
        if module.endswith(suffix):
            return f"{module[: -len(suffix)]}self_attn.qkv_proj.{leaf}", qkv_segment(index)
    for suffix, index in _FUSED_GATE_UP.items():
        if module.endswith(suffix):
            return f"{module[: -len(suffix)]}mlp.gate_up_proj.{leaf}", half(index)
    for suffix in _FLATTENED:
        if module.endswith(suffix):
            return f"{module}_{leaf}", whole
    return key, whole


def shard_dim(param_name: str) -> int | None:
    """Dimension of the incoming tensor that ``param_name`` is split along, if any.

    Vision-tower parameters (``vision_tower.*``) are never sharded — the vision
    encoder is replicated across TP ranks. The suffix-based match would otherwise
    falsely trigger on vision keys that share names with text projections
    (e.g. ``self_attn.o_proj``).
    """
    if param_name.startswith("vision_tower."):
        return None
    module = param_name
    for suffix in _PARAM_SUFFIXES:
        if module.endswith(suffix):
            module = module[: -len(suffix)]
            break
    for prefix, dim in _SHARD_DIM:
        if module.endswith(prefix):
            return dim
    return None


def tp_shard(param_name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Narrow ``tensor`` to this rank's slice, or return it unchanged.

    Splitting on the way in rather than loading the full tensor and slicing later
    is what keeps a TP rank's peak memory at its own share of the checkpoint.

    Raises:
        ValueError: If the dimension does not divide evenly across ranks.
    """
    world_size = get_tp_world_size()
    if world_size == 1:
        return tensor
    dim = shard_dim(param_name)
    if dim is None:
        return tensor
    size = tensor.shape[dim]
    if size % world_size != 0:
        raise ValueError(
            f"{param_name}: dimension {dim} of size {size} does not divide across "
            f"{world_size} tensor-parallel ranks"
        )
    return tensor.narrow(dim, get_tp_rank() * (size // world_size), size // world_size)


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
    shard: Sharder | None = None,
) -> None:
    """Copy a HuggingFace checkpoint into ``model``'s already-allocated parameters.

    Args:
        model: Model whose parameters have real storage (see
            :func:`lite_llama.executor.loader.materialise_parameters`).
        weights: ``(checkpoint key, tensor)`` pairs, in any order.
        translate: Maps a checkpoint key to its destination, or ``None`` to skip
            it (HF bookkeeping tensors, keys belonging to another submodule).
        tied: ``{target parameter: source parameter}`` pairs to satisfy by *aliasing*
            when the checkpoint omits the target. Checkpoints with
            ``tie_word_embeddings: true`` ship no ``lm_head.weight`` at all, and
            lite_llama keeps it as its own parameter, so it is pointed at the
            embedding table rather than given a copy of it — which is both half the
            memory and the only way ``lm_head.weight is embed_tokens.weight`` can hold
            once the two are sharded. A checkpoint that *does* ship the target wins:
            the tie is a fallback, not an override.
        shard: Narrows each incoming tensor to the slice this tensor-parallel rank
            owns; :func:`tp_shard` for the models that support TP, ``None`` for
            those that do not (the vision towers are replicated).

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

        if shard is not None:
            tensor = shard(name, tensor)
        view = destination(param.data, tensor)
        if view.shape != tensor.shape:
            raise ValueError(
                f"checkpoint key {key!r} has shape {tuple(tensor.shape)} but "
                f"{name!r} expects {tuple(view.shape)}"
            )
        view.copy_(tensor)
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
