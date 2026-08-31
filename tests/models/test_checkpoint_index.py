"""Validate the weight mapping against real published checkpoints, without loading them.

``test_weight_parity.py`` proves the mapping is right for checkpoints written by the
*installed* transformers. That leaves one gap: the checkpoints people actually
download were written years and several transformers versions ago, and their key
layouts differ (LLaVA-1.5 ships ``language_model.model.*``, Qwen3-MoE ships one
matrix per expert plus FP8 scale tables, Qwen3-VL omits ``lm_head`` entirely).

A sharded checkpoint carries ``model.safetensors.index.json``, which lists every
key in the repository. That is enough to answer both mapping questions — does each
key reach a real parameter, and is each parameter reached — for a 7B or 30B model
in milliseconds, with the skeleton on the meta device and not one byte of weight
read.

The tests skip when no such checkpoint is present, so they cost nothing in CI while
turning any local ``my_weight/`` checkout into mapping coverage. Point
``LITE_LLAMA_INDEX_DIRS`` (colon-separated) at other checkpoint directories to add
more.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lite_llama.executor.loader import init_empty_parameters
from lite_llama.models.config import ModelConfig
from lite_llama.models.registry import ModelRegistry
from tests.conftest import REPO_ROOT

#: Scale tables are consumed by the loader, not handed to the model.
_SCALE_SUFFIX = ".weight_scale_inv"

#: AWQ/GPTQ key renaming (matches the non-tensor part of adapt_int4_checkpoint).
_INT4_KEY_RENAMES: dict[str, str | None] = {
    ".qweight": ".weight",
    ".qzeros": ".weight_zeros",
    ".scales": ".weight_scale",
    ".g_idx": None,  # dropped
}


def _adapt_int4_key(key: str) -> str | None:
    """Rename an AWQ/GPTQ checkpoint key to canonical form (key only, no tensor)."""
    for suffix, replacement in _INT4_KEY_RENAMES.items():
        if key.endswith(suffix):
            if replacement is None:
                return None
            return key.removesuffix(suffix) + replacement
    return key


def _candidate_dirs() -> list[Path]:
    override = os.environ.get("LITE_LLAMA_INDEX_DIRS")
    if override:
        return [Path(p) for p in override.split(os.pathsep) if p]
    return sorted((REPO_ROOT / "my_weight").glob("*/"))


def _with_index() -> list[Path]:
    return [d for d in _candidate_dirs() if (d / "model.safetensors.index.json").is_file()]


@pytest.fixture(scope="module", params=_with_index(), ids=lambda p: p.name)
def checkpoint(request) -> Path:
    return request.param


@pytest.fixture(scope="module")
def mapping(checkpoint: Path) -> tuple[list[str], dict[str, str | None], set[str], ModelConfig]:
    """Return ``(checkpoint keys, key -> parameter, parameter names, config)``.

    The model is built on the meta device: parameter *names* are all these tests
    need, and a 30B skeleton then costs milliseconds and no memory.
    """
    keys = sorted(
        json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    )
    config = ModelConfig.from_pretrained(checkpoint, max_seq_len=1024)
    try:
        model_cls = ModelRegistry.resolve(config.model_type).load_class()
    except ValueError:
        # A checkpoint whose architecture the registry does not carry (a Qwen1
        # checkout, say) has no mapping to validate: skip with the reason rather
        # than error four tests at setup.
        pytest.skip(f"model_type {config.model_type!r} is not registered")
    with init_empty_parameters():
        model = model_cls(config)

    # For AWQ/GPTQ checkpoints, adapt_int4_checkpoint renames keys before
    # translate_weight_key sees them. Simulate that renaming here.
    is_int4 = config.quant is not None and getattr(config.quant, "is_int4", False)

    translated: dict[str, str | None] = {}
    for key in keys:
        effective_key = key
        if is_int4:
            effective_key = _adapt_int4_key(key)
            if effective_key is None:
                translated[key] = None  # dropped (e.g. g_idx)
                continue
        target = model.translate_weight_key(effective_key)
        translated[key] = None if target is None else target[0]
    return keys, translated, set(dict(model.named_parameters())), config


def _skip_if_none_found() -> None:
    if not _with_index():
        pytest.skip(
            "no sharded checkpoint found; put one in my_weight/ or set LITE_LLAMA_INDEX_DIRS"
        )


def test_at_least_one_checkpoint_was_examined():
    """Make the skip visible rather than reporting a vacuous pass."""
    _skip_if_none_found()
    assert _with_index()


def test_every_checkpoint_key_reaches_a_parameter(mapping):
    """An unmapped key means weights silently dropped on the floor."""
    _, translated, params, _ = mapping
    unmapped = [k for k, name in translated.items() if name is None or name not in params]
    assert not unmapped, f"{len(unmapped)} keys map nowhere, e.g. {unmapped[:5]}"


def test_every_parameter_is_covered(mapping):
    """The mirror image: a parameter no key targets would keep its uninitialised memory.

    ``lm_head.weight`` is exempt for tied checkpoints — they genuinely ship no
    ``lm_head``, and the loader aliases the embedding table onto it instead.
    """
    _, translated, params, config = mapping
    covered = set(translated.values())
    if config.tie_word_embeddings:
        covered |= {name for name in params if name.endswith("lm_head.weight")}
    assert not params - covered, f"uncovered parameters: {sorted(params - covered)[:5]}"


def test_fp8_checkpoints_pair_every_scale_with_a_weight(mapping):
    """A stray ``weight_scale_inv`` would mean an FP8 weight loaded un-dequantised."""
    keys, _, _, _ = mapping
    present = set(keys)
    scales = [k for k in keys if k.endswith(_SCALE_SUFFIX)]
    if not scales:
        pytest.skip("not an FP8 checkpoint")
    orphans = [s for s in scales if s.removesuffix(_SCALE_SUFFIX) + ".weight" not in present]
    assert not orphans, f"scale tables without a weight: {orphans[:5]}"


def test_config_and_checkpoint_agree_on_layer_count(mapping):
    """A config/checkpoint mismatch shows up as a whole layer never being written."""
    _, translated, _, config = mapping
    layers = set()
    for name in translated.values():
        _, marker, rest = (name or "").partition("layers.")
        index = rest.split(".")[0] if marker else ""
        if index.isdigit():
            layers.add(int(index))
    assert layers == set(range(config.num_layers)), (
        f"checkpoint covers {len(layers)} layers but the config declares {config.num_layers}"
    )
