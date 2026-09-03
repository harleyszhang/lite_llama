"""Validate the weight mapping against real published checkpoints, offline.

Index files (never weights) are read from locally available
checkpoints; every checkpoint key must reach a parameter and every
parameter be covered, including fp8 scale pairing.

Usage:
    pytest tests/models/test_checkpoint_index.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from lite_llama.executor.loader import init_empty_parameters
from lite_llama.models.config import ModelConfig
from lite_llama.models.registry import ModelRegistry
from tests.conftest import REPO_ROOT

#: Scale tables are consumed by the loader, not handed to the model.
_SCALE_SUFFIX = ".weight_scale_inv"

#: Layers the model never built map to ``None`` on purpose — the MTP/nextn
#: heads a DeepSeek checkpoint ships past its stack
#: (``num_nextn_predict_layers``), or the tail an ``hf_overrides`` trim cut
#: away (``DecoderLayer.translate_weight_key`` documents the drop). Every
#: other key must still reach a parameter: silently dropping those is
#: exactly what this file exists to catch.
_DROPPED_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _dropped_layer(key: str, config: ModelConfig) -> bool:
    match = _DROPPED_LAYER_RE.match(key)
    return match is not None and int(match.group(1)) >= config.num_layers


#: AWQ/GPTQ key renaming (matches the non-tensor part of adapt_packed_checkpoint).
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

    # For packed checkpoints (AWQ/GPTQ, either bit width),
    # adapt_packed_checkpoint renames keys before translate_weight_key sees
    # them. Simulate that renaming here.
    is_packed = config.quant is not None and getattr(config.quant, "is_packed", False)

    translated: dict[str, str | None] = {}
    for key in keys:
        effective_key = key
        if is_packed:
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
    _, translated, params, config = mapping
    unmapped = [
        k
        for k, name in translated.items()
        if (name is None or name not in params) and not _dropped_layer(k, config)
    ]
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
