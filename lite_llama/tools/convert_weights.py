"""Convert a HuggingFace / bin checkpoint into the flat layout ``lite_llama`` expects.

Design goals:

* No implicit device placement — load with ``torch.load(map_location="cpu")`` and
  save the same way, so a converter run doesn't require a GPU.
* Streaming when possible — for safetensors we open the file once with
  ``safe_open`` and read tensors lazily instead of instantiating the full HF model
  (which would demand transformers matching the checkpoint's exact version).
* One source of truth for the ``HF key -> lite_llama key`` renames per architecture,
  living in :data:`ARCHITECTURES`.

The output is a single ``<model_name>.pth`` file plus every ``*.json`` and
``tokenizer.model`` copied from the source, ready to feed to :class:`LLMEngine`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Rename rules
# --------------------------------------------------------------------------- #


def _text_layer_renames(prefix_in: str, prefix_out: str) -> dict[str, str]:
    """Return the per-layer HF-key -> lite_llama-key template for a text model."""
    return {
        f"{prefix_in}.{{i}}.self_attn.q_proj.weight": f"{prefix_out}.{{i}}.self_attn.q_proj_weight",
        f"{prefix_in}.{{i}}.self_attn.q_proj.bias": f"{prefix_out}.{{i}}.self_attn.q_proj_bias",
        f"{prefix_in}.{{i}}.self_attn.k_proj.weight": f"{prefix_out}.{{i}}.self_attn.k_proj_weight",
        f"{prefix_in}.{{i}}.self_attn.k_proj.bias": f"{prefix_out}.{{i}}.self_attn.k_proj_bias",
        f"{prefix_in}.{{i}}.self_attn.v_proj.weight": f"{prefix_out}.{{i}}.self_attn.v_proj_weight",
        f"{prefix_in}.{{i}}.self_attn.v_proj.bias": f"{prefix_out}.{{i}}.self_attn.v_proj_bias",
        f"{prefix_in}.{{i}}.self_attn.o_proj.weight": f"{prefix_out}.{{i}}.self_attn.o_proj_weight",
        f"{prefix_in}.{{i}}.self_attn.q_norm.weight": f"{prefix_out}.{{i}}.self_attn.q_norm_weight",
        f"{prefix_in}.{{i}}.self_attn.k_norm.weight": f"{prefix_out}.{{i}}.self_attn.k_norm_weight",
        f"{prefix_in}.{{i}}.mlp.gate_proj.weight": f"{prefix_out}.{{i}}.mlp.gate_proj.weight",
        f"{prefix_in}.{{i}}.mlp.up_proj.weight": f"{prefix_out}.{{i}}.mlp.up_proj.weight",
        f"{prefix_in}.{{i}}.mlp.down_proj.weight": f"{prefix_out}.{{i}}.mlp.down_proj.weight",
        f"{prefix_in}.{{i}}.input_layernorm.weight": f"{prefix_out}.{{i}}.input_layernorm_weight",
        f"{prefix_in}.{{i}}.post_attention_layernorm.weight": f"{prefix_out}.{{i}}.post_attention_layernorm_weight",
    }


@dataclass(frozen=True)
class ArchSpec:
    """Rename rules for one architecture.

    Attributes:
        common: Non-layer key mappings (embeddings, final norm, lm_head).
        per_layer: Per-layer mappings; ``{i}`` is filled in from the config's layer count.
        passthrough_prefixes: Keys with these prefixes are renamed by
            :attr:`passthrough_rename` (or copied verbatim if ``None``). Used for the
            vision tower of multimodal models.
        passthrough_rename: ``(hf_prefix, lite_prefix)`` pairs.
        tied_lm_head: ``(lm_head_key, embed_tokens_key)`` in the *output* layout,
            used when the checkpoint omits ``lm_head`` because embeddings are tied
            (Qwen3-VL). ``None`` means the architecture always ships an lm_head.
        post_rename: Optional hook ``(state_dict, num_layers) -> None`` run after
            the rename pass (and before K/V fusion) for architectures that need
            structural transforms, e.g. FP8 dequantisation + expert stacking.
    """

    common: dict[str, str]
    per_layer: dict[str, str]
    passthrough_rename: tuple[tuple[str, str], ...] = ()
    tied_lm_head: tuple[str, str] | None = None
    post_rename: Callable[[dict[str, torch.Tensor], int], None] | None = None

    def build_map(self, num_layers: int) -> dict[str, str]:
        mapping = dict(self.common)
        for i in range(num_layers):
            mapping.update({hf.format(i=i): out.format(i=i) for hf, out in self.per_layer.items()})
        return mapping


# --------------------------------------------------------------------------- #
# FP8 dequantisation and MoE expert stacking (qwen3_moe)
# --------------------------------------------------------------------------- #

# Block size of the fine-grained FP8 format used by Qwen FP8 checkpoints:
# ``weight`` is e4m3 and ``weight_scale_inv[i, j]`` scales the 128x128 block
# starting at ``(i*128, j*128)``.
_FP8_BLOCK = 128


def _dequant_block_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantise a block-wise FP8 (e4m3) matrix: ``W[i,j] = w8[i,j] * s[i//B, j//B]``.

    The multiply runs in fp32; casting the fp8 values first is exact (every
    e4m3 value is representable in fp16/fp32), so accuracy is governed solely
    by the final cast to fp16.
    """
    w = weight.to(torch.float32)
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
    # The last block may be partial when a dimension is not a multiple of 128.
    scale = scale[: w.shape[0], : w.shape[1]]
    return (w * scale).to(torch.float16)


def _qwen3_moe_post_rename(state: dict[str, torch.Tensor], num_layers: int) -> None:
    """Dequantise FP8 weights in place and stack per-expert matrices.

    Runs on the *renamed* state dict: text-projection weights already carry
    their lite_llama names (``q_proj_weight``) while expert weights kept the
    passthrough ``.weight`` suffix, so the partner of each ``*_scale_inv`` key
    is resolved by trying both endings.
    """
    for scale_key in [k for k in state if k.endswith(".weight_scale_inv")]:
        scale = state.pop(scale_key)
        base = scale_key[: -len(".weight_scale_inv")]
        weight_key = base + "_weight" if base + "_weight" in state else base + ".weight"
        state[weight_key] = _dequant_block_fp8(state[weight_key], scale)

    for i in range(num_layers):
        prefix = f"layers.{i}.mlp.experts"
        if f"{prefix}.0.gate_proj.weight" not in state:
            continue  # dense layer (mlp_only_layers)
        gates, ups, downs = [], [], []
        e = 0
        while (key := f"{prefix}.{e}.gate_proj.weight") in state:
            gates.append(state.pop(key))
            ups.append(state.pop(f"{prefix}.{e}.up_proj.weight"))
            downs.append(state.pop(f"{prefix}.{e}.down_proj.weight"))
            e += 1
        # [E, moe_inter, H] + [E, moe_inter, H] -> [E, 2*moe_inter, H]
        state[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
        )
        state[f"{prefix}.down_proj"] = torch.stack(downs)


# Canonical HF text-model layout shared by llama / qwen2 / qwen3 / qwen3_moe:
# ``model.{embed_tokens,norm}`` + ``lm_head`` at the top level, transformer blocks
# under ``model.layers``. Only multimodal wrappers deviate from it.
_TEXT_COMMON_RENAMES = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm_weight",
    "lm_head.weight": "lm_head_weight",
}
_TEXT_TIED_LM_HEAD = ("lm_head_weight", "embed_tokens.weight")


def _text_arch_spec(
    extra_layer: dict[str, str] | None = None,
    passthrough_rename: tuple[tuple[str, str], ...] = (),
    post_rename=None,
) -> ArchSpec:
    """Build the :class:`ArchSpec` for the canonical text layout (Factory).

    ``extra_layer`` entries merge into (and may override) the standard per-layer
    renames — e.g. the MoE router ``mlp.gate`` -> ``mlp.gate_weight``.
    """
    return ArchSpec(
        common=dict(_TEXT_COMMON_RENAMES),
        per_layer={**_text_layer_renames("model.layers", "layers"), **(extra_layer or {})},
        passthrough_rename=passthrough_rename,
        tied_lm_head=_TEXT_TIED_LM_HEAD,
        post_rename=post_rename,
    )


ARCHITECTURES: dict[str, ArchSpec] = {
    "llama": _text_arch_spec(),
    "qwen2": _text_arch_spec(),
    "qwen3": _text_arch_spec(),
    "qwen3_moe": _text_arch_spec(
        extra_layer={"model.layers.{i}.mlp.gate.weight": "layers.{i}.mlp.gate_weight"},
        passthrough_rename=(("model.", ""),),
        post_rename=_qwen3_moe_post_rename,
    ),
    "llava": ArchSpec(
        common={
            # Newer transformers (>= 4.52) drop the ``.model`` after language_model.
            "language_model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "language_model.norm.weight": "language_model.norm_weight",
            "language_model.lm_head.weight": "language_model.lm_head_weight",
            # Legacy layout kept working: it maps to the same lite_llama keys.
            "language_model.model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "language_model.model.norm.weight": "language_model.norm_weight",
            "language_model.model.lm_head.weight": "language_model.lm_head_weight",
            "multi_modal_projector.linear_1.weight": "multi_modal_projector.linear_1.weight",
            "multi_modal_projector.linear_1.bias": "multi_modal_projector.linear_1.bias",
            "multi_modal_projector.linear_2.weight": "multi_modal_projector.linear_2.weight",
            "multi_modal_projector.linear_2.bias": "multi_modal_projector.linear_2.bias",
        },
        per_layer={
            # Cover both new (``language_model.layers``) and legacy (``language_model.model.layers``) layouts.
            **_text_layer_renames("language_model.layers", "language_model.layers"),
            **_text_layer_renames("language_model.model.layers", "language_model.layers"),
        },
        passthrough_rename=(("vision_tower.", "vision_tower."),),
        tied_lm_head=("language_model.lm_head_weight", "language_model.embed_tokens.weight"),
    ),
    "qwen3_vl": ArchSpec(
        common={
            "model.language_model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "model.language_model.norm.weight": "language_model.norm_weight",
            "lm_head.weight": "language_model.lm_head_weight",
        },
        per_layer=_text_layer_renames("model.language_model.layers", "language_model.layers"),
        passthrough_rename=(("model.visual.", "vision_tower."),),
        tied_lm_head=("language_model.lm_head_weight", "language_model.embed_tokens.weight"),
    ),
}


# --------------------------------------------------------------------------- #
# Streaming loaders
# --------------------------------------------------------------------------- #


def _iter_safetensors(root: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` pairs from every ``*.safetensors`` shard in ``root``."""
    try:
        from safetensors import safe_open
    except ImportError as e:  # pragma: no cover — dependency of transformers
        raise RuntimeError("`safetensors` is required to read *.safetensors weights") from e

    shards = sorted(root.glob("*.safetensors"))
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            # safetensors >= 0.6 dropped iteration on safe_open; keys() works everywhere.
            for key in f.keys():  # noqa: SIM118 - safe_open is not iterable directly
                yield key, f.get_tensor(key)


def _iter_pytorch_bin(root: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` pairs from every ``pytorch_model*.bin`` shard."""
    shards = sorted(root.glob("pytorch_model*.bin"))
    if not shards:
        shards = sorted(root.glob("*.bin"))
    for shard in shards:
        state = torch.load(shard, map_location="cpu", weights_only=True)
        yield from state.items()


def _iter_state_dict(root: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Pick the appropriate streaming loader for the checkpoint format in ``root``."""
    if list(root.glob("*.safetensors")):
        return _iter_safetensors(root)
    if list(root.glob("*.bin")):
        return _iter_pytorch_bin(root)
    raise FileNotFoundError(f"no safetensors or pytorch_model*.bin found in {root}")


# --------------------------------------------------------------------------- #
# Fusion and conversion
# --------------------------------------------------------------------------- #


def _fuse_kv(state: dict[str, torch.Tensor], layer_prefix: str, num_layers: int) -> None:
    """Concatenate K and V projections along dim 0 into a single ``kv_proj_*``.

    The kernels benefit from one linear producing both halves in one launch;
    everything downstream expects the fused layout, so this happens in the
    converter rather than the model.
    """
    for i in range(num_layers):
        p = layer_prefix.format(i=i)
        for kind in ("weight", "bias"):
            k = f"{p}.k_proj_{kind}"
            v = f"{p}.v_proj_{kind}"
            if k in state and v in state:
                state[f"{p}.kv_proj_{kind}"] = torch.cat([state.pop(k), state.pop(v)], dim=0)


def _rename_passthrough(key: str, rules: tuple[tuple[str, str], ...]) -> str | None:
    """Return the renamed key when it matches a passthrough prefix, else ``None``."""
    for src, dst in rules:
        if key.startswith(src):
            return dst + key[len(src) :]
    return None


def _tie_lm_head(
    state: dict[str, torch.Tensor], spec: ArchSpec, src: Path, model_type: str
) -> None:
    """Materialise a missing ``lm_head`` from the embedding table when weights are tied.

    Checkpoints with ``tie_word_embeddings: true`` (Qwen3-VL) omit ``lm_head.weight``
    entirely, but the lite_llama model keeps a separate parameter and loads with
    ``strict=True``, so the tied weight must be created here. The *same* tensor is
    referenced (not cloned) so ``torch.save`` stores the table once.
    """
    if spec.tied_lm_head is None:
        return
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if model_type in ("llava", "qwen3_vl"):
        config = config.get("text_config", {}) or config
    if not config.get("tie_word_embeddings", False):
        return

    lm_head_key, embed_key = spec.tied_lm_head
    if lm_head_key in state or embed_key not in state:
        return
    state[lm_head_key] = state[embed_key]
    logger.info("lm_head absent from checkpoint; tied it to %s as %s", embed_key, lm_head_key)


def _convert(
    src_dir: Path,
    model_type: str,
    num_layers: int,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Stream a source checkpoint and return the renamed lite_llama state dict."""
    spec = ARCHITECTURES[model_type]
    rename_map = spec.build_map(num_layers)

    out: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, tensor in tqdm(_iter_state_dict(src_dir), desc=f"[{model_type}] rename"):
        # FP8 scale factors must stay fp32; everything else follows --dtype.
        if (
            dtype is not None
            and not key.endswith(".weight_scale_inv")
            and tensor.is_floating_point()
            and tensor.dtype != dtype
        ):
            tensor = tensor.to(dtype)

        # Explicit rename rules win over blanket passthrough prefixes.
        target = rename_map.get(key)
        if target is not None:
            out[target] = tensor
            continue
        renamed = _rename_passthrough(key, spec.passthrough_rename)
        if renamed is not None:
            out[renamed] = tensor
            continue
        dropped.append(key)

    if dropped:
        logger.debug("dropped %d unmapped keys (first few: %s)", len(dropped), dropped[:5])

    if spec.post_rename is not None:
        spec.post_rename(out, num_layers)

    # Text and multimodal layouts both use ``[language_model.]layers.{i}.self_attn``.
    layer_prefix = (
        "language_model.layers.{i}.self_attn"
        if model_type in ("llava", "qwen3_vl")
        else "layers.{i}.self_attn"
    )
    _fuse_kv(out, layer_prefix, num_layers)
    _tie_lm_head(out, spec, src_dir, model_type)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _copy_metadata(src: Path, dst: Path) -> None:
    for pattern in ("*.json", "tokenizer.model", "tokenizer.json", "special_tokens_map.json"):
        for file in src.glob(pattern):
            shutil.copy2(file, dst)


def _detect_model_type(src: Path) -> str:
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    model_type = config.get("model_type", "").lower()
    if model_type not in ARCHITECTURES:
        raise ValueError(
            f"unsupported model_type {model_type!r}; supported: {', '.join(sorted(ARCHITECTURES))}"
        )
    return model_type


def _detect_num_layers(src: Path, model_type: str) -> int:
    """Read the text model's layer count from ``config.json``.

    LLaVA-1.5 ships an abbreviated ``text_config`` (just tokenizer / rms defaults),
    so ``num_hidden_layers`` may be absent. When it is, resolve via the HF
    ``AutoConfig`` machinery, which walks ``_name_or_path`` to reconstruct the full
    text config (Vicuna-7B: 32 layers).
    """
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if model_type in ("llava", "qwen3_vl"):
        text_cfg = config.get("text_config", {})
        if "num_hidden_layers" in text_cfg:
            return text_cfg["num_hidden_layers"]
        from transformers import AutoConfig

        full = AutoConfig.from_pretrained(src, trust_remote_code=True)
        return full.text_config.num_hidden_layers
    return config["num_hidden_layers"]


_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lite-llama-convert",
        description="Convert HuggingFace weights into the lite_llama flat state_dict",
    )
    parser.add_argument("checkpoints_dir", type=Path, help="Source HF checkpoint directory")
    parser.add_argument(
        "--model-type",
        choices=sorted(ARCHITECTURES),
        help="Override model_type; auto-detected from config.json when omitted",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to ./my_weight/<src-name>/)",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="float16",
        help="Cast floating-point tensors to this dtype",
    )
    args = parser.parse_args(argv)

    src: Path = args.checkpoints_dir.resolve()
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")
    model_type = args.model_type or _detect_model_type(src)
    num_layers = _detect_num_layers(src, model_type)
    logger.info("model_type=%s, num_layers=%d", model_type, num_layers)

    out_dir = args.out_dir or (Path.cwd() / "my_weight" / src.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_state = _convert(src, model_type, num_layers, _DTYPES[args.dtype])
    ckpt_path = out_dir / f"{src.name}.pth"
    logger.info("saving %d tensors to %s", len(new_state), ckpt_path)
    torch.save(new_state, ckpt_path, _use_new_zipfile_serialization=True)
    _copy_metadata(src, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
