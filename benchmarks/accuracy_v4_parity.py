"""DeepSeek-V4-Flash-6layers: lite_llama vs transformers over the real DSpark checkpoint.

The checkpoint is DeepSeek's inference format (DSpark keys, fp8 linears,
MXFP4 experts). vLLM cannot serve it on these A10s: the Lightning Indexer
and FlashMLA attention require DeepGEMM, whose kernels are SM90+ only
(``vllm/platforms/cuda.py::support_deep_gemm``), so the reference arm is
transformers 5.15's eager DeepseekV4ForCausalLM instead. The DSpark weights
are converted on the fly — keys renamed to HF names, fp8 blocks and MXFP4
expert stacks dequantised — into a bf16 model that fits host memory:

    lite_llama venv, GPUs (TP-2):
        python benchmarks/accuracy_v4_parity.py --arm lite
    lite_llama venv, CPU (reference):
        python benchmarks/accuracy_v4_parity.py --arm hf
    either:
        python benchmarks/accuracy_v4_parity.py --compare LITE.json HF.json

Both arms consume fixed random token ids (seeded, identical) at three prefill
lengths and continue 32 greedy steps — a deterministic input both sides
share verbatim, so no tokenizer round-trip noise enters the comparison.
(The checkpoint does ship a tokenizer; the benchmark section uses it for
real-text prompts.) The comparison is token-level: greedy agreement,
per-step top-5 id agreement and shared-id logprob drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CKPT = "/data/shared/llm_weights/DeepSeek-V4-Flash-6layers"
PREFILL_LENS = [64, 256, 1024]
GREEDY_STEPS = 32
LOG_DIR = Path(__file__).parent / "logs"
SEED = 0


def prompt_ids(length: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED + length)
    return torch.randint(10, 120000, (1, length), generator=g)


def top5_logprobs(logits_1d: torch.Tensor) -> list[list[float]]:
    """[[id, logprob] * 5] for one vocab row, fp32."""
    logp = torch.log_softmax(logits_1d.float(), dim=-1)
    vals, ids = logp.topk(5)
    return [[int(i), float(v)] for i, v in zip(ids.tolist(), vals.tolist())]


def write_arm(name: str, payload: dict) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    from benchmarks.common import timestamped_log_path

    path = timestamped_log_path(LOG_DIR, f"accuracy_v4_{name}")
    path.write_text(json.dumps({"checkpoint": CKPT, **payload}, indent=2))
    print(f"json: {path}")
    return path


# --------------------------------------------------------------------- #
# lite_llama arm — TP-2 payload over the tests' tp harness
# --------------------------------------------------------------------- #


def _lite_payload(rank: int) -> dict:
    """Module-level so tp_harness's spawned workers can pickle it."""
    from lite_llama.executor.attention_metadata import AttentionMetadata
    from lite_llama.executor.loader import materialise_parameters
    from lite_llama.executor.weight_utils import hf_weights_iterator
    from lite_llama.models.config import ModelConfig
    from lite_llama.models.registry import ModelRegistry
    from lite_llama.distributed.parallel_state import tensor_model_parallel_all_gather

    config = ModelConfig.from_pretrained(CKPT, max_seq_len=2048)
    model = ModelRegistry.resolve("deepseek_v4").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(
        hf_weights_iterator(CKPT, "cuda", dequantize_fp8=False, dequant_dtype=config.dtype)
    )
    model.to("cuda").eval()

    out = []
    for length in PREFILL_LENS:
        ids = prompt_ids(length).cuda()
        # V4 attention carries its own window/compressor state; a prefill
        # step resets it inside the model's forward.
        meta = AttentionMetadata()
        meta.is_prefill = True
        meta.b_seq_len = torch.full((1,), length, dtype=torch.long)
        pos = torch.arange(length, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = model(ids, pos, meta)[:, -1]  # [1, vocab_shard]

        # Vocabulary-parallel head: gather the full row to greedy-pick.
        full = tensor_model_parallel_all_gather(logits[0].contiguous()) if logits.shape[-1] != config.vocab_size else logits[0]
        steps = []
        tokens = [int(full.argmax())]
        steps.append({"top5": top5_logprobs(full)})
        nxt = torch.tensor([[tokens[0]]], device="cuda")
        for step in range(GREEDY_STEPS - 1):
            meta = AttentionMetadata()
            meta.is_prefill = False
            meta.b_seq_len = torch.full((1,), length + step + 1, dtype=torch.long)
            pos = torch.full((1, 1), length + step, device="cuda")
            with torch.no_grad():
                logits = model(nxt, pos, meta)[:, -1]
            full = tensor_model_parallel_all_gather(logits[0].contiguous()) if logits.shape[-1] != config.vocab_size else logits[0]
            tokens.append(int(full.argmax()))
            steps.append({"top5": top5_logprobs(full)})
            nxt = torch.tensor([[tokens[-1]]], device="cuda")
        out.append({"seq_len": length, "greedy_tokens": tokens, "steps": steps})
    return {"prompts": out}


def arm_lite() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "distributed"))
    from tp_harness import needs_gpus, run_on_tp_ranks

    @needs_gpus(2)
    def run() -> None:
        results = run_on_tp_ranks(_lite_payload, tp_size=2)
        write_arm("lite", {"prompts": results[0]["prompts"]})

    run()


# --------------------------------------------------------------------- #
# transformers reference arm — DSpark -> HF weights on CPU
# --------------------------------------------------------------------- #

#: DSpark keys that map to an HF parameter with no numeric transform; the
#: target dtype follows the checkpoint (fp32 hc/sink/ape/bias, bf16 rest).
_STATIC_MAP = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_scale": "model.hc_head.hc_scale",
}
_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_scale": "ffn_hc.scale",
    "attn.attn_sink": "self_attn.sinks",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    "attn.indexer.weights_proj.weight": "self_attn.compressor.indexer.scorer.weights_proj.weight",
    "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
    "ffn.gate.weight": "mlp.gate.weight",
    "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid",
}
#: fp8 linears — ``.weight`` (e4m3) plus an ``.scale`` twin of e8m0 blocks.
#: The DSpark prefix drops the leading ``w`` (``wq_a`` -> ``q_a_proj``).
_FP8_NAME = {"wq_a": "q_a", "wq_b": "q_b", "wkv": "kv", "wo_a": "o_a", "wo_b": "o_b"}
_FP8_PATTERNS = [
    (r"^layers\.(\d+)\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight$", "self_attn.{m}_proj.weight"),
    (r"^layers\.(\d+)\.attn\.indexer\.wq_b\.weight$", "self_attn.compressor.indexer.q_b_proj.weight"),
    (r"^layers\.(\d+)\.ffn\.shared_experts\.(w1|w2|w3)\.weight$", "mlp.shared_experts.{m2}_proj.weight"),
]


def _dequant_fp8(w: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """[N, K] e4m3 blocks with [N/128, K/128] e8m0 scales -> fp32."""
    from lite_llama.modules.quantization.mxfp4 import e8m0_to_fp32

    if w.shape[0] % block or w.shape[1] % block:
        raise ValueError(f"fp8 tensor {tuple(w.shape)} is not {block}-divisible")
    s = e8m0_to_fp32(scale).repeat_interleave(block, 0).repeat_interleave(block, 1)
    return w.to(torch.float32) * s


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[N, K//2] byte-packed e2m1 + [N, K//32] e8m0 scales -> fp32 [N, K].

    Packing order per byte: even K position in the low nibble, odd in the
    high — the order the checkpoint (and lite_llama's repack) documents.
    """
    from lite_llama.modules.quantization.mxfp4 import e8m0_to_fp32

    b = packed.view(torch.uint8).to(torch.long)
    n, half_k = b.shape
    vals = torch.empty(n, half_k * 2, dtype=torch.float32)
    vals[:, 0::2] = _E2M1[b & 0xF]
    vals[:, 1::2] = _E2M1[b >> 4]
    return vals * e8m0_to_fp32(scale).repeat_interleave(32, dim=1)


def _fill_from_dspark(model) -> None:
    """Materialise a meta-device HF model straight from the DSpark files.

    The reference runs in fp32: every floating tensor is dequantised/cast
    up on the way in, so the arm is the strongest available oracle for the
    bf16 lite_llama arm to be measured against.
    """
    import time

    from safetensors.torch import safe_open

    index = json.loads((Path(CKPT) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    filled: set[str] = set()

    def assign(hf_key: str, tensor: torch.Tensor) -> None:
        """Swap the parameter wholesale — meta tensors reject ``.data =``."""
        if hf_key not in targets:
            raise KeyError(f"converted key has no HF parameter: {hf_key}")
        module_path, _, leaf = hf_key.rpartition(".")
        module = model.get_submodule(module_path)
        tensor = tensor.to(torch.float32) if tensor.is_floating_point() else tensor
        if leaf in dict(module.named_parameters(recurse=False)):
            import torch.nn as nn

            setattr(module, leaf, nn.Parameter(tensor, requires_grad=False))
        else:
            module.register_buffer(leaf, tensor)
        filled.add(hf_key)

    files = sorted(set(weight_map.values()))
    t0 = time.time()
    for fname in files:
        with safe_open(Path(CKPT) / fname, "pt") as f:
            for key in f.keys():  # noqa: SIM118
                if key.endswith(".scale"):
                    continue  # consumed with its .weight twin below
                tensor = f.get_tensor(key)

                if key in _STATIC_MAP:
                    assign(_STATIC_MAP[key], tensor)
                    continue
                m = re.match(r"^layers\.(\d+)\.(.+)$", key)
                if not m:
                    raise KeyError(f"unmapped DSpark key: {key}")
                layer, leaf = int(m.group(1)), m.group(2)
                hf_leaf = _LAYER_MAP.get(leaf)
                if hf_leaf:
                    assign(f"model.layers.{layer}.{hf_leaf}", tensor)
                    continue
                fp8_hit = next(
                    (p for p in _FP8_PATTERNS if re.match(p[0], key)), None
                )
                if fp8_hit:
                    scale = f.get_tensor(key.replace(".weight", ".scale"))
                    full = _dequant_fp8(tensor, scale)  # fp32
                    sub = re.match(fp8_hit[0], key)
                    g2 = sub.group(2) if sub.lastindex and sub.lastindex >= 2 else ""
                    name = fp8_hit[1].format(
                        m=_FP8_NAME.get(g2, g2),
                        m2={"w1": "gate", "w2": "down", "w3": "up"}.get(g2, g2),
                    )
                    assign(f"model.layers.{sub.group(1)}.{name}", full)
                    continue
                if re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$", key):
                    continue  # experts are filled layer-by-layer below
                raise KeyError(f"unmapped DSpark key: {key}")

    # Expert stacks: build each layer's [E, 2*inter, hidden] gate_up and
    # [E, hidden, inter] down one layer at a time so the dequantised bf16
    # never doubles up in host memory. safetensors handles stay open across
    # experts — one file typically carries many.
    handles: dict[str, object] = {}

    def open_file(fname: str):
        if fname not in handles:
            handles[fname] = safe_open(Path(CKPT) / fname, "pt")
        return handles[fname]

    n_layers = len(model.model.layers)
    for layer in range(n_layers):
        experts = model.model.layers[layer].mlp.experts
        e_total, gu_shape = experts.gate_up_proj.shape[0], experts.gate_up_proj.shape[1:]
        gate_up = torch.empty(e_total, *gu_shape, dtype=torch.float32)
        down = torch.empty(e_total, *experts.down_proj.shape[1:], dtype=torch.float32)
        inter = gu_shape[0] // 2
        for e in range(e_total):
            parts = {}
            for nm in ("w1", "w2", "w3"):
                wk = f"layers.{layer}.ffn.experts.{e}.{nm}.weight"
                sf = open_file(weight_map[wk])
                parts[nm] = (sf.get_tensor(wk), sf.get_tensor(wk.replace(".weight", ".scale")))
            # HF's gate_up packs gate (w1) first, up (w3) second.
            gate_up[e, :inter] = _dequant_mxfp4(*parts["w1"])
            gate_up[e, inter:] = _dequant_mxfp4(*parts["w3"])
            down[e] = _dequant_mxfp4(*parts["w2"])
        assign(f"model.layers.{layer}.mlp.experts.gate_up_proj", gate_up)
        assign(f"model.layers.{layer}.mlp.experts.down_proj", down)
        print(f"  experts layer {layer} dequantised ({time.time() - t0:.0f}s)")
    handles.clear()

    # Rope frequency tables are non-persistent buffers the checkpoint never
    # carries; rebuild every rotary module with real CPU tensors.
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    for name, module in list(model.named_modules()):
        if isinstance(module, DeepseekV4RotaryEmbedding):
            parent_path, _, leaf = name.rpartition(".")
            parent = model if not parent_path else model.get_submodule(parent_path)
            setattr(parent, leaf, DeepseekV4RotaryEmbedding(model.config))

    non_persistent: set[str] = set()
    for name, module in model.named_modules():
        for buf in module._non_persistent_buffers_set:
            non_persistent.add(f"{name}.{buf}" if name else buf)
    missing = (set(targets) - filled) - non_persistent
    if missing:
        raise KeyError(f"HF parameters never filled: {sorted(missing)[:5]}")


def arm_hf() -> None:
    """Reference arm: transformers eager V4 on CPU, DSpark weights converted."""
    import time

    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    from lite_llama.models.config import ModelConfig

    config = ModelConfig.from_pretrained(CKPT, max_seq_len=2048).hf_config
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(config)
    t0 = time.time()
    _fill_from_dspark(model)
    model.eval()  # weights keep their per-tensor dtypes as stored
    print(f"dspark->hf conversion took {time.time() - t0:.0f}s")

    torch.set_num_threads(min(32, torch.get_num_threads()))
    out = []
    for length in PREFILL_LENS:
        ids = prompt_ids(length)
        tokens, steps = [], []
        with torch.no_grad():
            t1 = time.time()
            step = model(ids, use_cache=True)
            cache = step.past_key_values
            logits = step.logits[:, -1]
            steps.append({"top5": top5_logprobs(logits[0])})
            nxt = logits.argmax(-1, keepdim=True)
            tokens.append(int(nxt))
            for _ in range(GREEDY_STEPS - 1):
                step = model(nxt, past_key_values=cache, use_cache=True)
                cache = step.past_key_values
                logits = step.logits[:, -1]
                steps.append({"top5": top5_logprobs(logits[0])})
                nxt = logits.argmax(-1, keepdim=True)
                tokens.append(int(nxt))
        print(f"  seq {length}: {GREEDY_STEPS} greedy steps in {time.time() - t1:.0f}s")
        out.append({"seq_len": length, "greedy_tokens": tokens, "steps": steps})
    write_arm("hf", {"prompts": out})


# --------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------- #


def compare(lite_path: str, hf_path: str) -> None:
    lite = json.loads(Path(lite_path).read_text())
    hf = json.loads(Path(hf_path).read_text())

    for a, b in zip(lite["prompts"], hf["prompts"], strict=True):
        toks_a, toks_b = a["greedy_tokens"], b["greedy_tokens"]
        agree = sum(t == u for t, u in zip(toks_a, toks_b, strict=True))
        first_div = next(
            (i for i, (t, u) in enumerate(zip(toks_a, toks_b, strict=True)) if t != u), -1
        )
        top5_agree = 0
        drift = []
        for sa, sb in zip(a["steps"], b["steps"], strict=True):
            ids_a = {i for i, _ in sa["top5"]}
            ids_b = {i for i, _ in sb["top5"]}
            top5_agree += len(ids_a & ids_b) / 5
            d = {i: v for i, v in sa["top5"]}
            e = {i: v for i, v in sb["top5"]}
            common = d.keys() & e.keys()
            drift.extend(abs(d[i] - e[i]) for i in common)
        n = len(a["steps"])
        print(
            f"seq {a['seq_len']:>5}: greedy {agree}/{len(toks_a)} "
            f"first-div {first_div} | top5-id-agree {top5_agree / n:.3f} "
            f"| shared-logprob max-drift {max(drift, default=0):.4f} "
            f"mean {sum(drift) / len(drift) if drift else 0:.5f}"
        )
        if first_div >= 0:
            print(f"  lite[{first_div}:{first_div + 6}] = {toks_a[first_div:first_div + 6]}")
            print(f"   hf[{first_div}:{first_div + 6}] = {toks_b[first_div:first_div + 6]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["lite", "hf"])
    parser.add_argument("--compare", nargs=2, metavar=("LITE", "HF"))
    args = parser.parse_args()

    if args.arm == "lite":
        arm_lite()
    elif args.arm == "hf":
        arm_hf()
    elif args.compare:
        compare(*args.compare)
    else:
        parser.error("pick --arm lite, --arm hf or --compare LITE HF")


if __name__ == "__main__":
    main()
