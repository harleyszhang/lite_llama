# Quantization Support

lite_llama supports multiple weight quantization schemes, with an architecture aligned to [sglang](https://github.com/sgl-project/sglang) for extensibility.

## Supported Schemes

| Scheme | Config Class | Weight | Activation | Scale Granularity | Use Case |
|--------|-------------|--------|------------|-------------------|----------|
| **fp8** | `Fp8Config` | fp8-e4m3 | fp16 | 128×128 block | Qwen/DeepSeek FP8 checkpoints |
| **w8a8_fp8** | `W8A8Fp8Config` | fp8-e4m3 | fp8-e4m3 (dynamic) | per-channel / per-token | True W8A8 runtime (`--quantization fp8`) |
| **blockwise_int8** | `BlockInt8Config` | int8 | fp16 | per-channel / group-wise | Runtime int8 (`--quantization int8`) |
| **w8a8_int8** | `W8A8Int8Config` | int8 | int8 (dynamic) | per-channel / per-token | SmoothQuant (`--quantization smoothquant`) |
| **awq** | `AWQConfig` | int4 | fp16 | group-wise (128) | Pre-quantised AWQ checkpoints |
| **gptq** | `GPTQConfig` | int4 | fp16 | group-wise (128) | Pre-quantised GPTQ checkpoints |
| **fp8 KV cache** | `Fp8KVCacheMethod` | — | — | per-tensor | `--kv-cache-dtype fp8` halves KV memory |

## Quick Start

### FP8 Checkpoints (Qwen3-30B-A3B-FP8)

FP8 checkpoints are detected automatically from `config.json`:

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

### Runtime INT8 Quantisation

Quantise an fp16 checkpoint to int8 at load time:

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen3-0.6B --quantization int8
```

### True W8A8 FP8 (no weight dequantisation)

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen3-0.6B --quantization fp8
```

### FP8 KV Cache (halves decode memory)

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen3-0.6B --kv-cache-dtype fp8
```

## Architecture

```
lite_llama/modules/quantization/
├── __init__.py            # BASE_QUANTIZATION_METHODS registry + factory functions
├── base_config.py         # QuantizeMethodBase / LinearMethodBase / FusedMoEMethodBase / QuantizationConfig ABC
├── fp8.py                 # Fp8Config + Fp8LinearMethod + Fp8MoEMethod
├── w8a8_fp8.py            # W8A8Fp8Config + W8A8Fp8LinearMethod + W8A8Fp8MoEMethod
├── w8a8_int8.py           # W8A8Int8Config + W8A8Int8LinearMethod + W8A8Int8MoEMethod
├── blockwise_int8.py      # BlockInt8Config + BlockInt8LinearMethod + BlockInt8MoEMethod
├── awq.py                 # AWQConfig + AWQLinearMethod + AWQMoEMethod
├── gptq.py                # GPTQConfig + GPTQLinearMethod + GPTQMoEMethod
├── unquant.py             # UnquantizedConfig (fp16 default)
├── kv_cache.py            # BaseKVCacheMethod + Fp8KVCacheMethod
├── parameter.py           # RawParameter (loader must not cast to fp16)
└── utils.py               # Quantise helpers + checkpoint layout adapters (AWQ/GPTQ)
```

### sglang Alignment Table

| lite_llama | sglang equivalent | Notes |
|------------|-------------------|-------|
| `QuantizationConfig` | `QuantizationConfig` | ABC with `get_quant_method(layer, prefix)` |
| `LinearMethodBase` | `LinearMethodBase` | `create_weights` + `apply` |
| `FusedMoEMethodBase` | `FusedMoEMethodBase` | Stacked expert strategy |
| `Fp8Config` | `Fp8Config` | Weight-only fp8 (block-wise scales) |
| `W8A8Fp8Config` | `W8A8Fp8Config` | True W8A8 fp8 (per-token act quantisation) |
| `W8A8Int8Config` | `W8A8Int8Config` | SmoothQuant W8A8 |
| `BlockInt8Config` | `BlockInt8Config` | Weight-only int8 |
| `AWQConfig` | `AWQConfig` | Int4 AWQ checkpoints |
| `GPTQConfig` | `GPTQConfig` | Int4 GPTQ checkpoints |
| `BASE_QUANTIZATION_METHODS` | `BASE_QUANTIZATION_METHODS` | `{name: ConfigClass}` registry |

### Registry & Config Flow

```python
from lite_llama.modules.quantization import (
    BASE_QUANTIZATION_METHODS,
    get_quantization_config,
    get_quant_config_from_hf,
    for_runtime_scheme,
)

# Checkpoint auto-detection: config.json → Config class → from_config()
quant = get_quant_config_from_hf(hf_config)  # Fp8Config / AWQConfig / None

# Runtime quantisation: --quantization int8
quant = for_runtime_scheme("int8")  # BlockInt8Config.per_channel()

# Layer asks its config for the right method:
method = quant.get_quant_method(layer, prefix)  # Fp8LinearMethod / ...
```

## Performance Benchmark

Benchmarks on A10 (24 GB), decode batch size 4, max_gen_len=64, greedy. Baseline: HuggingFace transformers fp16 (eager, same prompts).

### Qwen3-0.6B (dense, 28 layers, hidden=1024)

| Config | Model Mem | KV Capacity | TPOT (ms) | TPS | vs HF Speedup |
|--------|-----------|-------------|-----------|-----|---------------|
| HF fp16 (baseline) | 1.17 GB | — | 28.19 | 141.7 | 1.0× |
| lite fp16 | 1.40 GB | 147,875 tok | 4.14 | 918.8 | 6.5× |
| lite int8 | 0.99 GB | 141,549 tok | 4.16 | 904.1 | 6.4× |
| lite int8-blockwise | 1.00 GB | 138,385 tok | 4.44 | 849.4 | 6.0× |
| lite fp8 (W8A8) | 0.99 GB | 139,153 tok | 8.35 | 448.1 | 3.2× |
| lite smoothquant (W8A8) | 0.99 GB | 135,642 tok | 3.70 | 983.8 | 6.9× |

> Model Mem = model weights only; KV Capacity = max cached tokens (paged pool fills remaining GPU memory).
> Benchmark logs: [`docs/benchmark_logs/`](../docs/benchmark_logs/)

### Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, TP2)

30B 级 MoE checkpoint 的权重以 fp8-e4m3（uint8 存储）+ 128×128 block scales 直接驻留显存，kernel 在 `tl.dot` 前反量化（W8A16）。A10 (sm86) 无原生 fp8 tensor core，激活保持 16-bit——checkpoint 声明的 `activation_scheme: dynamic` 只在 sm89+ 的真 W8A8 路径上生效，这里不启用。checkpoint 的 `modules_to_not_convert`（`lm_head`、每层两个 norm、router `mlp.gate`，共 145 项）由 `Fp8Config.ignored` 接住，保持 bf16 不量化。

按层分发的量化算子（`Fp8Config.get_quant_method`）：

| 层 | Quant Method | Kernel | 权重格式 |
|----|--------------|--------|---------|
| `self_attn.qkv_proj` / `o_proj` | `Fp8LinearMethod` | `w8a16_matmul` | fp8-e4m3 + 128×128 block scales |
| `mlp.experts`（128 专家的 gate_up / down） | `Fp8MoEMethod` | `fused_moe` `QUANT_MODE=1` | fp8-e4m3 + block scales（反量化因子 256 折进 `DEQUANT_SCALE` bit-trick） |
| `mlp.gate`（router）/ `lm_head` | `UnquantizedLinearMethod` | cuBLAS 线性层 | bf16 |

两条 kernel 路径都接受 fp16 或 bf16 激活（checkpoint 的 `torch_dtype: bfloat16` 走 bf16）：反量化后的操作数统一对齐激活 dtype 再进 tensor core。TP2 下每卡存半个副本（权重 + KV 各半），decode 走 eager（NCCL 集合通信不能进 CUDA graph）。

A10×2 (22 GiB each), TP2, batch 4, max_gen_len=64, greedy（与上文 Qwen3-0.6B 同口径；HF 侧无法对照——fp8 checkpoint 反量化为 bf16 需 ~60 GB，双卡放不下）：

| Config | Model Mem | KV Capacity | TTFT (ms) | TPOT (ms) | TPS |
|--------|-----------|-------------|-----------|-----------|-----|
| lite fp8 checkpoint (TP2) | 29.06 GB | 104,528 tok | 81.0 | 82.77 | 48.3 |

> Model Mem 为全 replica 权重总量（rank 0 分片 ×2）；KV Capacity 是每卡容量（KV 按 TP 切分后每卡各存一半，同一数字即 replica 的 token 容量）。
> 复现：`PYTHONPATH=. python benchmarks/bench_quant.py --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8 --schemes fp16 --tp 2 --skip-hf`
> e2e 指标见 [`benchmark_models.md`](benchmark_models.md)；量化 kernel 精度回归：`python -m pytest tests/kernels/test_fused_moe.py -k fp8`（fp16 与 bf16 激活各一例，对 fp32 反量化参考）

### Performance Notes

- **lite fp16 vs HF**: 6.5× speedup from CUDA graphs + fused kernels + paged KV
- **int8 per-channel (W8A16)**: Matches fp16 throughput, saves ~0.4 GB weight memory
- **int8-blockwise (W8A16)**: Group-wise scales give finer granularity; slightly slower due to more scale loads
- **smoothquant (W8A8 int8)**: Fastest scheme — both operands are int8, leveraging int8 tensor cores (6.9×)
- **fp8 W8A8**: Per-token activation quantisation overhead on A10 (sm86 lacks native fp8 GEMM); improves on H100/sm90
- **INT4 MoE (AWQ/GPTQ)**: fused_moe kernel supports int4 packed weights with group-wise scales+zeros
- **KV cache fp8**: Not reflected in the table (orthogonal to weight quantisation); halves the KV cache footprint, enabling ~2× longer sequences

## Accuracy

Token-level accuracy comparison (greedy decode, same prompt set):

| Scheme | Token Match vs HF fp16 | Expected |
|--------|----------------------|----------|
| lite fp16 | ~25% | Normal — different attention kernel numerics cause divergence after first mismatch |
| int8 per-channel | ~23% | Within fp16 divergence range |
| fp8 W8A8 | ~5% | e4m3's 3 mantissa bits cause earlier divergence |

> **Logits-level accuracy** (measured per-token before greedy argmax) shows much higher
> agreement: int8 <0.03% relative error, fp8 <0.04% — confirming the quantisation does
> not degrade model quality, only greedy decode amplifies rounding differences.

## Running Benchmarks

```bash
# Single model with specific schemes
python benchmarks/bench_quant.py --model-dir /data/shared/llm_weights/Qwen3-0.6B \
    --schemes fp16 int8 fp8

# All representative models (plan subset)
python benchmarks/bench_quant.py --all

# Output JSON for CI tracking
python benchmarks/bench_quant.py --model-dir ... --json results.json
```

## Vision-Language Model Support

Both TP and quantization extend to multimodal (VLM) checkpoints. The vision tower stays replicated in its native dtype; only the language-model projections are sharded or quantised.

| Model | Weights | TP | INT8 | vl-chat |
|-------|---------|----|----|--------|
| Qwen3-VL-4B-Instruct | BF16 | ✓ | ✓ | ✓ |
| LLaVA-1.5-7B (HF) | FP16 | ✓ | ✓ | ✓ |
