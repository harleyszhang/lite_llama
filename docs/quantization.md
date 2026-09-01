# Quantization Support

lite_llama supports multiple weight quantization schemes, with an architecture aligned to [sglang](https://github.com/sgl-project/sglang) for extensibility.

## Supported Schemes

| Scheme | Config Class | Weight | Activation | Scale Granularity | Use Case |
|---|---|---|---|---|---|
| **fp8** | `Fp8Config` | fp8-e4m3 | fp16 | 128×128 block | Qwen/DeepSeek FP8 checkpoints |
| **w8a8_fp8** | `W8A8Fp8Config` | fp8-e4m3 | fp8-e4m3 (dynamic) | per-channel / per-token | True W8A8 runtime (`--quantization fp8`), dense **and** MoE |
| **blockwise_int8** | `BlockInt8Config` | int8 | fp16 | per-channel / group-wise | Runtime int8 (`--quantization int8`) |
| **w8a8_int8** | `W8A8Int8Config` | int8 | int8 (dynamic) | per-channel / per-token | SmoothQuant (`--quantization smoothquant`) |
| **nvfp4** | `NVFP4Config` | fp4-e2m1 | bf16/fp16 | 16-element block + per-tensor | Weight-only 4-bit (`--quantization nvfp4`), dense only |
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

### NVFP4 Weight-Only 4-bit

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen3-4B --quantization nvfp4
```

Smallest weights of any scheme here (2.85× below bf16), and **slower than bf16 at
every shape measured on H100** — see [NVFP4](#nvfp4-weight-only-fp4) before choosing it.

## Architecture

```
lite_llama/modules/quantization/
├── __init__.py            # BASE_QUANTIZATION_METHODS registry + factory functions
├── base_config.py         # QuantizeMethodBase / LinearMethodBase / FusedMoEMethodBase / QuantizationConfig ABC
├── fp8.py                 # Fp8Config + Fp8LinearMethod + Fp8MoEMethod
├── w8a8_fp8.py            # W8A8Fp8Config + W8A8Fp8LinearMethod + W8A8Fp8MoEMethod (A8 experts)
├── w8a8_int8.py           # W8A8Int8Config + W8A8Int8LinearMethod + W8A8Int8MoEMethod
├── blockwise_int8.py      # BlockInt8Config + BlockInt8LinearMethod + BlockInt8MoEMethod
├── nvfp4.py               # NVFP4Config + NVFP4LinearMethod (weight-only, dense only)
├── awq.py                 # AWQConfig + AWQLinearMethod + AWQMoEMethod
├── gptq.py                # GPTQConfig + GPTQLinearMethod + GPTQMoEMethod
├── unquant.py             # UnquantizedConfig (fp16 default)
├── kv_cache.py            # BaseKVCacheMethod + Fp8KVCacheMethod
├── parameter.py           # RawParameter (loader must not cast to fp16)
└── utils.py               # Quantise helpers + checkpoint layout adapters (AWQ/GPTQ)
```

### sglang Alignment Table

| lite_llama | sglang equivalent | Notes |
|---|---|---|
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

Single NVIDIA A10 (24 GB, sm86), decode batch size 4, max_gen_len=64, greedy; measured 2026-08-23 (the run's JSON predates environment logging — the A10 host's stack at the time was torch 2.11.0+cu129 / triton 3.6.0 / python 3.12, per the same week's e2e logs). Baseline: HuggingFace transformers fp16 (eager, same prompts).

### Qwen3-0.6B (dense, 28 layers, hidden=1024)

| Config | Model Mem | KV Capacity | TPOT (ms) | TPS | vs HF Speedup |
|---|---|---|---|---|---|
| HF fp16 (baseline) | 1.17 GB | — | 28.19 | 141.7 | 1.0× |
| lite fp16 | 1.40 GB | 147,875 tok | 4.14 | 918.8 | 6.5× |
| lite int8 | 0.99 GB | 141,549 tok | 4.16 | 904.1 | 6.4× |
| lite int8-blockwise | 1.00 GB | 138,385 tok | 4.44 | 849.4 | 6.0× |
| lite fp8 (W8A8) | 0.99 GB | 139,153 tok | 8.35 | 448.1 | 3.2× |
| lite smoothquant (W8A8) | 0.99 GB | 135,642 tok | 3.70 | 983.8 | 6.9× |

> Model Mem = model weights only; KV Capacity = max cached tokens (paged pool fills remaining GPU memory).
> Benchmark logs: [`docs/benchmark_logs/`](../docs/benchmark_logs/)
>
> The table above is an A10 result on a 0.6B model. For the full 2×H100 matrix on
> Qwen3-4B and Qwen3-30B-A3B — every scheme × TP/DP × CUDA graph × KV dtype, offline
> and online, with both accuracy references — see
> [`quant_matrix_20260901.md`](benchmark_logs/quant_matrix_20260901.md). Its headline
> result contradicts the ordering here: on an H100 at 4B, **no quantisation scheme
> beats bf16 on speed**, and quantisation buys KV capacity instead.

### Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, 2×H100)

30B 级 MoE checkpoint 的权重以 fp8-e4m3（uint8 存储）+ 128×128 block scales 直接驻留显存，kernel 在 `tl.dot` 前反量化（W8A16）。真 W8A8 路径（`--quantization fp8`，`w8a8_fp8` scheme）用 per-channel weight scale，与本 checkpoint 的 block-scale 布局不匹配——声明 `activation_scheme: dynamic` 的 A8 路径属于那条 runtime scheme，不在此 checkpoint 上启用。checkpoint 的 `modules_to_not_convert`（`lm_head`、每层两个 norm、router `mlp.gate`，共 145 项）由 `Fp8Config.ignored` 接住，保持 bf16 不量化。

按层分发的量化算子（`Fp8Config.get_quant_method`）：

| 层 | Quant Method | Kernel | 权重格式 |
|----|--------------|--------|---------|
| `self_attn.qkv_proj` / `o_proj` | `Fp8LinearMethod` | `w8a16_matmul` | fp8-e4m3 + 128×128 block scales |
| `mlp.experts`（128 专家的 gate_up / down） | `Fp8MoEMethod` | `fused_moe` `QUANT_MODE=1` | fp8-e4m3 + block scales（反量化因子 256 折进 `DEQUANT_SCALE` bit-trick） |
| `mlp.gate`（router）/ `lm_head` | `UnquantizedLinearMethod` | cuBLAS 线性层 | bf16 |

两条 kernel 路径都接受 fp16 或 bf16 激活（checkpoint 的 `torch_dtype: bfloat16` 走 bf16）：反量化后的操作数统一对齐激活 dtype 再进 tensor core。TP2 下每卡存半个副本（权重 14.53 GB + KV 各半）；连续批处理引擎的 TP-safe graph 捕获让 TP2 的 decode 同样走 graph（设计见 [tensor_parallel.md](tensor_parallel.md)）——早期"NCCL 集合通信不能进 CUDA graph 捕获"的 eager 限制属于旧引擎路径，已被推翻。

2×H100 80GB (sm90)，batch 4，max_gen_len=64，greedy，max_seq_len 1024。golden 基线为本 checkpoint 的 eager/TP1/KV-auto 配置（`scripts/golden_tokens.py` 录制，control row 复现 1.000）：

| Config | Model Mem | KV Capacity | TTFT (ms) | TPOT (ms) | TPS | golden prefix |
|--------|-----------|-------------|-----------|-----------|-----|---------------|
| tp1+graph | 29.03 GB | 435,879 tok | 66.1 | 13.16 | 285.9 | 1.000 |
| tp1+eager | 29.03 GB | 452,263 tok | 62.5 | 63.35 | 63.2 | 1.000 |
| tp2+graph | 14.53 GB | 1,171,324 tok | 77.5 | 12.76 | 290.3 | 0.638 |
| tp2+eager | 14.53 GB | 1,204,092 tok | 76.1 | 78.50 | 51.0 | 0.638 |
| tp1+kv-fp8+graph | 29.03 GB | 871,790 tok | 67.3 | 14.05 | 268.7 | 0.697 |
| tp1+kv-fp8+eager | 29.03 GB | 904,558 tok | 68.9 | 68.97 | 58.0 | 0.697 |
| tp2+kv-fp8+graph | 14.53 GB | 2,342,680 tok | 82.8 | 13.64 | 271.6 | 0.751 |
| tp2+kv-fp8+eager | 14.53 GB | 2,408,216 tok | 83.1 | 85.52 | 46.8 | 0.751 |
| tp1+dp2+graph | 29.03 GB/卡 | 435,879 tok/卡 | — | — | 559.2 | 0.872 |

读法：

- **CUDA graph 是这个尺寸的主要杠杆**：TPOT 13.16 vs 63.35 ms（4.8×）。48 层 MoE 每 token 的 kernel 数在 eager 下 launch-bound，graph 把整步折叠成一次 replay；TP2 同理（12.76 vs 78.50 ms）。
- **TP2 买容量不买速度**：290 vs 286 TPS 持平，KV 容量 ×2.7（1.17M token）——MoE 权重按专家维切分后每卡读取量减半，抵消了集合通信开销。
- **KV fp8 买容量只付 ~7%**：容量翻倍（436K→872K，TP2 下到 2.34M token），TPOT 13.16→14.05 ms。
- **DP2 近线性扩展**：559 TPS ≈ 2×285.9×0.98，每 replica 独立持有 graph。
- **精度**：TP1 下 graph 与 eager 数值等价（golden 1.000，26/26 exact）。TP2 的 0.638 与 KV fp8 的 0.697 都是 greedy 混沌对首个分叉 token 的放大（prefix 计首差前的长度）：TP2 的分叉来自 all-reduce 顺序，KV fp8 的来自 KV 舍入——量级与 0.6B 档 [quant_matrix_20260901.md](benchmark_logs/quant_matrix_20260901.md) §4 的误差谱一致。
- 同 checkpoint 在 A10×2（22 GiB、TP2 eager 旧口径）TPOT 82.77 ms / TPS 48.3——H100 单卡 graph 是它的 5.9×。

> Model Mem 为全 replica 权重总量（rank 0 分片 × TP）；KV Capacity 为每卡容量（KV 按 TP 切分后同一数字即 replica 的 token 容量）。
> 复现：`python benchmarks/bench_quant.py --model-dir <Qwen3-30B-A3B-Instruct-2507-FP8> --schemes fp16 --kv-cache-dtype auto fp8 --tp 1 2 --cuda-graph --no-cuda-graph --skip-hf --json docs/benchmark_logs/bench_quant_Qwen3-30B-A3B-FP8_20260901.json`；DP 行另跑 `--tp 1 --dp 2 --cuda-graph`；golden 基线：`python scripts/golden_tokens.py --save tests/golden/data/Qwen3-30B-A3B-Instruct-2507-FP8.json --model-dir <...>`。
> e2e 指标见 [`benchmark_models.md`](benchmark_models.md)；量化 kernel 精度回归：`python -m pytest tests/kernels/test_fused_moe.py -k fp8`（fp16 与 bf16 激活各一例，对 fp32 反量化参考）

### 未覆盖的 FP8 checkpoint

- **Qwen3.8-27B-FP8**（`model_type: qwen3_5`，`Qwen3_5ForConditionalGeneration`）：64 层中 48 层是 linear attention（gated-delta-net：conv kernel 4、16 key heads × 128 dim、48 value heads × 128 dim）、每 4 层插一层 full attention，另带 vision tower。`lite_llama/models/` 支持到 qwen3_moe / qwen3_vl，尚无 qwen3_5——需要 linear attention 的 chunked-scan 内核与混合层调度，属新模型实现而非量化路径问题（其 fp8 格式与 30B-A3B 完全一致：e4m3 + 128×128 block scales + dynamic activation）。
- **Qwen3-VL-235B-A22B-Instruct-FP8**：本地副本不完整——index 要求 24 个 shard 仅存在 3 个（22/23/24），且无 config.json（27 GB ≄ ~235 GB），物理上无法加载。

### Performance Notes

- **lite fp16 vs HF**: 6.5× speedup from CUDA graphs + fused kernels + paged KV
- **int8 per-channel (W8A16)**: Matches fp16 throughput, saves ~0.4 GB weight memory
- **int8-blockwise (W8A16)**: Group-wise scales give finer granularity; slightly slower due to more scale loads
- **smoothquant (W8A8 int8)**: Fastest scheme — both operands are int8, leveraging int8 tensor cores (6.9×)
- **fp8 W8A8**: Per-token activation quantisation overhead on A10 (sm86 lacks native fp8 GEMM); improves on H100/sm90
- **INT4 MoE (AWQ/GPTQ)**: fused_moe kernel supports int4 packed weights with group-wise scales+zeros
- **KV cache fp8**: Not reflected in the table (orthogonal to weight quantisation); halves the KV cache footprint, enabling ~2× longer sequences

## NVFP4 Weight-Only FP4

NVIDIA ModelOpt / TensorRT-LLM layout, implemented as `lite_llama/kernels/ops/quantization/nvfp4.py`
and dispatched as `native/linear_nvfp4`:

- weights fp4-e2m1, two values per `uint8` byte (low nibble = even index);
- one fp8-e4m3 block scale per 16 consecutive `k` elements, so a `BLOCK_K` that is a
  multiple of 16 keeps every k-tile inside one scale;
- one fp32 `weight_global_scale` per tensor;
- `w = e2m1(nibble) * dequant_e4m3(block_scale) * global_scale`.

TP sharding requires shards to be multiples of 32 (2 values/byte × 16-element block),
enforced by `NVFP4Config.shard_is_aligned`, so no shard splits a byte or a block scale.
MoE experts are **not** implemented: `get_quant_method` raises on a fused-MoE layer
rather than silently falling back to bf16 experts.

### What it costs and what it buys

sm90 has no fp4 MMA and Triton has no fp4 dtype, so this is weight-only by
construction: the nibbles are unpacked in registers and the `tl.dot` still runs at
bf16. The saving is bytes, and on an H100 bytes are not the decode bottleneck.

`qwen3-4b/qkv` (N=6144, K=2560), measured on an NVIDIA H100 80GB HBM3
(torch 2.13.0+cu130 / triton 3.7.1 / python 3.14.7), from
[`bench_quant_gemm_h100_20260901.json`](benchmark_logs/bench_quant_gemm_h100_20260901.json)
(run with `LITE_LLAMA_AUTOTUNE=0`, so these are the heuristic tiles a user gets
without a tuning cache):

| M | bf16 | fp8 W8A8 | int4 (awq) | nvfp4 |
|---|---|---|---|---|
| 1 | 21.7 µs | 24.0 µs | 22.2 µs | **49.0 µs** |
| 128 | 21.5 µs | 28.5 µs | 50.6 µs | **68.5 µs** |
| 2048 | 89.1 µs | 166.1 µs | 334.8 µs | **755.3 µs** |

End to end on Qwen3-4B-Thinking-2507 on the same H100 stack (batch 4, 64 new tokens,
greedy): weights 2.63 GB against bf16's 7.49 GB, TPOT 13.66 ms against bf16's 4.77 ms.

**Read that as a memory result, not a speed result.** The NVFP4 row moves ~3.6× fewer
weight bytes and is still 2.3× slower at decode and 8.5× slower at prefill, because
unpacking a nibble, widening e2m1 through bit arithmetic and applying two scales
costs more ALU work than an H100's HBM3 saves in time. NVFP4 is the right choice
when a checkpoint does not otherwise fit, and the wrong one when it does.

int4/AWQ is the instructive contrast, and it is not the same story: it reaches
**within 2% of cuBLAS at M=1** while reading a quarter of the weight bytes. That is
two different limits meeting rather than one shared one — bf16 streams 31 MB at 43%
of peak HBM and is bandwidth-bound, int4 streams 7.9 MB at 11.9% and is
unpack-bound — so the parity does not transfer to a wider weight, and by M=2048 the
gap is 3.8×. It also only holds after a heuristic fix that `--tune` found; see
[the tile heuristic section](#a-second-tile-heuristic-defect-in-w4a16) below.

Accuracy is the other cost: greedy prefix agreement against the bf16 baseline is
0.233 on Qwen3-4B, against 0.617 for fp8 and 0.822 for int8. Two of the three fp4
mantissa states are subnormal, and a 16-element block is a coarse unit to share an
exponent across.

## FP8 W8A8 Fused MoE

`W8A8Fp8MoEMethod` quantises the *activations* as well as the expert weights
(`fused_moe(..., act_fp8=True)`): per-token fp8 before GEMM1, per-row fp8 on the silu
output before GEMM2, both without a host synchronisation, so an MoE layer stays
capturable. Before this, `W8A8Fp8MoEMethod.apply` and `Fp8MoEMethod.apply` were the
same function and the activations were always bf16 — W8A8 was a label, not a path.

Qwen3-30B-A3B geometry (E=128, top_k=8, hidden 2048, moe_intermediate 768), measured on
an NVIDIA H100 80GB HBM3 (torch 2.13.0+cu130 / triton 3.7.1 / python 3.14.7), from
[`bench_fused_moe_h100_20260901.json`](benchmark_logs/bench_fused_moe_h100_20260901.json)
(run with `LITE_LLAMA_AUTOTUNE=0`, so these are the heuristic tiles a user gets
without a tuning cache):

| tokens | fp16 | fp8 W8A16 | **fp8 W8A8** | int8 | int4 |
|---|---|---|---|---|---|
| 1 | 360.7 µs | 365.2 µs | 481.1 µs | 364.3 µs | 366.2 µs |
| 8 | 363.7 µs | 368.5 µs | 481.5 µs | 367.3 µs | 367.2 µs |
| 64 | 531.7 µs | 439.1 µs | 484.8 µs | **398.9 µs** | 630.0 µs |
| 512 | 583.4 µs | 615.5 µs | **477.8 µs** | 576.9 µs | 691.8 µs |
| 4096 | 1573.2 µs | 2301.0 µs | **1469.4 µs** | 2096.7 µs | 2598.7 µs |

Decode and prefill are different operations here and are not averaged into one
speedup. At 1-8 tokens fp8-A8 is the **slowest** row: the quantisation kernels are
pure overhead on a layer that is launch-bound (the `moe_align_block_size` ablation
alone accounts for ~188 µs of every row, over half the decode time, and it is why
fp16 and all three weight-only formats land inside 1.5% of each other there while
reading 4× different weight bytes). At 64 tokens the weight-only formats win — int8
by 25.0%, W8A16 fp8 by 17.4% — and int4 does not, at 18.5% *slower* than fp16: that
path is bound by unpacking 8 nibbles per int32, not by traffic, and has the lowest
GB/s in the table while reading the fewest bytes. From 512 tokens on the ordering
inverts: fp8-A8 is the fastest row (18.1% under fp16 at 512, 6.6% at 4096, at
210 TFLOP/s) while W8A16 fp8 and int8 become regressions against plain fp16 (46%
and 33% slower at 4096), because dequantising a weight tile per row-block does not
amortise once the GEMM is compute-bound. The A8 gain is in the MMA, so it appears
exactly where the weight-only gains disappear.

One caveat the table does not show: Triton emits Hopper's fp8 `wgmma` only from
`BLOCK_M >= 64`, which `_launch_config` reaches above 64 tokens. The t1/t8/t64 A8
rows are widening both e4m3 operands to an fp16 `mma.sync`, so they are not
measuring the fp8 tensor cores at all — consistent with the win starting at 512.

### A tile heuristic defect these numbers found

`_launch_config` returned `BLOCK_K = 128 if quant_mode else 32`. Correct about the
memory transaction (an fp16 tile fills one at 32 elements, a byte tile needs 128),
wrong about this layer: an expert GEMM here is 768 wide against a 2048 hidden size,
so a narrow k-tile just multiplies the loop count. A tile sweep over the 17-config
space in `benchmarks/kernels/bench_fused_moe.py --tune` found **no** winning fp16
config with `BLOCK_K` below 64 at any token count, and the narrow tile was costing
25.5% at 64 tokens, 22.6% at 512 and 10.4% at 4096.

The defect only ever depressed the *unquantised baseline*, which is why no test
caught it and why every quantisation comparison on this kernel used to read better
than it was — at 512 tokens it was the difference between W8A16 fp8 looking like an
18% win and the 5.5% loss above. All modes now get 128. The benchmark keeps the
narrow tile as an ablation row so the fix stays measured: the two fp16 rows
converging means it came back.

The same sweep, persisted to `ConfigStore`, improved 13 of 15 store keys over the
heuristic (largest: fp16 at the M512 bucket, 2502.8 → 1694.1 µs, +32.3%). Search is
per `TuneKey`, not per token count, because `bucket_m` rounds M up to the next of
(16, 32, 64, 128, 256, 512) — t1 and t8 share one entry, so a per-token search would
have them overwrite each other. Run `--tune` on a new device; the persisted cache is
not committed.

## A second tile heuristic defect, in w4a16

The dense GEMMs have the same class of problem and only one of them can be fixed
through the cache. Of the five quantised kernels, **`w4a16_matmul` is the only one
that consults `ConfigStore`** — fp8 W8A8, fp8/int8 W8A16 and NVFP4 compute their
launch configuration unconditionally, so `bench_quant_gemm.py --tune` reports them as
having no consumer rather than writing entries no kernel would read. (`v0.5`'s
changelog claims autotune covers "量化 GEMM"; for the dense path that is one kernel of
five.)

On that one kernel the sweep found a heuristic defect rather than a per-shape tuning
opportunity. The `m <= 32` branch used `GROUP_M=1, num_stages=2`, and
`GROUP_M=8, num_stages=4` — the *same* 16×64 tile — won at **all 16** `m <= 32` store
keys (two Qwen3 geometries × four projections × the M16/M32 buckets) by 9.0–41.5%,
with the tile held fixed so those two knobs are the only variables. `GROUP_M=1` groups
nothing, so consecutive programs walk the grid row-major and share no weight tile in
L2; the deeper pipeline then covers a nibble unpack that a 16-row tile cannot hide.
Because the win was uniform, the fix belongs in the kernel's fallback, not in a cache
keyed on shape — it now ships to every device without a tuning run, and it is what
moved the M=1 int4 row above from 34.0 to 22.2 µs.

After the fix, per-shape tuning still has plenty left: 29 of 32 keys improve on the
corrected heuristic, by 9.7–46.0%. Only three report "heuristic already best" — the
M16 keys of `qwen3-4b/qkv`, `qwen3-4b/gate_up` and `qwen3-30b-a3b/qkv`, where 16×64 is
already right. Elsewhere the winners are *narrower* than the heuristic at decode
(16×32 or 64×32 across M16/M32) and much wider at prefill (128×64 through 128×256 at
M512). That spread is what a three-branch fallback cannot cover, and it is why this is
the one dense kernel worth keeping a cache for.

One structural caveat: a shared bucket entry is chosen for the *total* over the token
counts in it, so a width inside a bucket can regress while the entry is still a net
win. Spot-checking the M512 entry on `qwen3-30b-a3b/qkv` and `qwen3-4b/qkv`, both
widths improved on both keys (t512 +0.7% / +12.2%, t2048 +25.5% / +24.3%), so no
regression was observed here — but a decode-only deployment should still narrow
`--tokens` to the widths it serves rather than inherit a prefill-weighted entry.

The fix is worth 1.32× end to end on the configuration that exposes it: Qwen3-4B
int4 with decode graphs went 419.0 → 551.3 TPS (TPOT 9.28 → 6.97 ms) at tp1, 471.9 →
583.1 at tp2 and 816.8 → 1057.6 at dp2, with the greedy match rate unchanged at 0.157
as a tile change must leave it. The eager rows moved by 2-4%, which is the useful
control: with no graph, launch overhead dominates decode and a better tile has
nothing to reveal. Re-measured rows are in
[`quant_matrix_20260901.md`](benchmark_logs/quant_matrix_20260901.md) §2.

## Tensor Parallelism × CUDA Graphs

Decode graphs used to be refused whenever `tp_world_size > 1`. They are captured now,
behind three gates in `ModelRunner.enable_cuda_graph`, because a wrong graph under TP
fails by hanging in a collective rather than by raising:

| Gate | What it checks | On failure |
|---|---|---|
| `LITE_LLAMA_TP_CUDA_GRAPH=0` | kill-switch | eager, with a warning |
| grid agreement | all-reduce of `(len(batch_sizes), len(seq_len_buckets), hash(grid))` | **every** rank drops its graphs |
| numerical | graph vs eager output per `(bs, bucket)`, atol 1e-2 | drop graphs, fall back to eager |

Capture also needs `NCCL_GRAPH_MIXING_SUPPORT=1` (set in `init_parallel`) because
prefill stays eager while decode replays, mixing captured and uncaptured collectives
on one communicator, and a `warmup_collectives()` pass so no communicator is lazily
created inside the capture region.

Measured on 2× NVIDIA H100 80GB HBM3 (torch 2.13.0+cu130 / triton 3.7.1, 2026-09-01)
with Qwen3-4B-Thinking-2507, `fp8+tp2+graph`: 77 replays, weights
2.06 GB per rank (against 4.11 GB at tp1), KV capacity 955,832 tokens (against
465,750) — the freed weight memory becomes cache. Throughput is *lower* than tp1 for
this model (622 vs 664 tok/s): at 4B the per-step all-reduce costs more than the
second card's compute buys. TP here is a capacity feature, not a speed feature.

### Which engine can run TP

`LLM` (and therefore `TextGenerator`) cannot: its generate loop broadcasts sampled
tokens but never the step plan, so follower ranks would wait forever. It now raises
instead of accepting `tensor_parallel_size > 1` and silently running on one GPU —
which is what it did before, and what made a benchmark row labelled `tp2` a tp1
measurement. Use `ContinuousBatchingEngine.from_pretrained(...)`, `lite-llama serve`,
or `lite-llama batch`, whose executor broadcasts each step's plan.

## Accuracy

Token-level accuracy comparison on Qwen3-0.6B (A10, greedy decode, same prompt set and
same 2026-08-23 run as the [Performance Benchmark](#performance-benchmark) table):

| Scheme | Token Match vs HF fp16 | Expected |
|---|---|---|
| lite fp16 | ~25% | Normal — different attention kernel numerics cause divergence after first mismatch |
| int8 per-channel | ~23% | Within fp16 divergence range |
| fp8 W8A8 | ~5% | e4m3's 3 mantissa bits cause earlier divergence |

> **Logits-level accuracy** (measured per-token before greedy argmax) shows much higher
> agreement: int8 <0.03% relative error, fp8 <0.04% — confirming the quantisation does
> not degrade model quality, only greedy decode amplifies rounding differences.

## FP8 KV Cache: Why `k_scale = v_scale = 1.0`

`Fp8KVCacheMethod` ships fixed unit scales. That looks like an omission, so it was
measured rather than argued about: `scripts/quant_kv_error.py` on Qwen3-4B-Thinking-2507
(36 layers; NVIDIA H100 80GB HBM3, torch 2.13.0+cu130 / python 3.14.7), full log in
[`kv_fp8_error_qwen3-4b_20260901.json`](benchmark_logs/kv_fp8_error_qwen3-4b_20260901.json).

| Measurement | Result | Reads as |
|---|---|---|
| Largest \|K\|/\|V\| seen, any layer | 294.0 (`layers.0…attn.k`) | Below e4m3's 448 |
| Values clamped at 448 | **0 / 47,112,192** | Unit scale clips nothing |
| Non-finite values written | 0 | — |
| Mean relative RMS error at scale 1.0 | 2.66e-02 | The cost of 3 mantissa bits |
| Mean relative RMS with a per-call *oracle* scale | 2.59e-02 | **1.030× better** |
| Best single layer with the oracle scale | 1.185× | One layer, not the model |
| Largest subnormal share | 56.7% (`layers.0…attn.v`) | Where a scale *could* help |
| Greedy token agreement, `auto` vs `fp8_e4m3` | **0.3164** (162/512) | Real divergence |
| Same, `auto` vs `auto` (control) | 1.0000 | The divergence is fp8's, not noise |
| GSM8K exact match, 500 questions | 0.1920 → 0.1640 (**−2.8 pp**) | 1.96·se = 4.74 pp → **not resolved** |

Three findings that do not all point the same way, kept that way on purpose:

1. **Nothing is being clipped.** A per-tensor scale changes an e4m3 result in exactly two
   places: above 448, where it clamps, and near 2⁻⁶, where values go subnormal and lose
   mantissa bits. e4m3 is a *floating* format — 3 mantissa bits means ~6% relative spacing
   in every binade — so multiplying by a constant mostly just slides values along the
   exponent axis without changing their relative error. This is the opposite of int8, where
   the step size is absolute and calibration is unconditionally worth doing.
2. **The divergence is real.** 0.3164 token agreement against a 1.0000 same-dtype control
   rules out benchmark noise: fp8 KV does change what the model says. Greedy decoding is
   chaotic, so one flipped argmax rewrites the rest of the completion — first divergence
   lands at token 22 / 11 / 74 / 33 of 128.
3. **Calibration is not the fix.** The oracle scale here is `amax/448` computed *per call*,
   i.e. an optimistic upper bound on any static per-tensor calibration — no offline scale
   can beat a scale chosen with the data in hand. It buys 1.030× on the mean over layers.
   The decision uses the mean, not the best layer: every layer feeds the same residual
   stream, so what reaches the logits is the aggregate error. A single layer at 1.185× is
   evidence that one layer's activations sit further from a power of two, not that
   calibration helps.

**Conclusion:** the error is the format, not the scale. Offline calibration
(`scripts/calibrate_kv_scale.py`, per-layer per-tensor) was therefore **not** implemented —
it would remove ~3% of a 2.7% error. What is *not* claimed is that fp8 KV cache is free:
it demonstrably moves tokens, and the GSM8K arm came out 2.8 pp lower. At 500 questions
that is inside noise (unpaired 1.96·se = 4.74 pp), so the honest statement is that the task
effect is **unmeasured at this sample size**, not that it is zero. The 56.7% subnormal share
in the V tensors is the one place a scale *below* 1.0 could still buy something, and is
where to look first if a future model does show a task regression.

Throughput note from the same run: fp8 KV decoded at 1952 tok/s against 2297 tok/s for
bf16 KV. The read path is dequant-bound, not bandwidth-bound (see
`benchmarks/kernels/bench_paged_decode.py`), so halving the bytes does not halve the time —
fp8 KV buys **capacity**, roughly 2× the cached tokens, not speed.

Reproduce:

```bash
python scripts/quant_kv_error.py --model-dir $LITE_LLAMA_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
    --max-gen-len 128 --gsm8k 500 --json docs/benchmark_logs/kv_fp8_error.json
```

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
