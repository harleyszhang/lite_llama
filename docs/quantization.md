# Quantization Support

lite_llama supports multiple weight quantization schemes to reduce memory footprint
and improve inference speed on memory-bound workloads (decode).

## Supported Schemes

| Scheme | Weight | Activation | Scale Granularity | Use Case |
|--------|--------|------------|-------------------|----------|
| **fp8** | fp8-e4m3 | fp16 | 128×128 block | Qwen/DeepSeek FP8 checkpoints |
| **int8** | int8 | fp16 | per-channel | Runtime quantisation of fp16 models |
| **awq/gptq** | int4 | fp16 | group-wise (128) | Pre-quantised AWQ/GPTQ checkpoints |
| **smoothquant** | int8 | int8 (dynamic) | per-channel / per-token | W8A8 inference with int8 tensor cores |

## Quick Start

### FP8 Checkpoints (Qwen3-30B-A3B-FP8)

FP8 checkpoints are detected automatically from `config.json`:

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

The loader reads the `quantization_config` block and builds the model with
8-bit weights + block-wise scales. No conversion step is needed.

### Runtime INT8 Quantisation

Quantise an fp16 checkpoint to int8 at load time:

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen2.5-0.5B --quantization int8
```

This halves the weight memory at a small accuracy cost (~0.1% perplexity).

### Tensor Parallelism (Multi-GPU)

Run a 30B model on two 24 GB cards:

```bash
python -m lite_llama.cli --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --tensor-parallel-size 2
```

Each rank holds a slice of every weight matrix; the only communication is one
all-reduce per transformer block (after `o_proj` and after the MLP/MoE `down_proj`).

## Architecture

```
lite_llama/
├── kernels/
│   ├── w8a16.py          # fp8/int8 weight-only GEMM
│   ├── w4a16.py          # int4 (AWQ/GPTQ) GEMM
│   ├── smoothquant.py    # W8A8 dynamic quantisation GEMM
│   └── fused_moe.py      # MoE grouped GEMM (fp16/fp8/int8)
├── models/
│   ├── quantization.py   # QuantConfig registry + utilities
│   └── linear.py         # ColumnParallelLinear / RowParallelLinear
└── distributed/
    └── parallel_state.py # TP process group
```

### QuantConfig Registry

`QuantConfig.from_hf(hf_config)` reads the checkpoint's `quantization_config`
and returns the appropriate layout. The registry maps `quant_method` strings
to format identifiers:

```python
from lite_llama.models.quantization import QuantConfig, register_quant_method

# Built-in: fp8, int8, gptq, awq, smoothquant
qc = QuantConfig.from_hf(hf_config)

# Register a custom method
register_quant_method("my_quant", "int8")
```

### Linear Layers

Every projection in the decoder is a `LinearBase` subclass that dispatches to
the right kernel based on `quant.format`:

- `fp16` → `F.linear`
- `fp8` / `int8` → `w8a16_matmul`
- `int4` → `w4a16_matmul`
- `smoothquant` → `smoothquant_matmul`

Tensor parallelism is handled by `ColumnParallelLinear` (splits output features)
and `RowParallelLinear` (splits input features + all-reduce).

## Performance

Benchmarks on A10 (24 GB), decode batch size 1:

| Shape (M×N×K) | fp16 (ms) | w8a16 fp8 (ms) | Speedup |
|---------------|-----------|----------------|---------|
| 1×4096×4096 | 0.086 | 0.053 | 1.62× |
| 1×11008×4096 | 0.199 | 0.116 | 1.71× |
| 8×4096×4096 | 0.084 | 0.051 | 1.65× |

The speedup comes from halving the weight bytes streamed from HBM. For prefill
(large M), the kernel is compute-bound and the benefit diminishes.

## Accuracy

Relative error vs fp32 reference (typical):

| Scheme | Relative Error |
|--------|----------------|
| fp8 blockwise | < 0.04% |
| int8 per-channel | < 0.03% |
| int4 group-wise | < 5% |
| smoothquant W8A8 | < 2% |
