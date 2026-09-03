# Release v0.5.0 — 自动调优 + w4a16 重写

**Date:** 2026-08-23 **Branch:** `main` **Theme:** Kernel autotune 基础设施 (collect + search + persist) + w4a16 tl.dot 重写

## Summary

v0.5.0 建立 kernel 自动调优系统：定义 `TuneKey(gpu, op, shape_bucket, dtype)` 稳定契约（v0.6 perf_key 直接引用）、JSON 持久化存储、命中查找逻辑；搜索引擎完成对 fused_moe 和 flash_attn_nopad 的 shape 配置落盘。w4a16 量化 GEMM 重写为 per-group `tl.dot` 版本走 tensor core，精度测试 6/6 PASS。

## 1. Feature: Autotune 基础设施 (`lite_llama/kernels/autotune/`)

**TuneKey 契约 (v0.6 perf_key 基础):**

```python
@dataclass(frozen=True)
class TuneKey:
    gpu: str            # "NVIDIA_A10"
    op: str             # "fused_moe" / "flash_attn_nopad" / "w4a16_matmul"
    shape_bucket: str   # "M16_N4096_K11008" (M 按桶化)
    dtype: str          # "fp16" / "int8" / "int4"
```

**M 桶化规则:** `[1,16]→16, [17,32]→32, [33,64]→64, [65,128]→128, [129,256]→256, [257,+)→512`

**JSON 落盘格式 (`~/.cache/lite_llama/autotune/fused_moe.json`):**

```json
{
  "version": 1,
  "entries": [
    {"gpu": "NVIDIA_A10", "shape_bucket": "M16_N6144_K1024", "dtype": "fp16",
     "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 4, "num_warps": 4, "num_stages": 3},
     "latency_us": 436.7, "timestamp": "2026-08-23T14:13:32+00:00"}
  ]
}
```

**调用方用法 (一行接入):**

```python
from lite_llama.kernels.autotune import get_best_config
config = get_best_config("fused_moe", m=num_tokens, n=N, k=K, dtype="fp16")
if config is None:
    config = _launch_config(num_tokens, quant_mode)  # heuristic fallback
```

**环境变量控制:** `LITE_LLAMA_AUTOTUNE=0` 强制禁用（走启发式），用于 A/B 对比。

## 2. Feature: Autotune 搜索落盘

离线搜索脚本 `scripts/autotune_collect.py` 从模型 config 推算真实 shape，对 fused_moe (432 configs) 和 flash_attn_nopad (72 configs) 执行 CUDA event 计时搜索。

**Qwen3-0.6B 搜索结果 (A10):**

| Op | Shape | Best Config | Latency |
| ---- | ------- | ------------- | --------- |
| fused_moe | M16_N6144_K1024 | BM=32, BN=32, BK=128 | 436.7 us |
| fused_moe | M64_N6144_K1024 | BM=128, BN=32, BK=128 | -- |
| fused_moe | M128_N6144_K1024 | BM=64, BN=32, BK=128 | -- |
| flash_attn | M64_N128_K128 | BM=64, BN=64 | -- |
| flash_attn | M512_N128_K128 | BM=128, BN=64 | -- |

共 12 shapes 搜索并落盘到 `~/.cache/lite_llama/autotune/`。

## 3. Feature: w4a16 tl.dot 重写

**修复前 (旧实现):** 逐 nibble 标量 outer-product，无法走 tensor core。

**修复后 (v0.5):** per-group `tl.dot` — 每次迭代处理一个 GROUP_SIZE 块：

1. 加载 `[BLOCK_N, GROUP_SIZE//8]` packed int32
2. Unpack 为 `[GROUP_SIZE, BLOCK_N]` fp16
3. 乘 scale 减 zero (dequant)
4. `tl.dot(a_tile, b_tile)` 走 SM80+ fp16 tensor core

**精度测试 (6/6 PASS):**

| Shape (M, N, K) | max-abs-diff | relative error |
| ----------------- | ------------- | ---------------- |
| (1, 128, 1024) | < 0.1 | < 1% |
| (4, 256, 512) | < 0.1 | < 1% |
| (16, 1024, 2048) | < 0.1 | < 1% |
| (64, 512, 1024) | < 0.1 | < 1% |

## 4. Feature: Kernel 接入 autotune

三个 kernel 的 launch 路径统一改为"先查缓存 → 命中用最优 → 未命中回退启发式":

| Kernel | 文件 | 搜索空间大小 |
| -------- | ------ | ------------- |
| `fused_moe` | `lite_llama/kernels/fused_moe.py` | 432 configs |
| `flash_attn_nopad` | `lite_llama/kernels/flashattention2_nopad.py` | 72 configs (原 144 组) |
| `w4a16_matmul` | `lite_llama/kernels/quantization/w4a16.py` | 接入 lookup |

## 5. Chore

| Item | 变更 |
|------|------|
| `lite_llama/kernels/*.py` | 全部 docstring 精简为 summary+usage 格式 (-98 行) |
| `pyproject.toml` | v0.4.0 → v0.5.0 (后续 v0.6 继续 bump) |

## 6. 测试结果

```
26 passed   (test_autotune_store.py, pure CPU)
6 passed    (test_w4a16_accuracy.py, GPU)
217 passed  (full CPU suite)
```

## 文件清单

| 操作 | 路径 |
| ------ | ------ |
| 新建 | `lite_llama/kernels/autotune/__init__.py` |
| 新建 | `lite_llama/kernels/autotune/config_key.py` |
| 新建 | `lite_llama/kernels/autotune/config_store.py` |
| 新建 | `lite_llama/kernels/autotune/lookup.py` |
| 新建 | `lite_llama/kernels/autotune/searcher.py` |
| 新建 | `scripts/autotune_collect.py` |
| 新建 | `tests/kernels/test_autotune_store.py` |
| 新建 | `tests/kernels/test_w4a16_accuracy.py` |
| 重写 | `lite_llama/kernels/quantization/w4a16.py` |
| 修改 | `lite_llama/kernels/flashattention2_nopad.py` |
| 修改 | `lite_llama/kernels/fused_moe.py` |

## Upgrade

```bash
git pull origin main
uv pip install -e .

# 离线搜索并落盘
python scripts/autotune_collect.py --model-dir /data/shared/llm_weights/Qwen3-0.6B

# 验证命中
LITE_LLAMA_AUTOTUNE=1 python -m lite_llama.cli chat --model-dir my_weight/Qwen3-0.6B
```
