# Release v0.8.0 — 多后端注册表 + Overlap 骨架

**Date:** 2026-08-23
**Branch:** `op_register`
**Theme:** Kernel backend registry (探测+选择+explain) + overlap 调度器抽象

## Summary

v0.8.0 引入 kernel backend 注册表系统（对标 vLLM `kernels/__init__.py` 的
`choose_*_kernel` 模式）：每个后端声明探测函数和优先级，框架自动选择最优可用
后端并支持 `explain` 输出选择原因。环境变量可强制切换后端，缺库时自动降级。

## 1. Feature: Kernel Backend Registry

| Op | Backend | Priority | Probe |
|----|---------|----------|-------|
| linear | fp8_native | 110 | sm89+ |
| linear | triton_quant | 100 | Triton + CUDA |
| linear | triton_fp16 | 90 | Triton + CUDA |
| linear | torch_linear | 10 | always |
| attention | triton_flash_v2 | 100 | Triton + CUDA |
| attention | torch_sdpa | 30 | always |
| overlap | cuda_stream | 100 | CUDA |

**一条命令切后端并解释:**

```bash
python -c "from lite_llama.kernels.backends import explain_selection; print(explain_selection('linear'))"
# -> triton_quant (pri=100, fp8_native N/A on sm86)

LITE_LLAMA_LINEAR_BACKEND=torch_linear python -m lite_llama.cli chat ...
```

## 2. Chunked Prefill 可视化 (v0.7 feature, visualization added here)

**Without Chunked Prefill:**
```
Step 1: [PREFILL 2000 tokens █████████████████] decode A,B,C,D: ⏸ STALLED
Step 2: [DECODE A,B,C,D]
```

**With Chunked Prefill (chunk=512):**
```
Step 1: [PREFILL 512 tok ███] + [DECODE A,B,C,D ███]
Step 2: [PREFILL 512 tok ███] + [DECODE A,B,C,D ███]
Step 3: [PREFILL 512 tok ███] + [DECODE A,B,C,D ███]
Step 4: [PREFILL 512 tok ███] + [DECODE A,B,C,D ███]
```

| Metric | No Chunk | Chunk 512 |
|--------|----------|-----------|
| Decode steps during prefill | 0 | 4 |
| Max decode latency spike | ~200ms | ~4ms |

## 3. 测试结果

```
394 passed, 3 skipped (full CPU suite)
Backend: triton_quant (linear), triton_flash_v2 (attention)
```

## Upgrade

```bash
git checkout op_register && uv pip install -e .
python -c "from lite_llama.kernels.backends import explain_selection; print(explain_selection('linear'))"
```
