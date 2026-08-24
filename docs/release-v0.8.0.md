# Release v0.8.0 — 多后端注册表 + Overlap 骨架

**Date:** 2026-08-24
**Branch:** `op_register`
**Theme:** Kernel backend registry (探测 + 选择 + explain + 回退)

## Summary

v0.8.0 引入 kernel backend 注册表系统（对标 vLLM `kernels/__init__.py` 的 `choose_*_kernel` 模式）：每个后端声明探测函数和优先级，框架启动时自动探测硬件/库可用性，选择优先级最高的可用后端，并通过 `explain_selection()` 输出完整决策过程。环境变量可强制切换后端，指定未知后端时自动降级而非崩溃。

## 1. Feature: Kernel Backend Registry

**核心设计（对标 vLLM 的 MMLinearKernel 选择逻辑）：**

```python
@dataclass(frozen=True)
class Backend:
    name: str        # "triton_quant" / "torch_linear" / "fp8_native"
    op: str          # "linear" / "attention" / "overlap"
    priority: int    # 数值越大越优先
    probe: Callable  # 返回 True 表示当前机器可用
    reason: str      # 该后端的硬件/库要求说明
```

**注册的后端：**

| Op | Backend | Priority | Probe |
|----|---------|----------|-------|
| linear | fp8_native | 110 | sm89+ |
| linear | triton_quant | 100 | Triton + CUDA |
| linear | triton_fp16 | 90 | Triton + CUDA |
| linear | torch_linear | 10 | always |
| attention | triton_flash_v2 | 100 | Triton + CUDA |
| attention | torch_sdpa | 30 | always |
| overlap | cuda_stream | 100 | CUDA |

### 可视化：探测、选择、切换、回退

![backend registry](images/backend_registry.gif)

GIF 由 `scripts/gen_backend_registry_gif.py` 驱动**真实 BackendRegistry** 录制，每一行都是本机 `explain_selection()` 的实际输出，逐行展示探测过程。四个场景：

1. `--op linear`：A10 (sm86) 上 `fp8_native` 探测为 N/A，`triton_quant` 按优先级胜出
2. `--op attention`：选中 Triton FlashAttention-2，而非 torch SDPA
3. `LITE_LLAMA_LINEAR_BACKEND=torch_linear`：一个环境变量把箭头钉到 fallback，无需改代码
4. `LITE_LLAMA_LINEAR_BACKEND=cutlass`：未知后端不崩溃，registry 告警并回退到 triton_quant

### 实测输出（A10, sm86）

```
$ python -c "from lite_llama.kernels.backends import explain_selection; print(explain_selection('linear'))"
Backend 'linear' selection:
  [fp8_native] pri=110 N/A (Native fp8 tensor cores (sm89+))
  [triton_quant] pri=100 OK (Triton w8a16/w4a16/w8a8/fp8 quantised GEMM)
  [triton_fp16] pri=90 OK (Triton fp16 GEMM (for unquantised))
  [torch_linear] pri=10 OK (F.linear fallback (always available))
  -> triton_quant
```

**缺库自动回退：** `fp8_native` 在 A10 (sm86) 上探测为 N/A（需 sm89+），自动降级到 `triton_quant`；无 Triton 环境时进一步降级到 `torch_linear`。

## 2. Feature: Overlap 调度器抽象 (L1 骨架)

注册表中新增 `overlap` op 类型，为后续 L1 跨 stream 计算/通信重叠提供探测基础。当前 A10 环境下 `cuda_stream` 后端已就绪 (priority=100)，L1 timeline 实现留待后续版本。

## 3. 测试结果

```
394 passed, 3 skipped    (full CPU suite)
Backend selection on A10: triton_quant (linear), triton_flash_v2 (attention)
```

## 4. 文件清单

| 操作 | 路径 |
|------|------|
| 新建 | `lite_llama/kernels/backends/__init__.py` |
| 新建 | `lite_llama/kernels/backends/registry.py` |
| 新建 | `scripts/gen_backend_registry_gif.py` |
| 新建 | `docs/images/backend_registry.gif` |
| 修改 | `pyproject.toml` (0.7.0 → 0.8.0) |

## Upgrade

```bash
git checkout op_register && uv pip install -e .

# 查看后端选择过程
python -c "from lite_llama.kernels.backends import explain_selection; print(explain_selection('linear'))"

# 强制使用 torch 后端（无 Triton 环境）
LITE_LLAMA_LINEAR_BACKEND=torch_linear python -m lite_llama.cli chat --model-dir my_weight/Qwen3-0.6B
```
