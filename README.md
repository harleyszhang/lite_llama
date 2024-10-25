# Pytriton_llama

The llama model inference lite framework by tirton.

## GPU Information

趋动云 GPU 开发环境

```bash
/code# nvidia-smi
Using config file: /etc/orion/env/env.conf
+--------------------------------------------------------------------------------------------+
| ORION-SMI 1.0             Time: 2024-10-11 20:36:48            CUDA Version: N/A           |
+-----------------------------------------------+----------------------+---------------------+
| IP               vGPU Name       Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC|
| pGPU  vGPU       Physical GPU Name            |         Memory-Usage | GPU-Util  Compute M.|
|===============================================+======================+=====================|
| 0.0.0.0          Orion vGPU              Off  |   N/A            Off |                 N/A |
|  0     0         B1.gpu.small                 |      0MiB /  6062MiB |      0%     Default |
+--------------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------------+
| Processes:                                                                     vGPU Memory |
| IP               pGPU  vGPU   PID      Type  Process name                         Usage    |
|============================================================================================|
|  No running processes found                                                                |
+--------------------------------------------------------------------------------------------+
```

cuda 版本：
```bash
# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

Python 3.11.8 包版本:

```bash
torch                          2.1.2
triton                         2.1.0
triton-nightly                 3.0.0.post20240716052845
```

## kernels unit test

```bash
=========================================================================================== test session starts ============================================================================================
platform linux -- Python 3.11.8, pytest-8.3.3, pluggy-1.5.0 -- /root/miniconda3/bin/python
cachedir: .pytest_cache
rootdir: /gemini/code/pytriton_llama
plugins: anyio-4.3.0
collected 41 items                                                                                                                                                                                         

unit_test.py::test_fused_ffn[128-128-64] PASSED                                                                                                                                                      [  2%]
unit_test.py::test_rmsnorm[32-128] PASSED                                                                                                                                                            [  4%]
unit_test.py::test_rmsnorm[32-32] PASSED                                                                                                                                                             [  7%]
unit_test.py::test_rmsnorm[128-128] PASSED                                                                                                                                                           [  9%]
unit_test.py::test_rmsnorm[128-32] PASSED                                                                                                                                                            [ 12%]
unit_test.py::test_rmsnorm[64-128] PASSED                                                                                                                                                            [ 14%]
unit_test.py::test_rmsnorm[64-32] PASSED                                                                                                                                                             [ 17%]
unit_test.py::test_layernorm[32-128] PASSED                                                                                                                                                          [ 19%]
unit_test.py::test_layernorm[32-32] PASSED                                                                                                                                                           [ 21%]
unit_test.py::test_layernorm[32-64] PASSED                                                                                                                                                           [ 24%]
unit_test.py::test_layernorm[128-128] PASSED                                                                                                                                                         [ 26%]
unit_test.py::test_layernorm[128-32] PASSED                                                                                                                                                          [ 29%]
unit_test.py::test_layernorm[128-64] PASSED                                                                                                                                                          [ 31%]
unit_test.py::test_layernorm[64-128] PASSED                                                                                                                                                          [ 34%]
unit_test.py::test_layernorm[64-32] PASSED                                                                                                                                                           [ 36%]
unit_test.py::test_layernorm[64-64] PASSED                                                                                                                                                           [ 39%]
unit_test.py::test_softmax[32-128] PASSED                                                                                                                                                            [ 41%]
unit_test.py::test_softmax[32-32] PASSED                                                                                                                                                             [ 43%]
unit_test.py::test_softmax[32-64] PASSED                                                                                                                                                             [ 46%]
unit_test.py::test_softmax[128-128] PASSED                                                                                                                                                           [ 48%]
unit_test.py::test_softmax[128-32] PASSED                                                                                                                                                            [ 51%]
unit_test.py::test_softmax[128-64] PASSED                                                                                                                                                            [ 53%]
unit_test.py::test_softmax[64-128] PASSED                                                                                                                                                            [ 56%]
unit_test.py::test_softmax[64-32] PASSED                                                                                                                                                             [ 58%]
unit_test.py::test_softmax[64-64] PASSED                                                                                                                                                             [ 60%]
unit_test.py::test_flash_attention_v1[32-128-4-8] PASSED                                                                                                                                             [ 63%]
unit_test.py::test_flash_attention_v1[32-128-8-16] PASSED                                                                                                                                            [ 65%]
unit_test.py::test_flash_attention_v1[32-128-24-32] PASSED                                                                                                                                           [ 68%]
unit_test.py::test_flash_attention_v1[32-128-64-20] PASSED                                                                                                                                           [ 70%]
unit_test.py::test_flash_attention_v1[32-256-4-8] PASSED                                                                                                                                             [ 73%]
unit_test.py::test_flash_attention_v1[32-256-8-16] PASSED                                                                                                                                            [ 75%]
unit_test.py::test_flash_attention_v1[32-256-24-32] PASSED                                                                                                                                           [ 78%]
unit_test.py::test_flash_attention_v1[32-256-64-20] PASSED                                                                                                                                           [ 80%]
unit_test.py::test_flash_attention_v1[64-128-4-8] PASSED                                                                                                                                             [ 82%]
unit_test.py::test_flash_attention_v1[64-128-8-16] PASSED                                                                                                                                            [ 85%]
unit_test.py::test_flash_attention_v1[64-128-24-32] PASSED                                                                                                                                           [ 87%]
unit_test.py::test_flash_attention_v1[64-128-64-20] PASSED                                                                                                                                           [ 90%]
unit_test.py::test_flash_attention_v1[64-256-4-8] PASSED                                                                                                                                             [ 92%]
unit_test.py::test_flash_attention_v1[64-256-8-16] PASSED                                                                                                                                            [ 95%]
unit_test.py::test_flash_attention_v1[64-256-24-32] PASSED                                                                                                                                           [ 97%]
unit_test.py::test_flash_attention_v1[64-256-64-20] PASSED                                                                                                                                           [100%]

============================================================================================ 41 passed in 7.32s ============================================================================================
```

## kernels benchmark test

### softmax

softmax benchmark test result:

![softmax](./benchmark_result/softmax-performance.png)

### linear

linear(matmul) benchmark test result:

![matmul](./benchmark_result/matmul-performance-fp16.png)

### rmsnorm

rmsnorm benchmark test result:

![rms-norm](./benchmark_result/rms-norm-forward.png)

### layernorm

layernorm benchmark test result:

![layer-norm-forward](./benchmark_result/layer-norm-forward.png)

### flashattention

flashattention benchmark test result:

![flashattention benchmark test](./benchmark_result/fused-attention-batch8-head64-d64-fwd-causal=False.png)
![flashattention benchmark test](./benchmark_result/fused-attention-batch4-head32-d64-fwd-causal=False.png)
## Some triton kernels repositories

- https://github.com/ELS-RD/kernl/tree/main
- https://github.com/unslothai/unsloth/tree/main
https://triton-lang.org/main/getting-started/tutorials/