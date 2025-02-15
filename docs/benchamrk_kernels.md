## 一 triton 内核 benchmark 测试

### softmax

softmax benchmark test result:

![softmax](../images/benchamrk_result/softmax-performance.png)

### linear

linear(matmul) benchmark test result:

![matmul](../images/benchamrk_result/matmul-performance-fp16.png)

### rmsnorm

rmsnorm benchmark test result:

![rms-norm](../images/benchamrk_result/rms-norm-forward.png)

### layernorm

layernorm benchmark test result:

![layer-norm-forward](../images/benchamrk_result/layer-norm-forward.png)

### mlp_silu

MLP_Silu test result:

![MLP_Silu ](../images/benchamrk_result/mlp-silu-performance_ret.png)

### flashattention

flashattention benchmark test result:

![flashattention benchmark test](../images/benchamrk_result/fused-attention-batch8-head64-d64-fwd-causal=False.png)
![flashattention benchmark test](../images/benchamrk_result/fused-attention-batch4-head32-d64-fwd-causal=False.png)

### flashattentionv2_no_pad

flashattentionv2_no_pad benchmark test result:

![flashattention benchmark test](../images/flashattention_nopad_benchamrk.png)

### flashdecoding

flashdecoding benchmark test result:

![flashattention benchmark test](../images/flashdecoding_benchamrk.png)

## 二 内核单元测试结果

### triton 内核单元测试

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