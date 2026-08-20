## triton 内核 benchmark 测试

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