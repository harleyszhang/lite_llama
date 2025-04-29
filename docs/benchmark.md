## benchmark Performance Test

### Llama-3.2-1B Model Performance Comparison Test 

Virtaicloud environment for the `B1.small` equivalent to `1/4` of `3090`. Running the performance test against `python benchmark.py`, lite_llama runs at up to `4x` times the speed of transformers. `batch_size = 16` for the prompter, and `max_gen_len = 1900` for the benchmark performance test results:

```bash
lite_llama inference time: 67.8760 s
Transformers inference time: 131.8708 s
lite_llama throughput: 411.04 tokens/s
Transformers throughput: 104.70 tokens/s
lite_llama per token latency: 2.432831 ms/token
Transformers per token latency: 9.551007 ms/token
```

### Llama-3.2-3B Model Performance Comparison Test 

Virtaicloud environment for the `B1.big` equivalent to `3090`. Running the performance test against `python benchmark.py`, lite_llama runs up to `4x` times faster than transformers. Benchmark performance results with `max_gen_len = 1900`:

```bash
lite_llama inference time: 31.3463 s
Transformers inference time: 69.1433 s
lite_llama throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
lite_llama per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

For more performance test results refer to the documentation [benchmark_models](./docs/benchmark_models.md)(More model performance test results to be updated)
