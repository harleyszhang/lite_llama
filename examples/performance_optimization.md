## Performance Test

Input prompts：

```bash
prompts: List[str] = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """A brief message congratulating the team on the launch:

    Hi everyone,
    
    I just """,
    # Few shot prompt (providing a few examples before asking model to complete more);
    "Roosevelt was the first president of the United States, he has",
]
```

1. After optimizing the decode phase with cuda graph, the time for a single decode phase is `8.2402` ms, compared to `17.2241` ms before using cuda graph, which is a performance improvement of 2x times, which is almost the same as the performance improvement after applying cuda graph to vllm.

```bash
INFO: After apply cuda graph, Decode inference time: 8.2402 ms
INFO: Before apply cuda graph, Decode inference time: 17.2241 ms
```

2. On the basis of the previous, flashattention has been used to take off the original standard attention.

> flashattention1 is more helpful in training the model, and its speedup effect is limited when the prompt words are very short. The decode phase of inference should be flash-decoding.

```bash
INFO: input tokens shape is  torch.Size([8, 115])
# Before using flashattention
INFO:lite_llama.generate:Batch inference time: 3152.0476 ms
INFO:lite_llama.generate:Tokens per second: 97.71 tokens/s
# After using flashattention
INFO:lite_llama.generate:Batch inference time: 2681.3823 ms
INFO:lite_llama.generate:Tokens per second: 114.87 tokens/s
```

3. Continue optimization by upgrading `flashattention` to `flashattention2` to reduce some computation.

```bash
INFO:lite_llama.generate:Batch inference time: 2103.0737 ms
INFO:lite_llama.generate:Tokens per second: 146.45 tokens/s
```

4. Further optimized by using `flashdecoding` in the decoding phase to improve the parallelism of attention computation during decoding, thereby fully leveraging the GPU's computational power.

```bash
INFO:lite_llama.generate:Decode stage Batch inference time: 1641.4178 ms
INFO:lite_llama.generate:Decode stage tokens per second : 187.64 tokens/s
```

5. Further optimization includes efficient dynamic management of the KV cache (similar to TokenAttention), addressing issues of memory waste and inefficient allocation in KV cache usage.

```bash
INFO:lite_llama.generate:Decode stage Batch inference time: 1413.9111 ms
INFO:lite_llama.generate:Decode stage tokens per second : 217.84 tokens/s
```

6. A simple optimization is to replace the `repeat_kv` function with `GQA_KV_heads_index`.

7. A common and straightforward optimization is the fusion of the key and value linear layers.

8. A commonly used optimization is operator fusion: fusing the residual connection's skip operation with the `rmsnorm` operator to form a new `skip_rmsnorm` operator.

9. Refactored and optimized the `MHA` module, improving the `context_attention` and token_attention kernels to support `Nopad attention` as well as dynamic allocation and management of the `kv cache`.

- token_attention now supports directly passing kv_cache indices and the actual sequence length seq_len, reducing `concat` and `view` operations within the `MHA` module and enabling `Nopad` token_attention.
- During each prefill/decode step, the number of kv_cache indices is dynamically allocated based on the actual prompt length, instead of pre-allocating a continuous kv_cache space for `(max(prompt_len) + max_gen_len) * batch_size` tokens before inference.
