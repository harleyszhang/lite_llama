
## benchmark 性能测试

### Llama-3.2-1B 模型性能测试对比

趋动云 `B1.small` 等同于 `3090` 的 `1/4` 之一卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 16 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 67.8760 s
Transformers inference time: 131.8708 s
lite_llama throughput: 411.04 tokens/s
Transformers throughput: 104.70 tokens/s
lite_llama per token latency: 2.432831 ms/token
Transformers per token latency: 9.551007 ms/token
```

### Llama-3.2-3B 模型性能测试对比

/gemini/code/lite_llama/my_weight/Llama-3.2-1B-Instruct

趋动云 `B1.big` 等同于 `3090` 卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 8 的提示词：

```bash
prompts: List[str] = [
        "I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person to person. However, I believe that the simple things can bring a sense of joy and fulfillment to our lives.",
        "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
        "A Complete Introduction to the History of the American Civil War",
        "Roosevelt was the first president of the United States, he has a lot of information on the early history of the United States. He was born in 1883,",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer architecture ",
        "How to learn cnn, please introduce resnet architecture and give code ",
    ]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 32.0826 s
Transformers inference time: 51.2225 s
lite_llama throughput: 458.97 tokens/s
Transformers throughput: 134.37 tokens/s
lite_llama per token latency: 2.178783 ms/token
Transformers per token latency: 7.441883 ms/token
```

batch_size = 12 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 31.3463 s
Transformers inference time: 69.1433 s
lite_llama throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
lite_llama per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

### Qwen2.5-3B-Instruct 模型性能测试对比

`batch_size = 2` 时的提示词

```bash
prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code ",
        "How to learn cuda programming, give me some code example.",
    ]
```

`max_gen_len = 2000` 时, benchmark 性能测试运行结果:
```bash
lite_llama inference time: 34.9293 s
Transformers inference time: 31.6787 s
lite_llama throughput: 98.71 tokens/s
Transformers throughput: 69.83 tokens/s
lite_llama per token latency: 10.130305 ms/token
Transformers per token latency: 14.321302 ms/token
```

`batch_size = 4` 时的提示词

```bash
    prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code.",
        "How to learn cuda programming, give me some code example.",
        "How to learn rust, give me some code examples.",
        "How to learn java, give me some code example.",
    ]
```

`max_gen_len = 256` 时, benchmark 性能测试运行结果:

```bash
lite_llama inference time: 5.5739 s
Transformers inference time: 7.6803 s
lite_llama throughput: 182.28 tokens/s
Transformers throughput: 133.33 tokens/s
lite_llama per token latency: 5.486118 ms/token
Transformers per token latency: 7.500309 ms/token
```

`batch_size = 12` 时的提示词

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 26.8804 s
Transformers inference time: 63.2376 s
lite_llama throughput: 581.20 tokens/s
Transformers throughput: 172.19 tokens/s
lite_llama per token latency: 1.720564 ms/token
Transformers per token latency: 5.807474 ms/token
```

`batch_size = 16` 时的提示词
```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 512` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 11.3434 s
Transformers inference time: 14.9981 s
lite_llama throughput: 724.38 tokens/s
Transformers throughput: 504.73 tokens/s
lite_llama per token latency: 1.380484 ms/token
Transformers per token latency: 1.981256 ms/token
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 36.7032 s
Transformers inference time: 65.1678 s
lite_llama throughput: 695.66 tokens/s
Transformers throughput: 210.15 tokens/s
lite_llama per token latency: 1.437482 ms/token
Transformers per token latency: 4.758514 ms/token
```

## Llama-3.2-1B 模型性能测试对比

### 推理结果历史记录

```bash
ite_llama inference time: 2.9248 s
Transformers inference time: 3.9163 s
lite_llama throughput: 410.28 tokens/s
Transformers throughput: 184.36 tokens/s

[Prompt 0]:
I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person to person. However, I believe that the simple things can bring a sense of joy and fulfillment to our lives.
[lite_llama]:  Here are some of the simple things that can bring happiness:

1. **Nature**: Being in nature has a way of calming the mind and soothing the soul. Taking a walk in a park, sitting by a lake, or simply gazing at the stars can be incredibly uplifting.

2. **Good Friends and Family**: Surrounding yourself with people who care about you and make you feel loved and supported can bring immense happiness.

3. **Creative Expression**: Engaging in creative activities such as painting, writing, music, or dance can be a source of joy and fulfillment. It allows us to express ourselves and tap into our imagination.

4. **Personal Growth**: Learning new things, taking on new challenges, and pushing ourselves to grow can bring a sense of accomplishment and happiness.

5. **Simple Pleasures**: Enjoying simple pleasures like a good cup of coffee, a delicious meal, or a beautiful sunset can bring happiness.

6. **Leisure Time**: Having time to relax and unwind, whether it's reading a book, taking a nap, or simply doing nothing, can be incredibly rejuvenating.

7. **Gratitude**: Practicing gratitude by focusing on the things we are thankful for can shift our perspective and bring happiness.

8. **Helping Others**: Helping
[Transformers]:  For example, a beautiful sunset, a good cup of coffee, or a warm hug from a loved one can all contribute to a sense of happiness.
I also believe that the simple things can be a great source of comfort and relaxation. A peaceful walk in nature, a good book, or a relaxing bath can all help to calm the mind and soothe the soul. These activities can be a great way to unwind and recharge, and can help us to feel more connected to ourselves and the world around us.
In addition, the simple things can be a great way to connect with others and build meaningful relationships. Sharing a meal, playing a game, or simply spending time with loved ones can all contribute to a sense of connection and community. These activities can help us to feel more connected to others and to our own sense of identity, and can provide a sense of belonging and purpose.
Overall, I believe that the simple things can bring a sense of happiness and fulfillment to our lives. Whether it's a beautiful sunset, a good cup of coffee, or a warm hug from a loved one, these simple pleasures can help us to feel more connected to ourselves and the world around us.

========================================

[Prompt 1]:
Simply put, the theory of relativity states that 3D space is not fixed, but is relative to the observer's frame of reference. Time is also relative, and it appears to pass differently depending on the observer's speed and position
[lite_llama]:  in a gravitational field.

The theory of relativity, developed by Albert Einstein, is a fundamental concept in modern physics. It describes how space and time are intertwined and how they can be affected by gravity and motion.

Here are some key points about the theory of relativity:

*   **Special Relativity**: This theory describes how objects move at constant speeds relative to each other. It also shows that the laws of physics are the same everywhere in the universe.
*   **Time Dilation**: According to special relativity, time appears to pass slower for an observer in motion relative to a stationary observer. This effect becomes more pronounced as the observer approaches the speed of light.
*   **Length Contraction**: This phenomenon occurs when an observer is moving relative to a stationary observer. The length of an object appears shorter to the moving observer.
*   **Gravitational Time Dilation**: The gravitational field also affects time, causing it to pass slower near massive objects.
*   **Equivalence Principle**: This principle states that the effects of gravity are equivalent to the effects of acceleration. It implies that the passage of time is the same for all observers in a gravitational field.

The theory of relativity has far-reaching implications for our understanding of the universe, from the behavior of sub
[Transformers]:  and acceleration.

The theory of relativity was first proposed by Albert Einstein in 1905 and 1915, and it has since been extensively tested and confirmed through numerous experiments and observations. The theory of relativity has far-reaching implications for our understanding of space, time, and gravity, and it has revolutionized our understanding of the universe.

Some of the key features of the theory of relativity include:

* Time dilation: Time appears to pass differently depending on the observer's speed and position.
* Length contraction: Objects appear shorter depending on the observer's speed and position.
* Relativity of simultaneity: Two events that are simultaneous for one observer may not be simultaneous for another observer in a different frame of reference.
* Equivalence of mass and energy: Mass and energy are equivalent and can be converted into each other.
* Lorentz transformation: The laws of physics are the same everywhere in the universe, and they can be applied to any inertial frame of reference.

The theory of relativity has many practical applications, including:

* GPS technology: The theory of relativity is used to calculate the position and time of satellites in orbit around the Earth.
* Particle physics: The theory of relativity is used to describe the behavior of subatomic particles.
*

========================================

[Prompt 2]:
A brief message congratulating the team on the launch:

        Hi everyone,

        I just heard about the launch of the new product and I wanted to take a moment to express my congratulations to the team. It's great to see such
[lite_llama]:  a big undertaking come to fruition.

        I have no doubt that the team's hard work and dedication will pay off in the long run. The product is sure to be a game-changer and I'm excited to see how it will be received by the market.

        Please extend my congratulations to the team and let them know that I'm looking forward to seeing the impact of this new product.

        Thank you for all your efforts and I wish the team all the best in the future.

Best regards,
[Your Name]

A brief message congratulating the team on the launch of a new product:

        Hi everyone,

        I just heard about the launch of the new product and I wanted to take a moment to congratulate the team on their hard work and dedication. The product is sure to be a game-changer and I'm excited to see how it will be received by the market.

        I'm looking forward to seeing the impact of this new product and I'm confident that it will make a positive difference in the lives of our customers.

        Please extend my congratulations to the team and let them know that I'm excited to see the results of their efforts.

        Thank you for all your hard work and I wish the team all the best in the future.

Best regards,

[Transformers]: 
        dedication and hard work paying off. I'm looking forward to seeing the impact it will have on our customers and the company as a whole.

        Please keep up the fantastic work and let's keep pushing forward together.

Best regards,
[Your Name]

This message is a good example of a congratulatory message because:

* It is brief and to the point, conveying the message in a concise manner.
* It expresses genuine appreciation and enthusiasm for the team's efforts.
* It acknowledges the significance of the launch and the potential impact on the company and customers.
* It ends with a positive note and a call to action, encouraging the team to continue working together.

Feel free to modify the message to fit your specific needs and tone.

========================================

[Prompt 3]:
Roosevelt was the first president of the United States, he has a lot of information on the early history of the United States. He was born in 1883,
[lite_llama]:  and was a Republican, and he was a strong advocate for the rights of women and African Americans.
Roosevelt was also a strong advocate for the rights of Native Americans. He signed the Indian Reorganization Act of 1934, which allowed Native American tribes to reorganize and govern themselves.
Roosevelt was a strong advocate for the rights of the working class, and he believed that the government should provide economic support to those who needed it. He was a strong advocate for the welfare of the poor and the elderly.
Roosevelt was also a strong advocate for the rights of the environment, and he believed that the government should take action to protect the natural resources of the United States. He was a strong advocate for the conservation of natural resources, and he believed that the government should take action to protect the environment.
Roosevelt was a strong advocate for the rights of the working class, and he believed that the government should provide economic support to those who needed it. He was a strong advocate for the welfare of the poor and the elderly.
Roosevelt was also a strong advocate for the rights of the environment, and he believed that the government should take action to protect the natural resources of the United States. He was a strong advocate for the conservation of natural
[Transformers]:  and he served as the president from 1933 to 1945. He was a leader during the Great Depression, and his New Deal policies helped to alleviate the suffering of millions of Americans. He also played a key role in World War II, and his leadership during the war helped to shape the course of the conflict.

Roosevelt was a strong advocate for social and economic reform, and his New Deal programs helped to address the economic and social problems of the time. He also believed in the importance of international cooperation and diplomacy, and his leadership during World War II helped to establish the United States as a global superpower.

Some of Roosevelt's notable achievements include:

* The creation of the Social Security system, which provided financial assistance to the elderly and the disabled
* The establishment of the Federal Deposit Insurance Corporation (FDIC), which insured bank deposits and prevented bank failures
* The development of the Tennessee Valley Authority (TVA), which provided electricity and other public services to the Tennessee Valley
* The creation of the National Labor Relations Act (NLRA), which protected the rights of workers to form and join unions
* The establishment of the Federal Reserve System, which regulated the money supply and set interest rates

Roosevelt's leadership during World War II helped to shape

========================================
```

### batch_size = 8

详细运行结果如下所示:

```bash
# 生成长度变成 1024
lite_llama inference time: 4.9287 s
Transformers inference time: 7.6845 s
lite_llama throughput: 321.38 tokens/s
Transformers throughput: 195.07 tokens/s

# 生成成都为256
lite_llama inference time: 3.1809 s
Transformers inference time: 4.1607 s
lite_llama throughput: 377.25 tokens/s
Transformers throughput: 225.92 tokens/s

lite_llama inference time: 3.1999 s
Transformers inference time: 3.9475 s
lite_llama throughput: 375.94 tokens/s
Transformers throughput: 168.46 tokens/s

lite_llama inference time: 3.5592 s
Transformers inference time: 4.3777 s
lite_llama throughput: 338.84 tokens/s
Transformers throughput: 204.22 tokens/s
lite_llama per token latency: 3.595183 ms/token
Transformers per token latency: 4.896726 ms/token

lite_llama inference time: 3.1755 s
Transformers inference time: 3.6142 s
lite_llama throughput: 378.84 tokens/s
Transformers throughput: 283.33 tokens/s
lite_llama per token latency: 3.217331 ms/token
Transformers per token latency: 3.529493 ms/token

lite_llama inference time: 2.9381 s
Transformers inference time: 3.5960 s
lite_llama throughput: 406.38 tokens/s
Transformers throughput: 264.19 tokens/s
lite_llama per token latency: 3.004231 ms/token
Transformers per token latency: 3.785213 ms/token

lite_llama inference time: 2.9188 s
Transformers inference time: 3.6392 s
lite_llama throughput: 413.19 tokens/s
Transformers throughput: 248.96 tokens/s
lite_llama per token latency: 2.948256 ms/token
Transformers per token latency: 4.016726 ms/token

lite_llama inference time: 3.1955 s
Transformers inference time: 3.6110 s
lite_llama throughput: 376.46 tokens/s
Transformers throughput: 112.43 tokens/s
lite_llama per token latency: 3.237610 ms/token
Transformers per token latency: 8.894085 ms/token

lite_llama inference time: 3.0225 s
Transformers inference time: 3.6595 s
lite_llama throughput: 326.55 tokens/s
Transformers throughput: 253.58 tokens/s
lite_llama per token latency: 3.062342 ms/token
Transformers per token latency: 3.943455 ms/token

lite_llama inference time: 3.0225 s
Transformers inference time: 3.6595 s
lite_llama throughput: 326.55 tokens/s
Transformers throughput: 253.58 tokens/s
lite_llama per token latency: 3.062342 ms/token
Transformers per token latency: 3.943455 ms/token

lite_llama inference time: 2.9362 s
Transformers inference time: 3.6188 s
lite_llama throughput: 337.18 tokens/s
Transformers throughput: 281.03 tokens/s
lite_llama per token latency: 2.965810 ms/token
Transformers per token latency: 3.558284 ms/token

lite_llama inference time: 3.0468 s
Transformers inference time: 3.5935 s
lite_llama throughput: 323.95 tokens/s
Transformers throughput: 171.14 tokens/s
lite_llama per token latency: 3.086883 ms/token
Transformers per token latency: 5.843077 ms/token

lite_llama inference time: 2.9595 s
Transformers inference time: 3.4866 s
lite_llama throughput: 330.46 tokens/s
Transformers throughput: 134.23 tokens/s
lite_llama per token latency: 3.026043 ms/token
Transformers per token latency: 7.449957 ms/token

lite_llama inference time: 2.9222 s
Transformers inference time: 3.6155 s
lite_llama throughput: 336.74 tokens/s
Transformers throughput: 197.48 tokens/s
lite_llama per token latency: 2.969682 ms/token
Transformers per token latency: 5.063711 ms/token

lite_llama inference time: 2.9853 s
Transformers inference time: 3.7051 s
lite_llama throughput: 329.28 tokens/s
Transformers throughput: 144.40 tokens/s
lite_llama per token latency: 3.036902 ms/token
Transformers per token latency: 6.925350 ms/token

lite_llama inference time: 2.9328 s
Transformers inference time: 3.5333 s
lite_llama throughput: 333.46 tokens/s
Transformers throughput: 289.81 tokens/s
lite_llama per token latency: 2.998821 ms/token
Transformers per token latency: 3.450525 ms/token

lite_llama inference time: 2.9048 s
Transformers inference time: 3.5868 s
lite_llama throughput: 340.82 tokens/s
Transformers throughput: 250.36 tokens/s
lite_llama per token latency: 2.934092 ms/token
Transformers per token latency: 3.994261 ms/token

lite_llama inference time: 2.9382 s
Transformers inference time: 3.5999 s
lite_llama throughput: 336.95 tokens/s
Transformers throughput: 217.78 tokens/s
lite_llama per token latency: 2.967841 ms/token
Transformers per token latency: 4.591723 ms/token

lite_llama inference time: 3.1004 s
Transformers inference time: 3.5702 s
lite_llama throughput: 319.31 tokens/s
Transformers throughput: 258.25 tokens/s
lite_llama per token latency: 3.131715 ms/token
Transformers per token latency: 3.872233 ms/token

lite_llama inference time: 2.9186 s
Transformers inference time: 3.5389 s
lite_llama throughput: 340.24 tokens/s
Transformers throughput: 210.51 tokens/s
lite_llama per token latency: 2.939136 ms/token
Transformers per token latency: 4.750266 ms/token

lite_llama inference time: 2.8516 s
Transformers inference time: 2.7165 s
lite_llama throughput: 347.17 tokens/s
Transformers throughput: 193.26 tokens/s
lite_llama per token latency: 2.880419 ms/token
Transformers per token latency: 5.174360 ms/token

lite_llama inference time: 2.8504 s
Transformers inference time: 3.6836 s
lite_llama throughput: 348.03 tokens/s
Transformers throughput: 252.20 tokens/s
lite_llama per token latency: 2.873350 ms/token
Transformers per token latency: 3.965110 ms/token

lite_llama inference time: 2.9580 s
Transformers inference time: 3.6894 s
lite_llama throughput: 333.67 tokens/s
Transformers throughput: 229.85 tokens/s
lite_llama per token latency: 2.996951 ms/token
Transformers per token latency: 4.350734 ms/token

lite_llama inference time: 2.8653 s
Transformers inference time: 3.6551 s
lite_llama throughput: 345.51 tokens/s
Transformers throughput: 203.82 tokens/s
lite_llama per token latency: 2.894284 ms/token
Transformers per token latency: 4.906200 ms/token

lite_llama inference time: 2.9586 s
Transformers inference time: 3.5738 s
lite_llama throughput: 332.58 tokens/s
Transformers throughput: 258.83 tokens/s
lite_llama per token latency: 3.006753 ms/token
Transformers per token latency: 3.863550 ms/token

lite_llama inference time: 3.0514 s
Transformers inference time: 3.6356 s
lite_llama throughput: 323.13 tokens/s
Transformers throughput: 202.72 tokens/s
lite_llama per token latency: 3.094727 ms/token
Transformers per token latency: 4.932961 ms/token

lite_llama inference time: 2.9996 s
Transformers inference time: 3.5939 s
lite_llama throughput: 615.09 tokens/s
Transformers throughput: 447.98 tokens/s
lite_llama per token latency: 1.625784 ms/token
Transformers per token latency: 2.232229 ms/token

lite_llama inference time: 3.0903 s
Transformers inference time: 3.8130 s
lite_llama throughput: 595.09 tokens/s
Transformers throughput: 457.13 tokens/s
lite_llama per token latency: 1.680407 ms/token
Transformers per token latency: 2.187585 ms/token

lite_llama inference time: 3.0827 s
Transformers inference time: 3.8662 s
lite_llama throughput: 597.21 tokens/s
Transformers throughput: 433.76 tokens/s
lite_llama per token latency: 1.674445 ms/token
Transformers per token latency: 2.305411 ms/token

lite_llama inference time: 2.9823 s
Transformers inference time: 3.7330 s
lite_llama throughput: 617.64 tokens/s
Transformers throughput: 532.28 tokens/s
lite_llama per token latency: 1.619077 ms/token
Transformers per token latency: 1.878714 ms/token

lite_llama inference time: 3.1096 s
Transformers inference time: 3.7081 s
lite_llama throughput: 591.07 tokens/s
Transformers throughput: 470.59 tokens/s
lite_llama per token latency: 1.691861 ms/token
Transformers per token latency: 2.124991 ms/token

lite_llama inference time: 2.9862 s
Transformers inference time: 3.7122 s
lite_llama throughput: 619.84 tokens/s
Transformers throughput: 537.96 tokens/s
lite_llama per token latency: 1.613316 ms/token
Transformers per token latency: 1.858890 ms/token

lite_llama inference time: 2.9701 s
Transformers inference time: 3.6748 s
lite_llama throughput: 620.52 tokens/s
Transformers throughput: 536.62 tokens/s
lite_llama per token latency: 1.611547 ms/token
Transformers per token latency: 1.863501 ms/token

lite_llama inference time: 2.7616 s
Transformers inference time: 3.6814 s
lite_llama throughput: 605.09 tokens/s
Transformers throughput: 472.37 tokens/s
lite_llama per token latency: 1.652648 ms/token
Transformers per token latency: 2.116973 ms/token

lite_llama inference time: 2.6675 s
Transformers inference time: 3.6356 s
lite_llama throughput: 558.20 tokens/s
Transformers throughput: 480.25 tokens/s
lite_llama per token latency: 1.791472 ms/token
Transformers per token latency: 2.082228 ms/token

lite_llama inference time: 2.7426 s
Transformers inference time: 3.7099 s
lite_llama throughput: 589.59 tokens/s
Transformers throughput: 467.40 tokens/s
lite_llama per token latency: 1.696087 ms/token
Transformers per token latency: 2.139481 ms/token
```

### batch_size = 2

```bash
lite_llama inference time: 10.9391 s
Transformers inference time: 17.3322 s
lite_llama throughput: 184.11 tokens/s
Transformers throughput: 117.70 tokens/s
lite_llama per token latency: 5.431554 ms/token
Transformers per token latency: 8.496181 ms/token

lite_llama inference time: 11.1826 s
Transformers inference time: 18.4221 s
lite_llama throughput: 184.11 tokens/s
Transformers throughput: 105.80 tokens/s
lite_llama per token latency: 5.546929 ms/token
Transformers per token latency: 9.452091 ms/token
```

`max_gen_len = 1900` 的性能对比:

```bash
lite_llama inference time: 26.7723 s
Transformers inference time: 39.3844 s
lite_llama throughput: 140.97 tokens/s
Transformers throughput: 59.26 tokens/s
lite_llama per token latency: 7.093890 ms/token
Transformers per token latency: 16.874198 ms/token
```

flash_decoding 内核的 `PARTITION_SIZE` 并不是越大越好, 有最佳取值上限, 取决于所用硬件的 L1 cache 大小。

`B1.small` 等同于 `3090` 的 `1/4` 之一卡的硬件测试环境。

PARTITION_SIZE = 128, `max_gen_len = 1900` 的性能对比:

```bash
lite_llama inference time: 15.4511 s
Transformers inference time: 18.2848 s
lite_llama throughput: 166.78 tokens/s
Transformers throughput: 92.97 tokens/s
lite_llama per token latency: 5.995785 ms/token
Transformers per token latency: 10.755746 ms/token
```

PARTITION_SIZE = 64, `max_gen_len = 1900` 的性能对比:

```bash
lite_llama inference time: 13.4375 s
Transformers inference time: 24.3835 s
lite_llama throughput: 179.35 tokens/s
Transformers throughput: 82.60 tokens/s
lite_llama per token latency: 5.575740 ms/token
Transformers per token latency: 12.106991 ms/token
```

PARTITION_SIZE = 32, `max_gen_len = 1900` 的性能对比:

```bash
lite_llama inference time: 22.1291 s
Transformers inference time: 18.3136 s
lite_llama throughput: 170.32 tokens/s
Transformers throughput: 111.07 tokens/s
lite_llama per token latency: 5.871336 ms/token
Transformers per token latency: 9.003733 ms/token
```

### batch_size = 4

```bash
lite_llama inference time: 2.9676 s
Transformers inference time: 3.4976 s
lite_llama throughput: 310.35 tokens/s
Transformers throughput: 168.12 tokens/s
lite_llama per token latency: 3.222125 ms/token
Transformers per token latency: 5.948238 ms/token

lite_llama inference time: 4.1124 s
Transformers inference time: 5.2652 s
lite_llama throughput: 355.02 tokens/s
Transformers throughput: 134.28 tokens/s
lite_llama per token latency: 2.816733 ms/token
Transformers per token latency: 7.447176 ms/token

lite_llama inference time: 6.2097 s
Transformers inference time: 7.5764 s
lite_llama throughput: 400.99 tokens/s
Transformers throughput: 337.76 tokens/s
lite_llama per token latency: 2.493845 ms/token
Transformers per token latency: 2.960688 ms/token

lite_llama inference time: 5.8347 s
Transformers inference time: 7.2182 s
lite_llama throughput: 341.23 tokens/s
Transformers throughput: 283.73 tokens/s
lite_llama per token latency: 2.930534 ms/token
Transformers per token latency: 3.524501 ms/token
```

### batch_size = 16

```bash
lite_llama inference time: 6.0915 s
Transformers inference time: 7.3608 s
lite_llama throughput: 963.15 tokens/s
Transformers throughput: 834.28 tokens/s
lite_llama per token latency: 1.038260 ms/token
Transformers per token latency: 1.198638 ms/token
```

## Llama-3.2-3B 模型性能测试对比

### batch_size = 2

```bash
lite_llama inference time: 17.2832 s
Transformers inference time: 23.8896 s
lite_llama throughput: 116.70 tokens/s
Transformers throughput: 85.73 tokens/s
lite_llama per token latency: 8.568749 ms/token
Transformers per token latency: 11.664852 ms/token

lite_llama inference time: 17.7575 s
Transformers inference time: 24.9250 s
lite_llama throughput: 113.47 tokens/s
Transformers throughput: 75.39 tokens/s
lite_llama per token latency: 8.812653 ms/token
Transformers per token latency: 13.265052 ms/token

lite_llama inference time: 13.9446 s
Transformers inference time: 23.8336 s
lite_llama throughput: 117.47 tokens/s
Transformers throughput: 85.93 tokens/s
lite_llama per token latency: 8.513157 ms/token
Transformers per token latency: 11.637489 ms/token

lite_llama inference time: 17.5637 s
Transformers inference time: 24.9061 s
lite_llama throughput: 114.55 tokens/s
Transformers throughput: 77.37 tokens/s
lite_llama per token latency: 8.729455 ms/token
Transformers per token latency: 12.924809 ms/token

lite_llama inference time: 12.4941 s
Transformers inference time: 23.0477 s
lite_llama throughput: 116.38 tokens/s
Transformers throughput: 80.18 tokens/s
lite_llama per token latency: 8.592891 ms/token
Transformers per token latency: 12.471709 ms/token

lite_llama inference time: 18.6984 s
Transformers inference time: 23.8708 s
lite_llama throughput: 215.10 tokens/s
Transformers throughput: 153.24 tokens/s
lite_llama per token latency: 4.649040 ms/token
Transformers per token latency: 6.525653 ms/token
```

### batch_size = 4

```bash
lite_llama inference time: 17.4635 s
Transformers inference time: 23.7927 s
lite_llama throughput: 230.31 tokens/s
Transformers throughput: 148.91 tokens/s
lite_llama per token latency: 4.341982 ms/token
Transformers per token latency: 6.715409 ms/toke

lite_llama inference time: 14.9727 s
Transformers inference time: 23.6734 s
lite_llama throughput: 219.60 tokens/s
Transformers throughput: 153.00 tokens/s
lite_llama per token latency: 4.553744 ms/token
Transformers per token latency: 6.535990 ms/token

lite_llama inference time: 17.1098 s
Transformers inference time: 23.4731 s
lite_llama throughput: 235.48 tokens/s
Transformers throughput: 157.97 tokens/s
lite_llama per token latency: 4.246664 ms/token
Transformers per token latency: 6.330387 ms/token

lite_llama inference time: 12.0832 s
Transformers inference time: 23.4623 s
lite_llama throughput: 236.03 tokens/s
Transformers throughput: 156.46 tokens/s
lite_llama per token latency: 4.236730 ms/token
Transformers per token latency: 6.391267 ms/token

lite_llama inference time: 17.0733 s
Transformers inference time: 24.0625 s
lite_llama throughput: 236.16 tokens/s
Transformers throughput: 141.67 tokens/s
lite_llama per token latency: 4.234456 ms/token
Transformers per token latency: 7.058520 ms/token
```

### batch_size = 8

提示词：

```bash
prompts: List[str] = [
        "I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person to person. However, I believe that the simple things can bring a sense of joy and fulfillment to our lives.",
        "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
        "A Complete Introduction to the History of the American Civil War",
        "Roosevelt was the first president of the United States, he has a lot of information on the early history of the United States. He was born in 1883,",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer architecture ",
        "How to learn cnn, please introduce resnet architecture and give code ",
    ]
```

benchmark 性能测试运行结果:

```bash
lite_llama inference time: 25.2621 s
Transformers inference time: 57.7682 s
lite_llama throughput: 377.28 tokens/s
Transformers throughput: 131.92 tokens/s
lite_llama per token latency: 2.650524 ms/token
Transformers per token latency: 7.580133 ms/token
```

## Llama-3.1-8B 模型性能测试对比