import numpy as np
import matplotlib.pyplot as plt

# 原始 logits 值
logits = np.array([2.0, 1.0, 0.5])

# 不同温度值
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

plt.figure(figsize=(12, 8))

for i, T in enumerate(temperatures):
    # 应用温度系数并计算 softmax
    scaled_logits = logits / T
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # 数值稳定
    probs = exp_logits / np.sum(exp_logits)
    
    plt.subplot(2, 3, i+1)
    bars = plt.bar(range(len(probs)), probs, color='skyblue')
    plt.ylim(0, 1.1)
    plt.title(f'T = {T}', fontsize=12)
    plt.xlabel('类别索引', fontsize=10)
    plt.ylabel('概率', fontsize=10)
    
    # 在柱子上方添加概率值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.suptitle('温度系数对 Softmax 概率分布的影响', fontsize=14, y=1.02)
plt.show()