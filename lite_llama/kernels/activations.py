import triton
import triton.language as tl 
import math 

sqrt2 = math.sqrt(2.0)

# 激活函数都是逐元素操作算子，所以无需指定维度参数
@triton.jit
def relu(x):
    """ReLU(Rectified Linear Unit, 修正线性单元), only support inference.
    max(0, x)
    """
    return tl.maximum(0, x)

# Leaky ReLU
@triton.jit
def leaky_relu(x):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    scale = 1e-2
    scale = scale.to(x.dtype)
    return tl.where(x >= 0, x, scale * x)


@triton.jit
def tanh(x):
    """
    Tanh(双曲正切)函数也是一种 Sigmoid 型函数，可以看作放大并平移的 Sigmoid 函数, only support inference.
    2 / (1+e^{-2x}) -1
    """
    return 2 / (1 + tl.exp(-2*x)) - 1

@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU), only support inference."""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x / sqrt2))

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)