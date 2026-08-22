"""Triton kernels used by the lite_llama model implementations.

The public surface is intentionally small: exactly the kernels the model and
engine layers call, plus the standalone activation kernels that are useful on
their own. ``flash_attention2_no_pad`` serves the prefill (context) phase on
variable-length batches, while ``flash_decoding`` serves the single-token decode
phase against the paged KV buffer. ``w8a16_matmul`` and the quantised branch of
``fused_moe`` serve the 8-bit-weight models.
"""

from .activations import gelu, leaky_relu, relu, tanh
from .flashattention2_nopad import flash_attention2_no_pad
from .flashdecoding import flash_decoding
from .fused_moe import fused_moe, moe_align_block_size
from .rope_emb import rope_emb_forward
from .skip_rmsnorm import skip_rmsnorm
from .swiglu import swiglu_forward
from .update_kv_buffer import update_kv_buffer
from .update_kv_index import update_kv_index
from .w8a16 import w8a16_matmul
from .w4a16 import w4a16_matmul
from .smoothquant import smoothquant_matmul

__all__ = [
    "flash_attention2_no_pad",
    "flash_decoding",
    "fused_moe",
    "gelu",
    "leaky_relu",
    "moe_align_block_size",
    "relu",
    "rope_emb_forward",
    "skip_rmsnorm",
    "swiglu_forward",
    "tanh",
    "update_kv_buffer",
    "update_kv_index",
    "w4a16_matmul",
    "w8a16_matmul",
    "smoothquant_matmul",
]
