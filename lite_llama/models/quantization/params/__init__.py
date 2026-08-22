"""Parameter factory & loading strategy per precision format.

Import the public API from here::

    from lite_llama.models.quantization.params import quantize_int8_per_channel
    from lite_llama.models.quantization.params import quantize_fp8_per_channel
"""

from .fp8 import quantize_fp8_per_channel
from .int4 import quantize_int4_groupwise
from .int8 import quantize_int8_groupwise, quantize_int8_per_channel

__all__ = [
    "quantize_fp8_per_channel",
    "quantize_int4_groupwise",
    "quantize_int8_groupwise",
    "quantize_int8_per_channel",
]
