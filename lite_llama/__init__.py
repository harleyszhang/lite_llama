"""Public API: :class:`LLMEngine`, :class:`TextGenerator`, :class:`VisionGenerator`.

Typical usage::

    from lite_llama import TextGenerator, SamplingParams

    gen = TextGenerator(checkpoints_dir="my_weight/Qwen2.5-0.5B")
    print(gen.generate(["What is the capital of France?"], SamplingParams(temperature=0.0)))
"""

from .engine import LLMEngine, Sampler, SamplingParams, TextGenerator, VisionGenerator

__version__ = "0.2.0"

__all__ = [
    "LLMEngine",
    "Sampler",
    "SamplingParams",
    "TextGenerator",
    "VisionGenerator",
    "__version__",
]
