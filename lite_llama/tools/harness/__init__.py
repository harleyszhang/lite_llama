"""Harness tools: run one layer of a model instead of all of it.

The debugging counterpart to :mod:`lite_llama.tools.profiling`. Profiling describes a
model that already runs; a harness makes a layer runnable before the model is — which is
how a new attention variant gets checked on one GPU while the checkpoint it belongs to
needs sixteen.

Usage:
    from lite_llama.tools.harness import HFLayerReference, SingleLayerHarness

    harness = SingleLayerHarness.from_pretrained("my_weight/Qwen3-0.6B", layer_index=0)
    reference = HFLayerReference(harness.config, harness.layer_index, device=harness.device)
    print(harness.run(batch=2, seq_len=64, reference=reference).render())
"""

from .reference import HFLayerReference, hf_decoder_layer_class
from .single_layer import (
    RANDOM_STD,
    Diff,
    LayerReference,
    LayerReport,
    ModuleTimer,
    OpTiming,
    SingleLayerCache,
    SingleLayerHarness,
    dispatched_kernels,
    layer_keys,
)

__all__ = [
    "RANDOM_STD",
    "Diff",
    "HFLayerReference",
    "LayerReference",
    "LayerReport",
    "ModuleTimer",
    "OpTiming",
    "SingleLayerCache",
    "SingleLayerHarness",
    "dispatched_kernels",
    "hf_decoder_layer_class",
    "layer_keys",
]
