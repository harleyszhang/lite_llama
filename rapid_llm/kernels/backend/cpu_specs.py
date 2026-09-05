"""CPU dispatch entries; registration does not import PyTorch or Triton."""

from ...platform import CapabilityRequirement
from ..dispatcher import GoldenRecord, KernelSpec, LayoutRequirement, register


def available() -> bool:
    return True


_TARGETS = {
    "attention.prefill": "flash_attention2_no_pad",
    "attention.chunked_prefill": "flash_attention2_chunked",
    "attention.decode": "flash_decoding",
    "attention.mla_prefill": "mla_prefill",
    "attention.mla_decode": "mla_decode",
    "rmsnorm": "skip_rmsnorm",
    "rope": "rope_emb_forward",
    "kv_write": "update_kv_buffer",
    "moe": "fused_moe",
}

for op, target in _TARGETS.items():
    register(
        KernelSpec(
            name=f"cpu/{target}",
            op=op,
            backend="cpu",
            target=f"rapid_llm.kernels.backend.cpu:{target}",
            available="rapid_llm.kernels.backend.cpu_specs:available",
            capability=(CapabilityRequirement("cpu"),),
            dtypes=() if op == "kv_write" else ("fp32", "fp16", "bf16"),
            schemes=("unquantized", "fp8_kv") if op == "attention.decode" else ("unquantized",),
            layout=LayoutRequirement(required=("kv:mla_latent",))
            if "mla_" in op
            else LayoutRequirement(),
            golden=GoldenRecord(verified=True, baseline="PyTorch reference; tests/cpu"),
            priority=1,
            graph_safe=False,
        )
    )
