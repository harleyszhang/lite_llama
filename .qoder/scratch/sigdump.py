"""Dump exact signatures needed to write accurate docstring Usage lines."""
import ast
import re
from pathlib import Path

BASE = Path("/home/honggao/projects/lite_llama")


def sigs(rel, names):
    p = BASE / rel
    if not p.exists():
        print(f"!! missing {rel}")
        return
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name in names:
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if m.name in ("__init__", "generate", "step", "add_request", "chat",
                                  "build", "run", "search", "try_replay", "capture"):
                        args = [a.arg for a in m.args.args]
                        kw = [a.arg for a in m.args.kwonlyargs]
                        d = len(m.args.defaults)
                        print(f"{rel}::{n.name}.{m.name}({', '.join(args)}[kw:{kw}]"
                              f" ndefaults:{d})")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names:
            args = [a.arg for a in n.args.args]
            print(f"{rel}::def {n.name}({', '.join(args)})")


CHECKS = [
    ("lite_llama/engine/async_engine.py", {"AsyncLLMEngine"}),
    ("lite_llama/engine/async_data_parallel.py", {"AsyncDataParallelEngine"}),
    ("lite_llama/engine/continuous_engine.py", {"ContinuousBatchingEngine"}),
    ("lite_llama/engine/data_parallel.py", {"DataParallelEngine"}),
    ("lite_llama/engine/llm_engine.py", {"LLMEngine"}),
    ("lite_llama/engine/llm.py", {"LLM"}),
    ("lite_llama/engine/generator.py", {"TextGenerator", "VisionGenerator"}),
    ("lite_llama/engine/scheduler.py", {"Scheduler", "SchedulerConfig"}),
    ("lite_llama/engine/detokenizer.py", {"IncrementalDetokenizer"}),
    ("lite_llama/engine/prefix_cache.py", {"PrefixCache", "PrefixCache.__init__"}),
    ("lite_llama/engine/stop_criteria.py", {"StopCriteria", "detect_repetition"}),
    ("lite_llama/engine/sampler.py", {"sample_top_p", "apply_repetition_penalty"}),
    ("lite_llama/engine/multimodal.py", {"MultimodalPreparer"}),
    ("lite_llama/engine/dp_load_balancer.py", {"make_load_balancer"}),
    ("lite_llama/executor/cuda_graph.py", {"CUDAGraphManager", "CUDAGraphRunner"}),
    ("lite_llama/executor/model_runner.py", {"ModelRunner"}),
    ("lite_llama/executor/slot_batch.py", {"SlotBatch", "flatten_extend_rows"}),
    ("lite_llama/executor/attention_metadata.py", {"AttentionMetadata"}),
    ("lite_llama/executor/worker.py", {"ModelWorker", "ModelInput"}),
    ("lite_llama/executor/executor.py", {"UniProcExecutor", "launch_tensor_parallel",
                                          "MultiprocExecutor"}),
    ("lite_llama/executor/loader.py", {"DefaultModelLoader"}),
    ("lite_llama/executor/weight_utils.py", {"hf_weights_iterator", "hf_weight_files"}),
    ("lite_llama/executor/overlap.py", {"OverlapPolicy", "StreamPool", "Timeline"}),
    ("lite_llama/kernels/dispatcher/dispatch.py", {"dispatch", "set_perf_provider"}),
    ("lite_llama/kernels/dispatcher/registry.py", {"register", "OpRegistry"}),
    ("lite_llama/kernels/dispatcher/spec.py", {"KernelSpec", "ShapeRequirement"}),
    ("lite_llama/kernels/dispatcher/autotune/searcher.py", {"AutotuneSearcher"}),
    ("lite_llama/kernels/dispatcher/autotune/config_key.py", {"TuneKey"}),
    ("lite_llama/kernels/dispatcher/autotune/config_store.py", {"ConfigStore"}),
    ("lite_llama/kernels/dispatcher/autotune/lookup.py", {"get_best_config"}),
    ("lite_llama/kernels/dispatcher/autotune/frozen.py",
     {"freeze_record", "install_frozen_perf_provider"}),
    ("lite_llama/kernels/backend/probe.py", {"survey", "library_present"}),
    ("lite_llama/kernels/ops/moe/fused_moe.py", {"fused_moe", "moe_align_block_size"}),
    ("lite_llama/kernels/ops/gemm/linear.py", {"linear_torch"}),
    ("lite_llama/kernels/ops/quantization/fp8.py", {"fp8_matmul"}),
    ("lite_llama/kernels/ops/quantization/w8a16.py", {"w8a16_matmul"}),
    ("lite_llama/kernels/ops/quantization/w4a16.py", {"w4a16_matmul"}),
    ("lite_llama/kernels/ops/quantization/w8a8.py", {"smoothquant_matmul"}),
    ("lite_llama/kernels/ops/rope/rope_emb.py", {"rope_emb_forward"}),
    ("lite_llama/kernels/ops/attention/flashattention2_nopad.py",
     {"flash_attention2_nopad"}),
    ("lite_llama/kernels/ops/attention/flashdecoding.py", {"flash_decoding"}),
    ("lite_llama/kernels/ops/kvcache/update_kv_buffer.py", {"update_kv_buffer"}),
    ("lite_llama/kernels/ops/kvcache/update_kv_index.py", {"update_kv_index"}),
    ("lite_llama/kernels/ops/layernorm/skip_rmsnorm.py", {"skip_rmsnorm"}),
    ("lite_llama/kernels/ops/activation/swiglu.py",
     {"swiglu_forward", "swiglu_forward_fused"}),
    ("lite_llama/kernels/ops/embeddings/vocab_embedding.py", {"vocab_parallel_embedding"}),
    ("lite_llama/kernels/ops/interfaces.py", {"LogicalOp"}),
    ("lite_llama/models/config.py", {"ModelConfig"}),
    ("lite_llama/models/registry.py", {"ModelRegistry", "ModelSpec"}),
    ("lite_llama/models/weights.py", {"load_weights", "translate_text_key"}),
    ("lite_llama/models/base.py", {"CausalLM", "DecoderLayer"}),
    ("lite_llama/models/mla_single_layer.py", {"MinimalMlaLayer"}),
    ("lite_llama/modules/attention.py", {"PagedAttention"}),
    ("lite_llama/modules/linear.py", {"ColumnParallelLinear", "QKVParallelLinear"}),
    ("lite_llama/modules/mlp.py", {"FusedMLP"}),
    ("lite_llama/modules/moe.py", {"SparseMoeBlock"}),
    ("lite_llama/modules/rotary_embedding.py", {"RotaryEmbedding", "MRotaryEmbedding"}),
    ("lite_llama/modules/vocab_parallel.py", {"VocabParallelEmbedding", "ParallelLMHead"}),
    ("lite_llama/modules/quantization/__init__.py", {"get_quantization_config",
                                                     "for_runtime_scheme"}),
    ("lite_llama/modules/quantization/base_config.py", {"QuantizationConfig"}),
    ("lite_llama/modules/quantization/utils.py", {"quantize_fp8_per_token"}),
    ("lite_llama/observe/metrics.py", {"Counter", "Gauge", "Histogram", "metrics"}),
    ("lite_llama/observe/trace.py", {"Tracer"}),
    ("lite_llama/platform/interface.py", {"current_platform", "register_platform"}),
    ("lite_llama/platform/spec.py", {"capabilities_match"}),
    ("lite_llama/platform/cuda.py", {"CudaPlatform"}),
    ("lite_llama/tools/harness/single_layer.py", {"SingleLayerHarness"}),
    ("lite_llama/tools/harness/reference.py", {"HFLayerReference"}),
    ("lite_llama/tools/observability/collective_stats.py", {"CollectiveStats"}),
    ("lite_llama/tools/profiling/structure.py", {"print_structure_tree",
                                                 "export_structure_tree"}),
    ("lite_llama/tools/profiling/memory.py", {"compute_memory_budget",
                                              "print_memory_budget"}),
    ("lite_llama/utils/prompt_templates.py", {"get_prompter", "ChatPrompter"}),
    ("lite_llama/entrypoints/api_server.py", {"build_app", "run_server", "OpenAIServer"}),
    ("lite_llama/entrypoints/protocol.py", {"CompletionRequest", "ChatCompletionRequest"}),
]

print("========== SIGNATURES ==========")
for rel, names in CHECKS:
    sigs(rel, names)

print("\n========== __ALL__ ==========")
for rel in [
    "lite_llama/__init__.py",
    "lite_llama/engine/__init__.py",
    "lite_llama/distributed/__init__.py",
    "lite_llama/kernels/__init__.py",
    "lite_llama/kernels/dispatcher/__init__.py",
    "lite_llama/kernels/dispatcher/autotune/__init__.py",
    "lite_llama/kernels/ops/__init__.py",
    "lite_llama/kernels/ops/quantization/__init__.py",
    "lite_llama/kernels/backend/__init__.py",
    "lite_llama/modules/__init__.py",
    "lite_llama/modules/quantization/__init__.py",
    "lite_llama/observe/__init__.py",
    "lite_llama/platform/__init__.py",
    "lite_llama/entrypoints/__init__.py",
    "lite_llama/tools/harness/__init__.py",
    "lite_llama/tools/observability/__init__.py",
    "lite_llama/tools/profiling/__init__.py",
]:
    src = (BASE / rel).read_text(encoding="utf-8")
    m = re.search(r"__all__\s*=\s*\[(.*?)\]", src, re.S)
    if m:
        items = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        print(f"{rel}: {items}")

print("\n========== ARGPARSE FLAGS (benchmarks/examples) ==========")
for rel in sorted(
    list((BASE / "benchmarks").glob("*.py"))
    + list((BASE / "benchmarks/kernels").glob("*.py"))
    + list((BASE / "examples").glob("*.py"))
):
    if "__pycache__" in rel.parts:
        continue
    src = rel.read_text(encoding="utf-8")
    flags = re.findall(r'add_argument\(\s*["\'](--[\w-]+)', src)
    pos = re.findall(r'add_argument\(\s*["\']([\w-]+)["\'],\s*help', src)
    if flags or pos:
        print(f"{rel.relative_to(BASE)}: flags={flags} pos={pos}")
