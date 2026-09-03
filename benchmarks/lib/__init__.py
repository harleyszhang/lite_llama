"""Shared benchmark library — the counterpart of vllm's ``benchmarks/lib``.

Layering, bottom-up (dependencies point one way):

- ``workloads``  what to generate and how to sample
- ``metrics``    the TTFT / TPOT / TPS vocabulary every script reports
- ``backends``   the measured systems, behind one ABC + factory
- ``utils``      GPU hygiene, footprints, JSON logs, table rendering
- ``dp``         data-parallel scaffolding

Scenario scripts import from the package root:

    from benchmarks.lib import BenchResult, make_backend, write_json_log
"""

from .backends import (
    Backend,
    EngineBackend,
    HFBackend,
    LiteBackend,
    VisionBackend,
    VLLMBackend,
    checkpoint_dtype,
    dtype_tag,
    make_backend,
)
from .dp import TimedRow, add_dp_args, measure_dp, print_run_header
from .metrics import BenchResult, RequestRun, print_table, run_requests, steps_to_result
from .utils import (
    count_gen_tokens,
    describe_footprint,
    environment,
    footprint_stats,
    free_gpu,
    gpu_tag,
    measure_generate,
    peak_mem_gb,
    print_row_table,
    report_agreement,
    require_gpus,
    reset_peak_mem,
    timestamped_log_path,
    write_json_log,
)
from .workloads import GREEDY_PARAMS, PROMPTS, SAMPLE_KW, expand_prompts, sampling_params

__all__ = [
    "GREEDY_PARAMS",
    "PROMPTS",
    "SAMPLE_KW",
    "Backend",
    "BenchResult",
    "EngineBackend",
    "HFBackend",
    "LiteBackend",
    "RequestRun",
    "TimedRow",
    "VLLMBackend",
    "VisionBackend",
    "add_dp_args",
    "checkpoint_dtype",
    "count_gen_tokens",
    "describe_footprint",
    "dtype_tag",
    "environment",
    "expand_prompts",
    "footprint_stats",
    "free_gpu",
    "gpu_tag",
    "make_backend",
    "measure_dp",
    "measure_generate",
    "peak_mem_gb",
    "print_row_table",
    "print_run_header",
    "print_table",
    "report_agreement",
    "require_gpus",
    "reset_peak_mem",
    "run_requests",
    "sampling_params",
    "steps_to_result",
    "timestamped_log_path",
    "write_json_log",
]
