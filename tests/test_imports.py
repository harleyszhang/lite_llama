"""Public imports must not eagerly initialise the GPU execution stack."""

from __future__ import annotations

import subprocess
import sys


def _imported_modules(statement: str) -> set[str]:
    script = (
        f"{statement}\n"
        "import sys\n"
        "print('\\n'.join(sorted(name for name in sys.modules "
        "if name.startswith('lite_llama'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def test_sampling_import_does_not_load_executor_or_kernels() -> None:
    modules = _imported_modules("from lite_llama import SamplingParams")

    assert "lite_llama.engine.sampler" in modules
    assert not any(name.startswith("lite_llama.executor") for name in modules)
    assert not any(name.startswith("lite_llama.kernels") for name in modules)


def test_engine_submodule_import_does_not_load_implementations() -> None:
    modules = _imported_modules("from lite_llama.engine.scheduler import Scheduler")

    assert "lite_llama.engine.scheduler" in modules
    assert "lite_llama.engine.llm_engine" not in modules
    assert "lite_llama.engine.continuous_engine" not in modules


def test_model_components_do_not_eagerly_import_triton_kernels() -> None:
    modules = _imported_modules(
        "from lite_llama.modules import RotaryEmbedding, ReplicatedLinear"
    )

    assert "lite_llama.modules.rotary_embedding" in modules
    assert "lite_llama.modules.linear" in modules
    assert "lite_llama.kernels.activations" not in modules
    assert "triton" not in modules
