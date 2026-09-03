"""Overlap matrix: feature combinations on one model, then model by model.

Two tables, one discipline — every cell runs the same greedy workload through
the real TP=2 continuous engine, so the numbers stay comparable across rows:

* **Combination matrix** (Qwen2.5-1.5B-Instruct): the eight on/off cells of
  ``LITE_LLAMA_OVERLAP`` (L1, pinned-copy overlap) × ``LITE_LLAMA_TBO`` (L2,
  decode two-batch overlap) × ``LITE_LLAMA_COMM_OVERLAP`` (L3, chunked
  all-reduce). L2 and L3 occupy the *same* row-parallel all-reduce site, so
  when L2 is on, L3 yields — the ``l2l3`` and ``all`` cells exist to *prove*
  that demotion: their completions must match ``l2``'s and ``l1l2``'s bit for
  bit, or the dispatch order has drifted.

* **Model matrix** (baseline vs the recommended overlap mix on each): every
  TP-2 model the release claims support for — dense Qwen2.5, DeepSeek-V2-Lite
  (MLA + MoE), DeepSeek-V3-4layers (biased grouped routing), and the trimmed
  V4 (mHC + compressors + hash MoE, built here from transformers 5.8 with the
  Qwen tokenizer, since V4 ships no public weights). The recommended mix is
  L1+L3 everywhere: M3's PCIe analysis puts L2 under water at these batch
  sizes, and V4's mHC stack does not wire TBO in this version.

L4 (tile-signaling) is a kernel-level primitive with no model-path switch, so
it is not a matrix axis — its own bench (bench_overlap_l4.py) covers it. The
DP + CUDA-graph axis lives in bench_dp_graph.py for the same reason.

Usage:
    python benchmarks/bench_overlap_matrix.py [--json PATH] [--part a|b|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_deepseek_v4 import CONFIG as V4_BODY
from benchmarks.common import (
    PROMPTS,
    BenchResult,
    expand_prompts,
    make_backend,
    report_agreement,
    require_gpus,
    write_json_log,
)

COMBO_CKPT = "my_weight/Qwen2.5-1.5B-Instruct"
MODEL_DIRS = {
    "qwen2.5-1.5b": "my_weight/Qwen2.5-1.5B-Instruct",
    "deepseek-v2-lite": "my_weight/DeepSeek-V2-Lite",
    "deepseek-v3-4layers": "my_weight/DeepSeek-V3-4layers-MTP-BF16",
}
#: The trimmed V4 is built once into this scratch dir (config + random weights
#: + a borrowed tokenizer); ``--v4-rebuild`` forces a fresh seed.
V4_WORKDIR = Path(__file__).resolve().parent / ".v4_matrix_tmp"
TOKENIZER_SRC = "my_weight/Qwen2.5-1.5B-Instruct"
V4_VOCAB = 151936  # the borrowed tokenizer's

L1_ENV, L2_ENV, L3_ENV = "LITE_LLAMA_OVERLAP", "LITE_LLAMA_TBO", "LITE_LLAMA_COMM_OVERLAP"

BATCH = 16
GEN = 64
#: 64-char prompts: prefill keeps L3 above its min_rows, generation is long
#: enough that TPOT — not TTFT — dominates the total.
PROMPT_CHARS = 64


def short_prompts(batch: int) -> list[str]:
    base = expand_prompts(PROMPTS, batch)
    return [prompt[:PROMPT_CHARS] for prompt in base]


def _set_policies(l1: bool, l2: bool, l3: bool) -> None:
    """Flip the three switches and re-read the module-level policies.

    The follower ranks are rebuilt per arm (the executor tears them down with
    the engine), so they pick the environment up at spawn; the rank-0 process
    outlives the arm and needs the explicit resets.
    """
    import os

    os.environ[L1_ENV] = "1" if l1 else "0"
    os.environ[L2_ENV] = "1" if l2 else "0"
    os.environ[L3_ENV] = "1" if l3 else "0"
    from lite_llama.batch_overlap.comm_overlap import reset_comm_overlap_policy
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_comm_overlap_policy()
    reset_tbo_policy()


def measure(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int,
    l1: bool,
    l2: bool,
    l3: bool,
    *,
    max_gpu_num_blocks: int | None = None,
) -> tuple[BenchResult, list[str]]:
    """One matrix cell: the only differences are the three overlap switches."""
    _set_policies(l1, l2, l3)
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_seq_len=2048,
        max_num_seqs=32,
        max_gpu_num_blocks=max_gpu_num_blocks,
    )
    try:
        return backend.measure(prompts, max_gen_len, greedy=True), backend.texts()
    finally:
        backend.close()


# --------------------------------------------------------------------------- #
# Combination matrix: L1 x L2 x L3 on one model
# --------------------------------------------------------------------------- #
COMBOS: list[tuple[str, bool, bool, bool]] = [
    ("baseline", False, False, False),
    ("l1", True, False, False),
    ("l2", False, True, False),
    ("l3", False, False, True),
    ("l1l2", True, True, False),
    ("l1l3", True, False, True),
    # L2 owns the all-reduce site whenever it is on: these two cells must
    # behave exactly like l2 / l1l2 — that equivalence is the test.
    ("l2l3", False, True, True),
    ("all", True, True, True),
]


def run_combination_matrix(model_dir: str, prompts: list[str], gen: int) -> dict:
    results: dict[str, BenchResult] = {}
    texts: dict[str, list[str]] = {}
    for label, l1, l2, l3 in COMBOS:
        print(f"--- {label} (l1={int(l1)} l2={int(l2)} l3={int(l3)})")
        results[label], texts[label] = measure(model_dir, prompts, gen, l1, l2, l3)

    print(f"\n{model_dir}  TP=2  batch={len(prompts)}  gen={gen}")
    for label, result in results.items():
        print(result.row(label))

    base = results["baseline"]
    print("\nvs baseline:")
    for label, result in results.items():
        print(
            f"  {label:9s} TPOT {result.tpot_ms - base.tpot_ms:+7.2f} ms | "
            f"TTFT {result.ttft_ms - base.ttft_ms:+7.1f} ms | "
            f"wall {result.total_s - base.total_s:+6.2f} s"
        )
    report_agreement(texts["baseline"], [(k, v) for k, v in texts.items() if k != "baseline"])
    if texts["l2l3"] != texts["l2"] or texts["all"] != texts["l1l2"]:
        print("!! L3 failed to yield to L2: demotion cells drifted from their plain twins")

    return {
        "tpot_ms": {k: v.tpot_ms for k, v in results.items()},
        "ttft_ms": {k: v.ttft_ms for k, v in results.items()},
        "total_s": {k: v.total_s for k, v in results.items()},
        "demotion_holds": texts["l2l3"] == texts["l2"] and texts["all"] == texts["l1l2"],
    }


# --------------------------------------------------------------------------- #
# Model matrix: baseline vs the recommended mix, model by model
# --------------------------------------------------------------------------- #
def build_v4(workdir: Path) -> Path:
    """Config + random weights + borrowed tokenizer: a servable trimmed V4."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    workdir.mkdir(parents=True, exist_ok=True)
    body = {**V4_BODY, "vocab_size": V4_VOCAB, "tie_word_embeddings": False}
    (workdir / "config.json").write_text(json.dumps(body))
    for leaf in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (workdir / leaf).write_bytes((Path(TOKENIZER_SRC) / leaf).read_bytes())

    torch.manual_seed(0)
    hf = DeepseekV4ForCausalLM(AutoConfig.for_model("deepseek_v4", **body))
    state = {key: value.detach().clone() for key, value in hf.state_dict().items()}
    save_file(state, str(workdir / "model.safetensors"), metadata={"format": "pt"})
    print(f"built trimmed V4 at {workdir}")
    return workdir


def run_model_matrix(prompts: list[str], gen: int, v4_rebuild: bool) -> dict:
    summary: dict[str, dict] = {}
    v4_dir = str(build_v4(V4_WORKDIR) if v4_rebuild or not V4_WORKDIR.exists() else V4_WORKDIR)
    for name, model_dir in [*MODEL_DIRS.items(), ("deepseek-v4-trimmed", v4_dir)]:
        print(f"\n=== {name}: baseline vs l1l3 (recommended mix)")
        blocks = 4096 if name == "deepseek-v4-trimmed" else None
        base_result, base_texts = measure(
            model_dir, prompts, gen, False, False, False, max_gpu_num_blocks=blocks
        )
        mix_result, mix_texts = measure(
            model_dir, prompts, gen, True, False, True, max_gpu_num_blocks=blocks
        )
        print(base_result.row("baseline"))
        print(mix_result.row("l1l3"))
        delta = base_result.tpot_ms - mix_result.tpot_ms
        print(
            f"-> overlap mix changes TPOT by {-delta:+.2f} ms "
            f"({-delta / base_result.tpot_ms:+.1%}), "
            f"TTFT by {mix_result.ttft_ms - base_result.ttft_ms:+.1f} ms"
        )
        report_agreement(base_texts, [("l1l3", mix_texts)])

        summary[name] = {
            "model_dir": model_dir,
            "tpot_ms": {"baseline": base_result.tpot_ms, "l1l3": mix_result.tpot_ms},
            "ttft_ms": {"baseline": base_result.ttft_ms, "l1l3": mix_result.ttft_ms},
            "total_s": {"baseline": base_result.total_s, "l1l3": mix_result.total_s},
            "tpot_change_pct": round(-delta / base_result.tpot_ms * 100, 2),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="default: docs/benchmark_logs/overlap_matrix_<stamp>.json",
    )
    parser.add_argument("--v4-rebuild", action="store_true", help="rebuild the trimmed V4 seed")
    args = parser.parse_args()
    if args.json is None:
        from benchmarks.common import timestamped_log_path

        args.json = str(
            timestamped_log_path(
                Path(__file__).resolve().parent.parent / "docs" / "benchmark_logs",
                "overlap_matrix",
            )
        )

    require_gpus(2)
    print(f"device: 2x {torch_name()}")
    prompts = short_prompts(BATCH)

    results: dict = {}
    if args.part in ("a", "all"):
        print(f"\n########## Part A: combination matrix ({COMBO_CKPT}) ##########")
        results["combination_matrix"] = run_combination_matrix(COMBO_CKPT, prompts, GEN)
    if args.part in ("b", "all"):
        print("\n########## Part B: model matrix (baseline vs l1l3) ##########")
        results["model_matrix"] = run_model_matrix(prompts, GEN, args.v4_rebuild)

    write_json_log(
        args.json,
        {
            "part": args.part,
            "batch": BATCH,
            "gen": GEN,
            "prompt_chars": PROMPT_CHARS,
            "tensor_parallel": 2,
            "cuda_graph": False,
        },
        results,
    )
    return 0


def torch_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


if __name__ == "__main__":
    sys.exit(main())
