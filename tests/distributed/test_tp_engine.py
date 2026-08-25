"""End-to-end tensor parallelism: what two ranks answer, one rank must answer too.

Every other test in ``tests/distributed`` checks one sharded piece against a
reference computed in the same process. This one checks the assembled thing: a
real :class:`~lite_llama.engine.continuous_engine.ContinuousBatchingEngine` with
``tensor_parallel_size=2``, a real follower process, real NCCL collectives, driven
through the public ``generate`` API — the configuration a user actually runs.

Design: each width is measured by a *probe* — a spawned, non-daemonic process
(non-daemonic because it must spawn the follower itself) that builds one engine,
answers a fixed set of prompt layouts, and reports both the text and the facts
about the object it built. Two probes, one per width, and the assertions compare
their reports. The parent process never touches CUDA or ``parallel_state``, so a
crashed rank cannot leak a TP grid into the rest of the session.

One probe collects everything because loading a checkpoint and rendezvousing a
group is the expensive part; the assertions are cheap and read from its report.

Parity is asserted where it *is* an identity. Sharding is exact in exact
arithmetic, but fp16 reduction is not associative: a row-parallel GEMM plus an
all-reduce adds the same products in a different order. On this checkpoint the
prompt "The history of the Roman Empire spans many centuries, and" sits on a
greedy tie fourteen tokens in, and merely running it inside a batch instead of
alone flips it -- at the very same character, with no tensor parallelism in the
picture at all. Demanding byte equality everywhere would assert something about
fp16, not about the shards.

So every prompt is answered twice, in its batch and on its own, and the
single-GPU probe disagreeing with *itself* is what defines the noise floor. A
prompt that answers the same either way is not near a tie, and on those the two
widths must agree byte for byte. On the rest the weaker claim is the one a
sharding bug would still fail: a wrong offset or a leaked row corrupts the
*first* token, so the answers must share a substantial prefix. And the stable
prompts have to be the majority, or the strong assertion would be quietly
vacuous.

Usage:
    pytest tests/distributed/test_tp_engine.py     # skips below 2 GPUs
"""

from __future__ import annotations

import queue as queue_module
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch.multiprocessing as mp

from lite_llama import SamplingParams
from tests.distributed.tp_harness import needs_gpus

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

#: Both probes plus a follower hold their own weights and KV cache, so this is
#: sized to leave the machine room rather than to stress the scheduler.
_KV_TOKENS = 4096
_MAX_SEQ_LEN = 512
_MAX_NUM_SEQS = 8

#: Loading a checkpoint, profiling a cache and rendezvousing a group. Generous on
#: purpose: it exists to turn a wedged rank into a failure instead of a hang.
_PROBE_TIMEOUT_S = 600.0

#: Greedy, no repetition penalty, no early exit: the only thing that may move a
#: token between the two widths is the arithmetic itself.
_GREEDY = SamplingParams(
    temperature=0.0, max_gen_len=24, repetition_penalty=1.0, stop_on_repeat=False
)

#: Prompt layouts, not a golden baseline (``tests/golden`` owns that). What varies
#: here is the *shape* of the batch a plan describes, because that is what the
#: driver broadcasts and what every rank has to derive identically: one row, rows
#: of unequal length in one prefill, and more rows than one prefill group admits.
_CASES: list[tuple[str, list[str]]] = [
    ("single", ["The capital of France is"]),
    ("mixed", ["Hi", "The history of the Roman Empire spans many centuries, and"]),
    (
        "batch6",
        [
            "One plus one equals",
            "The sun rises in the",
            "Water boils at",
            "Machine learning is",
            "The largest planet is",
            "Python is a language that",
        ],
    ),
]


#: Characters two answers must share before a divergence counts as arithmetic
#: noise rather than a broken shard. A wrong offset or an unmasked row corrupts
#: the first token, so any real prefix at all is evidence; this is comfortably
#: more than one token and comfortably less than a short answer.
_MIN_SHARED_PREFIX = 16

#: Fraction of prompts that must be batch-shape stable. Below this the strong
#: parity assertion would cover too little to mean anything.
_MIN_STABLE_FRACTION = 2 / 3

#: Which prompts the online (async) probe serves concurrently, as ``(case, index)``
#: so its answers can be held against the offline ones.
_ONLINE = [("single", 0), ("batch6", 0)]


def _prompt_at(name: str, index: int) -> str:
    return next(prompts for case, prompts in _CASES if case == name)[index]


def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Build one engine of the requested width, answer every case, report the facts.

    Runs in a spawned process, so it takes only picklable arguments and imports
    torch itself. A build failure is reported as a traceback rather than left to
    time out, because a rank that dies during rendezvous takes the group with it.
    """
    try:
        from lite_llama.engine.continuous_engine import ContinuousBatchingEngine

        engine = ContinuousBatchingEngine.from_pretrained(
            model=spec["model"],
            device="cuda:0",
            max_seq_len=_MAX_SEQ_LEN,
            max_gpu_num_blocks=_KV_TOKENS,
            max_num_seqs=_MAX_NUM_SEQS,
            # Eager on both sides: TP decodes eager anyway (a captured graph would
            # replay collectives), and comparing eager against a graph would fold
            # a second variable into a difference. tests/golden owns that one.
            use_cuda_graph=False,
            tensor_parallel_size=spec["tp_size"],
        )
        try:
            model = engine.engine.model_runner.model
            report: dict[str, Any] = {
                # Every prompt twice: in its batch, and by itself. The pair is what
                # separates "this shard is wrong" from "this token is a coin flip".
                "batched": {name: _answer(engine, prompts) for name, prompts in _CASES},
                "alone": {
                    name: [_answer(engine, [prompt])[0] for prompt in prompts]
                    for name, prompts in _CASES
                },
                "executor": type(engine._executor).__name__,
                # Read before shutdown, which reaps them.
                "children": len(mp.active_children()),
                "embed_rows": model.embed_tokens.local_vocab_size,
                "embed_bytes": model.embed_tokens.weight.numel()
                * model.embed_tokens.weight.element_size(),
                "head_rows": model.lm_head.local_vocab_size,
                "vocab_size": model.embed_tokens.vocab_size,
                "tied": model.lm_head.weight is model.embed_tokens.weight,
            }
        finally:
            engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _answer(engine, prompts: list[str]) -> list[str]:
    """Greedy completions for one batch, in submission order."""
    return [output.outputs[0].text for output in engine.generate(prompts, _GREEDY)]


def _async_probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Serve :data:`_ONLINE` concurrently from a two-rank async engine.

    Online serving puts tensor parallelism somewhere the offline path never does:
    :class:`~lite_llama.engine.async_engine.AsyncLLMEngine` steps the engine on a
    background thread, so every plan broadcast and every NCCL collective is issued
    off the main thread while coroutines register requests concurrently. That the
    ranks stay in step there is a separate fact from the arithmetic, and it is what
    a deployment actually exercises.
    """
    try:
        import asyncio

        from lite_llama.engine.async_engine import AsyncLLMEngine

        async def serve() -> list[str]:
            engine = AsyncLLMEngine.from_pretrained(
                spec["model"],
                device="cuda:0",
                max_seq_len=_MAX_SEQ_LEN,
                max_gpu_num_blocks=_KV_TOKENS,
                max_num_seqs=_MAX_NUM_SEQS,
                use_cuda_graph=False,
                tensor_parallel_size=2,
            )
            async with engine:
                streams = [engine.generate_text(prompt, _GREEDY) for prompt in spec["prompts"]]
                return [output.text for output in await asyncio.gather(*streams)]

        answers = asyncio.run(serve())
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", {"answers": answers}))


def _run_probe(
    model_dir: Path,
    tp_size: int,
    target=_probe,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one probe to completion, surfacing its traceback as this test's failure."""
    context = mp.get_context("spawn")
    results = context.Queue()
    spec = {"model": str(model_dir), "tp_size": tp_size, **(extra or {})}
    # Not a daemon: with tp_size > 1 the probe spawns the follower ranks, and a
    # daemonic process is not allowed children.
    process = context.Process(target=target, args=(spec, results), daemon=False)
    process.start()
    try:
        try:
            status, payload = results.get(timeout=_PROBE_TIMEOUT_S)
        except queue_module.Empty:
            pytest.fail(f"tp={tp_size} probe produced nothing in {_PROBE_TIMEOUT_S:.0f}s")
        if status == "error":
            pytest.fail(f"tp={tp_size} probe failed:\n{payload}")
        return payload
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()


@pytest.fixture(scope="module")
def probes(model_dir: Path) -> dict[int, dict[str, Any]]:
    """One report per width. Module-scoped: two checkpoint loads, many assertions."""
    return {tp_size: _run_probe(model_dir, tp_size) for tp_size in (1, 2)}


def _first_difference(want: str, got: str) -> str:
    """Where two completions diverge, with enough either side to recognise it."""
    for position, (a, b) in enumerate(zip(want, got, strict=False)):
        if a != b:
            return (
                f"diverges at character {position}\n"
                f"  tp=1: ...{want[max(0, position - 30) : position + 30]!r}\n"
                f"  tp=2: ...{got[max(0, position - 30) : position + 30]!r}"
            )
    return f"one is a prefix of the other\n  tp=1: {want!r}\n  tp=2: {got!r}"


def _entries() -> list[tuple[str, int]]:
    """Every ``(case, index)`` pair the probes answer, the unit of comparison."""
    return [(name, index) for name, prompts in _CASES for index in range(len(prompts))]


def _shared_prefix(one: str, two: str) -> int:
    for position, (a, b) in enumerate(zip(one, two, strict=False)):
        if a != b:
            return position
    return min(len(one), len(two))


def _stable(probes) -> set[tuple[str, int]]:
    """Entries whose answer does not depend on the batch they rode in.

    Measured on the *single-GPU* probe, so it is a statement about the checkpoint's
    numerics alone: one engine, one device, no collectives, only the reduction
    shape of the GEMM changing. Whatever survives that is not sitting on a tie.
    """
    return {
        (name, index)
        for name, index in _entries()
        if probes[1]["batched"][name][index] == probes[1]["alone"][name][index]
    }


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_two_ranks_answer_exactly_what_one_rank_answers(probes):
    """Where greedy decoding is determinate, sharding must not move a single byte.

    Sharding is meant to be an arithmetic identity: a row-parallel GEMM plus an
    all-reduce computes the same sum as the whole GEMM, and the sampler's
    two-scalar exchange reconstructs the same log-softmax as the full vocabulary.
    Byte equality says so in the way that catches the failures which produce
    *plausible* numbers — an off-by-one shard offset, a mask that lets another
    rank's rows into the sum — and which no relational check would notice.

    Restricted to the entries the single-GPU probe answers identically batched and
    alone: those are the ones where the assertion is about the shards rather than
    about fp16 associativity. Both groupings are then compared, so a bug that only
    shows up in a multi-row prefill has nowhere to hide.
    """
    stable = _stable(probes)
    assert stable, "nothing was batch-shape stable; this assertion would be vacuous"
    for name, index in sorted(stable):
        for grouping in ("batched", "alone"):
            one = probes[1][grouping][name][index]
            two = probes[2][grouping][name][index]
            assert one == two, (
                f"{grouping} case {name!r} prompt {index}: {_first_difference(one, two)}"
            )


@needs_gpus(2)
def test_an_unstable_prompt_still_starts_the_same(probes):
    """A prompt on a tie may end differently, but it may not *begin* differently.

    This is the half of parity that survives without determinism. Every sharding
    bug worth fearing is wrong immediately — the first token is drawn from the
    same logits as the thousandth — so a shared prefix separates "the arithmetic
    reordered" from "a rank read the wrong rows".
    """
    for name, index in sorted(set(_entries()) - _stable(probes)):
        one = probes[1]["batched"][name][index]
        two = probes[2]["batched"][name][index]
        shared = _shared_prefix(one, two)
        assert shared >= _MIN_SHARED_PREFIX, (
            f"case {name!r} prompt {index} diverges after only {shared} characters, "
            f"which is too early to be a tie:\n{_first_difference(one, two)}"
        )


@needs_gpus(2)
def test_most_prompts_are_batch_shape_stable(probes):
    """Guards the strong assertion's reach: ties must be the exception.

    If a regression made this checkpoint broadly non-deterministic, every entry
    would fall out of the stable set and byte parity would stop being checked
    while still passing. So the size of the set is itself asserted.
    """
    entries = _entries()
    stable = _stable(probes)
    unstable = sorted(set(entries) - stable)
    # Printed, not just asserted: how much of the prompt set the byte-parity check
    # actually covers is the thing a reader of a green run wants to know. Visible
    # under `pytest -s`.
    print(f"\nbatch-shape stable: {len(stable)}/{len(entries)}; on a tie: {unstable}")
    assert len(stable) >= _MIN_STABLE_FRACTION * len(entries), (
        f"only {len(stable)}/{len(entries)} prompts were batch-shape stable; unstable: {unstable}"
    )


@needs_gpus(2)
def test_neither_width_answers_nothing(probes):
    """Guards the comparisons above: two empty answers are also byte-identical."""
    for width in (1, 2):
        for grouping in ("batched", "alone"):
            for name, prompts in _CASES:
                answers = probes[width][grouping][name]
                assert len(answers) == len(prompts)
                assert all(answer.strip() for answer in answers), f"tp={width}, {grouping} {name}"


# --------------------------------------------------------------------------- #
# Online serving
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_online_serving_over_two_ranks_answers_what_offline_does(probes, model_dir):
    """An async, two-rank engine must serve concurrent requests, and serve them right.

    Two things are asserted at once because they fail together: that the ranks stay
    in step when the plans are broadcast from a worker thread rather than the main
    one (a desynchronised group hangs, which the probe timeout reports), and that
    what comes out is what the offline path produced for the same prompt.

    Byte equality is demanded only for prompts the offline probes showed to be
    batch-shape stable; the others are held to a shared prefix, since an async
    arrival order groups requests into batches the offline path never formed.
    """
    answers = _run_probe(
        model_dir,
        tp_size=2,
        target=_async_probe,
        extra={"prompts": [_prompt_at(*entry) for entry in _ONLINE]},
    )["answers"]
    assert len(answers) == len(_ONLINE)

    stable = _stable(probes)
    for (name, index), online in zip(_ONLINE, answers, strict=True):
        offline = probes[2]["alone"][name][index]
        assert online.strip(), f"case {name!r} prompt {index} came back empty"
        if (name, index) in stable:
            assert online == offline, (
                f"case {name!r} prompt {index}: {_first_difference(offline, online)}"
            )
        else:
            assert _shared_prefix(offline, online) >= _MIN_SHARED_PREFIX


# --------------------------------------------------------------------------- #
# What sharding is for: the weights actually get smaller
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_two_ranks_hold_half_the_vocabulary_each(probes):
    """The embedding and the head shrink by exactly the TP width.

    This is the *point* of vocabulary parallelism, and the one property token
    parity cannot see: an implementation that replicated the full table on every
    rank and masked at the end would answer identically while saving nothing.
    """
    vocab = probes[1]["vocab_size"]
    assert vocab % 2 == 0, "an odd vocabulary would make every shard assertion vacuous"
    assert probes[1]["embed_rows"] == vocab
    assert probes[2]["embed_rows"] == vocab // 2
    assert probes[2]["head_rows"] == vocab // 2
    assert probes[2]["embed_bytes"] * 2 == probes[1]["embed_bytes"]


@needs_gpus(2)
def test_tying_survives_sharding(probes):
    """A tied head must stay the *same tensor* as the embedding on every rank.

    Both ends own the same vocabulary slice, so tying stays one allocation rather
    than becoming a special case — and if it silently untied, the head would cost
    a second copy of the largest tensor in a small model.
    """
    assert probes[2]["tied"] == probes[1]["tied"]


# --------------------------------------------------------------------------- #
# Process topology: the cost of parallelism is only paid when asked for
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_one_gpu_stays_in_one_process(probes):
    """The single-GPU default must remain debuggable: no processes, no broadcasts.

    A breakpoint in the engine loop is only a breakpoint in the kernel while the
    forward runs in the calling process, which is why ``UniProcExecutor`` exists
    at all rather than TP=1 being a one-rank group.
    """
    assert probes[1]["executor"] == "UniProcExecutor"
    assert probes[1]["children"] == 0


@needs_gpus(2)
def test_two_gpus_cost_exactly_one_extra_process(probes):
    """TP=2 is two ranks, not a driver plus two workers.

    The driver is rank 0 *and* a worker, so the process count is the rank count.
    """
    assert probes[2]["executor"] == "MultiprocExecutor"
    assert probes[2]["children"] == 1
