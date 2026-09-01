"""End-to-end correctness of continuous batching against the one-shot batch path.

The same prompts run through the static engine and the continuous engine
with arrivals spread over time; texts must match, with no foreign tail
leaking between sequences.

Usage:
    pytest tests/engine/test_continuous_batching.py
"""

from __future__ import annotations

import gc
from dataclasses import replace

import pytest
import torch

from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
from lite_llama.engine.llm_engine import LLMEngine
from lite_llama.engine.sampler import SamplingParams
from lite_llama.engine.scheduler import SchedulerConfig

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

# Small enough to keep two engines on one card, long enough that several decode
# steps happen after the first request finishes.
_MAX_SEQ_LEN = 512
_KV_BLOCKS = 8192
_MAX_GEN = 24

# Mixed lengths on purpose: the prefill grid pads to the longest, and the short
# answers finish first, which is what reshapes the batch mid-flight.
PROMPTS = [
    "The capital of France is",
    "List three prime numbers.",
    "Explain in one sentence what a GPU does.",
    "Write a haiku about the sea.",
]

GREEDY = SamplingParams(temperature=0.0, max_gen_len=_MAX_GEN, repetition_penalty=1.0)


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def assert_no_foreign_tail(
    texts: dict[str, str], reference: dict[str, str], tail: int = 25
) -> None:
    """Assert no completion ends up carrying another request's prose.

    The sharp end of every batching bug in this engine looked the same: a request
    kept its own opening and then continued with a neighbour's ending, because
    something indexed the KV cache by batch position instead of by cache slot.
    A prefix comparison misses that -- the opening still matches -- so this looks
    for foreign *endings* instead. Twenty-five characters of model prose is long
    enough to be unique unless the two answers genuinely converge, which is
    allowed for explicitly.
    """
    for prompt, text in texts.items():
        for other, other_reference in reference.items():
            if other == prompt:
                continue
            foreign = other_reference[-tail:]
            if not foreign or foreign in reference[prompt]:
                continue  # the two answers really do end the same way
            assert foreign not in text, f"{prompt!r} emitted {other!r}'s ending {foreign!r}"


@pytest.fixture(scope="module")
def reference(model_dir) -> dict[str, str]:
    """Static-path completions, one prompt at a time.

    Torn down before returning so the continuous engine gets the whole card; a
    second live engine would profile a KV cache of nearly zero blocks.
    """
    engine = LLMEngine(
        str(model_dir),
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_BLOCKS,
        use_cuda_graph=False,
    )
    texts = {
        prompt: LLMEngine.generate_text(
            engine, [engine.tokenizer.encode(prompt, add_special_tokens=True)], GREEDY
        )[0]
        for prompt in PROMPTS
    }
    del engine
    _free()
    return texts


def build_engine(
    model_dir,
    *,
    max_num_seqs=8,
    use_cuda_graph=False,
    max_seq_len=_MAX_SEQ_LEN,
    max_chunk_size=0,
):
    return ContinuousBatchingEngine(
        LLMEngine(
            str(model_dir),
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=_KV_BLOCKS,
            use_cuda_graph=use_cuda_graph,
        ),
        SchedulerConfig(
            max_seq_len=max_seq_len,
            max_num_seqs=max_num_seqs,
            max_chunk_size=max_chunk_size,
        ),
    )


@pytest.fixture(scope="module")
def engine(model_dir):
    built = build_engine(model_dir)
    yield built
    del built
    _free()


def drain(engine) -> None:
    while engine.has_unfinished_requests():
        engine.step()


# --------------------------------------------------------------------------- #
# Parity with the one-shot path
# --------------------------------------------------------------------------- #
def test_a_single_request_matches_the_static_path(engine, reference):
    """Baseline: with nothing else running the two paths must agree exactly."""
    request = engine.add_request(PROMPTS[0], GREEDY)
    drain(engine)
    assert request.text == reference[PROMPTS[0]]


def test_co_tenants_do_not_take_each_others_output(engine, reference):
    """Four requests sharing every step must each answer their own prompt."""
    requests = [engine.add_request(prompt, GREEDY) for prompt in PROMPTS]
    drain(engine)

    texts = {r.prompt: r.text for r in requests}
    assert all(texts.values())
    assert_no_foreign_tail(texts, reference)


def test_survivors_are_byte_identical_when_a_neighbour_leaves(model_dir):
    """The regression that motivated threading ``b_req_idx`` into the decode kernel.

    Two runs prefill the same four prompts together, so prefill is shape-identical.
    In the second run one request stops after three tokens, and the survivors then
    decode as a batch of three instead of four. CUDA graphs are on, so both runs
    still execute decode at the captured width of four -- identical arithmetic,
    which makes a byte-exact comparison the right assertion rather than a
    tolerance.

    Before the fix the survivors shifted position within the batch and began
    attending over the departed request's KV, continuing *its* text instead.
    """
    baseline = build_engine(model_dir, use_cuda_graph=True)
    try:
        kept = [baseline.add_request(prompt, GREEDY) for prompt in PROMPTS]
        drain(baseline)
        expected = {r.prompt: r.text for r in kept}
    finally:
        del baseline
        _free()

    engine = build_engine(model_dir, use_cuda_graph=True)
    try:
        requests = [engine.add_request(prompt, GREEDY) for prompt in PROMPTS]
        early = engine.add_request(
            PROMPTS[1], SamplingParams(temperature=0.0, max_gen_len=3, repetition_penalty=1.0)
        )
        # Submitted after the group above, so it prefills in its own step and
        # leaves the shared decode batch three tokens later.
        drain(engine)

        assert early.finish_reason == "length"
        assert {r.prompt: r.text for r in requests} == expected
    finally:
        del engine
        _free()


def test_staggered_arrivals_keep_every_request_on_its_own_prompt(engine, reference):
    """Requests joining a running batch must not perturb it, or be perturbed.

    Each arrival interrupts the decode batch with a prefill step and then widens
    it, so the batch changes shape repeatedly while everyone is mid-answer.
    """
    requests = [engine.add_request(PROMPTS[0], GREEDY)]
    for step_index, prompt in enumerate(PROMPTS[1:], start=1):
        for _ in range(3 * step_index):
            engine.step()
        requests.append(engine.add_request(prompt, GREEDY))
    drain(engine)

    texts = {r.prompt: r.text for r in requests}
    assert all(texts.values())
    assert_no_foreign_tail(texts, reference)


def test_more_requests_than_slots_are_served_in_waves(model_dir, reference):
    """Queued requests must be admitted as slots free up, with identical output."""
    small = build_engine(model_dir, max_num_seqs=2)
    try:
        requests = [small.add_request(prompt, GREEDY) for prompt in PROMPTS]
        assert small.scheduler.num_waiting > 0, "the queue must actually be exercised"
        drain(small)

        # Prompts admitted two at a time are padded to a different prefill grid
        # than a solo run, so their text is compared for ownership rather than
        # byte equality; what must hold exactly is that everyone got served.
        texts = {r.prompt: r.text for r in requests}
        assert all(texts.values())
        assert all(r.finish_reason for r in requests)
        assert_no_foreign_tail(texts, reference)
        assert small.scheduler.num_free_slots == small.scheduler.num_slots
    finally:
        del small
        _free()


def test_cuda_graph_replay_matches_eager(model_dir, reference):
    """Padding odd batch sizes onto the captured grid must not change results."""
    graphed = build_engine(model_dir, use_cuda_graph=True)
    try:
        # Three requests never match a captured batch size, so every decode step
        # is padded with a filler row pointing at the reserved slot.
        prompts = PROMPTS[:3]
        requests = [graphed.add_request(prompt, GREEDY) for prompt in prompts]
        drain(graphed)

        texts = {r.prompt: r.text for r in requests}
        assert all(texts.values())
        assert_no_foreign_tail(texts, {p: reference[p] for p in prompts})
    finally:
        del graphed
        _free()


# --------------------------------------------------------------------------- #
# Chunked prefill
# --------------------------------------------------------------------------- #
_LONG_PROMPT = PROMPTS[2]  # ~10 tokens: three chunks at max_chunk_size=4
_PREFIX = 40  # chars of opening prose that must survive chunking


def test_a_chunked_prompt_keeps_its_continuation(model_dir, reference):
    """Splitting a prompt across steps must not derail its continuation.

    ``max_chunk_size=4`` forces the prompt through three prefill steps; every
    chunk after the first resumes mid-prompt and runs through the extend rows,
    where each token's query must attend over the already-cached prefix. A
    dropped prefix would change the very first sampled token, so the opening is
    compared against the unchunked reference. Late tokens keep the looser
    ownership check: the chunked and one-shot paths tile their reductions
    differently, so a rare top-2 logit flip is inherent, not a defect.
    """
    chunked = build_engine(model_dir, max_chunk_size=4)
    try:
        request = chunked.add_request(_LONG_PROMPT, GREEDY)
        chunked.step()
        assert 0 < request.num_computed_tokens < request.prompt_len, "must actually chunk"
        drain(chunked)

        expected = reference[_LONG_PROMPT]
        assert request.text[:_PREFIX] == expected[:_PREFIX]
        assert_no_foreign_tail({_LONG_PROMPT: request.text}, {_LONG_PROMPT: expected})
    finally:
        del chunked
        _free()


def test_chunked_prefill_survives_cuda_graph_replay(model_dir, reference):
    """Extend rows are one token wide, so they too can land on a captured graph.

    The filler rows that pad an odd row count onto the captured grid point at
    the reserved slot and carry a fake length; their logits are discarded. The
    continuation must survive that padding unchanged in its opening.
    """
    graphed = build_engine(model_dir, use_cuda_graph=True, max_chunk_size=4)
    try:
        request = graphed.add_request(_LONG_PROMPT, GREEDY)
        drain(graphed)

        expected = reference[_LONG_PROMPT]
        assert request.text[:_PREFIX] == expected[:_PREFIX]
        assert_no_foreign_tail({_LONG_PROMPT: request.text}, {_LONG_PROMPT: expected})
    finally:
        del graphed
        _free()


def test_one_step_carries_prefill_extend_and_decode(model_dir, reference):
    """A resumed chunk, a new prefill and a decode share a single step.

    With A decoding, B mid-prompt and C freshly queued, one step runs all three
    attention shapes: the grid route for C's first chunk, the extend route for
    B's resumed chunk, and a decode for A. Everyone must still answer their own
    prompt, and B's chunked continuation must match the unchunked opening.
    """
    engine = build_engine(model_dir, max_chunk_size=4)
    try:
        a = engine.add_request(PROMPTS[0], GREEDY)  # short: prefill completes in step 1
        b = engine.add_request(_LONG_PROMPT, GREEDY)  # long: chunks across steps
        engine.step()  # both first chunks; A completes, B stays partial
        assert 0 < b.num_computed_tokens < b.prompt_len

        c = engine.add_request(PROMPTS[1], GREEDY)
        engine.step()  # B resumes (extend), C prefills its first chunk (grid), A decodes
        assert c.num_computed_tokens > 0, "C admitted and prefilled in the shared step"
        assert b.num_computed_tokens == 8, "B advanced by exactly one chunk"
        drain(engine)

        texts = {r.prompt: r.text for r in (a, b, c)}
        assert all(texts.values())
        assert all(r.finish_reason for r in (a, b, c))
        assert_no_foreign_tail(texts, reference)
        assert b.text[:_PREFIX] == reference[_LONG_PROMPT][:_PREFIX]
    finally:
        del engine
        _free()


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_per_request_sampling_params_are_honoured(engine, reference):
    """A greedy request must stay greedy while a sampled one shares the batch."""
    greedy = engine.add_request(PROMPTS[0], GREEDY)
    sampled = engine.add_request(
        PROMPTS[1], SamplingParams(temperature=1.5, top_p=0.9, max_gen_len=_MAX_GEN)
    )
    drain(engine)

    assert_no_foreign_tail(
        {PROMPTS[0]: greedy.text}, {PROMPTS[0]: reference[PROMPTS[0]], PROMPTS[1]: sampled.text}
    )
    assert greedy.text, "the greedy request must still produce its own answer"
    assert sampled.text, "the sampled request should still produce something"


def test_the_repetition_penalty_only_sees_generated_tokens(engine):
    """Enabling it must not corrupt a batch whose members are at different lengths."""
    penalised = SamplingParams(temperature=0.0, max_gen_len=_MAX_GEN, repetition_penalty=1.2)
    first = engine.add_request(PROMPTS[0], penalised)
    for _ in range(5):
        engine.step()
    second = engine.add_request(PROMPTS[0], penalised)
    drain(engine)

    # Same prompt, same params: the penalty window is per request, so a request
    # that joined late must reach the same text as the one already running.
    assert first.text == second.text


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_the_length_cap_is_exact(engine):
    request = engine.add_request(
        PROMPTS[0], SamplingParams(temperature=0.0, max_gen_len=5, repetition_penalty=1.0)
    )
    drain(engine)

    assert request.finish_reason == "length"
    assert len(request.output_token_ids) == 5


def test_streamed_deltas_reconstruct_the_final_text(engine):
    request = engine.add_request(PROMPTS[2], GREEDY)
    deltas = []
    while engine.has_unfinished_requests():
        for touched in engine.step():
            deltas.append(touched.delta)

    assert "".join(deltas) == request.text


def test_aborting_frees_the_slot_immediately(engine):
    request = engine.add_request(PROMPTS[0], SamplingParams(max_gen_len=200))
    for _ in range(4):
        engine.step()

    engine.abort(request.request_id)

    assert request.finish_reason == "abort"
    assert engine.scheduler.num_running == 0
    assert engine.scheduler.num_free_slots == engine.scheduler.num_slots
    assert not engine.has_unfinished_requests()


def test_repeated_waves_do_not_leak_slots_or_cache(engine):
    """Ten rounds through the same engine must end exactly where they started."""
    for _ in range(10):
        for prompt in PROMPTS[:2]:
            engine.add_request(
                prompt,
                SamplingParams(temperature=0.0, max_gen_len=4, repetition_penalty=1.0),
            )
        drain(engine)

    assert engine.scheduler.num_free_slots == engine.scheduler.num_slots
    assert engine.scheduler.num_running == 0


def test_generate_returns_outputs_in_submission_order(engine, reference):
    outputs = engine.generate(PROMPTS, GREEDY)

    assert [o.prompt for o in outputs] == PROMPTS
    assert all(o.text for o in outputs)
    assert all(o.outputs[0].finish_reason for o in outputs)
    assert_no_foreign_tail({o.prompt: o.text for o in outputs}, reference)


def test_an_oversized_prompt_is_refused(engine):
    with pytest.raises(ValueError):
        engine.add_request("word " * (_MAX_SEQ_LEN + 10), GREEDY)


def test_multimodal_checkpoints_are_rejected(engine, monkeypatch):
    """Vision prefill needs per-request processor outputs the grid cannot carry.

    ``ModelSpec`` is frozen, so the runner is given a replaced spec rather than a
    mutated one.
    """
    runner = engine.engine.model_runner
    monkeypatch.setattr(runner, "spec", replace(runner.spec, is_multimodal=True))
    with pytest.raises(NotImplementedError):
        ContinuousBatchingEngine(engine.engine)
