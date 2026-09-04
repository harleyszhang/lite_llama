"""CPU-only tests for the Sampler + SamplingParams pair.

Param validation, greedy argmax behaviour, nucleus-sampling bounds and
the repetition penalty — logits are small tensors built inline.

Usage:
    pytest tests/engine/test_sampler.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.engine.sampler import (
    BatchedSamplingParams,
    GeneratedSpan,
    Sampler,
    SamplingParams,
    _distribution_records,
    apply_repetition_penalty,
    greedy_ids,
    rows_logprobs,
    sample_top_p,
)


def test_sampling_params_rejects_negative_temperature():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-0.1)


def test_sampling_params_rejects_top_p_outside_range():
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)


@pytest.mark.parametrize("value", [0, -1])
def test_sampling_params_rejects_non_positive_generation_caps(value):
    with pytest.raises(ValueError, match="max_gen_len"):
        SamplingParams(max_gen_len=value)


def test_greedy_flag():
    assert SamplingParams(temperature=0.0).is_greedy
    assert not SamplingParams(temperature=0.7).is_greedy


def test_greedy_picks_argmax_over_last_position():
    sampler = Sampler()
    logits = torch.tensor([[[0.1, 0.9, 0.0], [0.5, 0.1, 0.4]]])  # last step: token 0
    result = sampler.sample(logits, SamplingParams(temperature=0.0))
    assert result.shape == (1, 1)
    assert result.item() == 0


def test_greedy_supports_two_dim_logits():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.5, 0.1, 0.4]])
    result = sampler.sample(logits, SamplingParams(temperature=0.0))
    assert result.squeeze(-1).tolist() == [1, 0]


def test_top_p_keeps_only_the_nucleus():
    """Top-p 0.1 must keep exactly one token (the dominant one)."""
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    torch.manual_seed(0)
    for _ in range(10):
        token = sample_top_p(probs.clone(), top_p=0.1).item()
        assert token == 0


def test_top_p_small_pool_matches_full_vocab_behaviour():
    """A pool narrower than the vocabulary must not change the draw.

    When the pool's mass exceeds ``top_p`` the nucleus is inside it, so the
    small-``k`` shortcut and a full-vocabulary sort keep the same tokens. Here
    the nucleus is a single token, which makes the check deterministic.
    """
    torch.manual_seed(0)
    vocab = 400
    logits = torch.randn(1, vocab) * 6  # concentrated: top token holds > 0.5
    probs = torch.softmax(logits, dim=-1)

    for _ in range(10):
        token = sample_top_p(probs.clone(), top_p=probs.max().item() * 0.9, k=8).item()
        assert token == probs.argmax().item()


def test_top_p_pool_never_leaks_beyond_candidates():
    """Every drawn id must be one of the pool's own ids."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 500) * 3, dim=-1)
    pool = set(probs.topk(16, dim=-1).indices.flatten().tolist())

    tokens = sample_top_p(probs, top_p=0.99, k=16)
    assert set(tokens.flatten().tolist()) <= pool


def test_top_p_flat_distribution_keeps_whole_pool():
    """A row flatter than the pool can cover must not crash or drop everything.

    Uniform probabilities never accumulate ``top_p`` inside the pool, so the
    fallback keeps every candidate — the draw stays inside the pool and the
    renormalised sum stays at 1.
    """
    torch.manual_seed(0)
    vocab = 2000
    probs = torch.full((2, vocab), 1.0 / vocab)
    pool = set(probs.topk(16, dim=-1).indices.flatten().tolist())

    tokens = sample_top_p(probs, top_p=0.999, k=16)
    assert set(tokens.flatten().tolist()) <= pool


def test_top_p_accepts_per_row_thresholds():
    """The [batch, 1] tensor form gives each row its own nucleus."""
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    top_p = torch.tensor([[0.1], [0.99]])
    torch.manual_seed(0)

    tokens = sample_top_p(probs.clone(), top_p)

    assert tokens.shape == (2, 1)
    assert tokens[0].item() == 0  # nucleus of one: the dominant token
    assert 0 <= tokens[1].item() < 3  # nucleus covers the whole row


def test_top_p_one_samples_the_full_distribution_without_sorting(monkeypatch):
    """top_p=1 is exact categorical sampling, not a top-1024 approximation."""
    seen_widths: list[int] = []

    def fake_multinomial(probs, num_samples):
        seen_widths.append(probs.shape[-1])
        return torch.full((probs.shape[0], num_samples), probs.shape[-1] - 1)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    token = sample_top_p(torch.full((1, 2048), 1 / 2048), top_p=1.0)

    assert seen_widths == [2048]
    assert token.item() == 2047


def test_sampled_temperature_stays_within_vocab():
    sampler = Sampler()
    logits = torch.randn(4, 100)
    tokens = sampler.sample(logits, SamplingParams(temperature=0.8, top_p=0.9))
    assert tokens.shape == (4, 1)
    assert tokens.min() >= 0
    assert tokens.max() < 100


# --------------------------------------------------------------------------- #
# The greedy draw runs only where a row will use it
# --------------------------------------------------------------------------- #
@pytest.fixture
def greedy_draw_calls(monkeypatch) -> list[int]:
    """Count the greedy draws a pass makes, delegating so the output is unchanged."""
    calls: list[int] = []

    def counting(logits, offset):
        calls.append(logits.shape[0])
        return greedy_ids(logits, offset)

    monkeypatch.setattr("rapid_llm.engine.sampler.greedy_ids", counting)
    return calls


def test_a_wholly_stochastic_batch_skips_the_greedy_draw(greedy_draw_calls):
    """No row wants the argmax, so the pass must not compute it.

    The draw is a full-vocabulary reduction, and under TP two collectives on
    top of it. Running it for a batch that then selects the nucleus draw for
    every row is waste on every step of a sampling run.
    """
    Sampler().sample(torch.randn(4, 100), SamplingParams(temperature=0.7, top_p=0.9))
    assert greedy_draw_calls == []


def test_a_greedy_batch_still_draws_the_argmax(greedy_draw_calls):
    """Skipping the draw must not skip the greedy path it exists for."""
    logits = torch.randn(4, 100)
    tokens = Sampler().sample(logits, SamplingParams(temperature=0.0))
    assert greedy_draw_calls == [4]
    assert tokens.flatten().tolist() == logits.argmax(-1).tolist()


def test_a_mixed_batch_draws_the_argmax_once(greedy_draw_calls):
    """Greedy and stochastic rows share one pass, so one argmax covers them all.

    Splitting the batch by configuration would multiply the launches, which is
    what :class:`BatchedSamplingParams` exists to avoid.
    """
    logits = torch.randn(4, 100)
    params = BatchedSamplingParams.build(
        [
            SamplingParams(temperature=0.0),
            SamplingParams(temperature=0.7, top_p=0.9),
            SamplingParams(temperature=0.0),
            SamplingParams(temperature=0.7, top_p=0.9),
        ],
        "cpu",
    )
    tokens = Sampler().sample_batched(logits, params)
    assert greedy_draw_calls == [4]
    # The greedy rows are the argmax; the stochastic ones are free to differ.
    assert tokens[0].item() == logits[0].argmax().item()
    assert tokens[2].item() == logits[2].argmax().item()


# --------------------------------------------------------------------------- #
# apply_repetition_penalty
#
# The vectorised implementation replaced a per-row ``torch.unique`` loop, so
# these tests pin the semantics it has to reproduce: HuggingFace's
# ``RepetitionPenaltyLogitsProcessor`` (divide positive logits, multiply
# negative ones), applied only to *generated* tokens, idempotent for repeats,
# and correct in the presence of padding.
# --------------------------------------------------------------------------- #
def _span(token_ids: list[list[int]], mask: list[list[bool]]) -> GeneratedSpan:
    return GeneratedSpan(
        token_ids=torch.tensor(token_ids), mask=torch.tensor(mask, dtype=torch.bool)
    )


def test_penalty_of_one_is_a_no_op():
    logits = torch.tensor([[1.0, -2.0, 3.0]])
    out = apply_repetition_penalty(logits.clone(), _span([[0, 2]], [[True, True]]), 1.0)
    torch.testing.assert_close(out, logits)


def test_positive_logits_are_divided_and_negative_multiplied():
    """HF's asymmetric rule: both directions must move the logit *down*.

    Dividing a negative logit would raise it, i.e. reward the repetition, which
    is why the sign branch exists at all.
    """
    logits = torch.tensor([[4.0, -4.0, 1.0]])
    out = apply_repetition_penalty(logits, _span([[0, 1]], [[True, True]]), 2.0)
    assert out[0, 0].item() == pytest.approx(2.0)  # 4 / 2
    assert out[0, 1].item() == pytest.approx(-8.0)  # -4 * 2
    assert out[0, 2].item() == pytest.approx(1.0)  # untouched


def test_only_listed_tokens_are_penalised():
    logits = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    out = apply_repetition_penalty(logits, _span([[1]], [[True]]), 2.0)
    assert out[0].tolist() == pytest.approx([2.0, 1.0, 2.0, 2.0])


def test_repeated_token_is_penalised_once():
    """Idempotence: matching ``torch.unique`` semantics, not a per-occurrence loop.

    Applying the penalty twice for a token seen twice would compound to 1.0 here
    instead of 2.0 and would make the penalty depend on span length.
    """
    logits = torch.tensor([[4.0, 0.0]])
    out = apply_repetition_penalty(logits, _span([[0, 0, 0]], [[True, True, True]]), 2.0)
    assert out[0, 0].item() == pytest.approx(2.0)


def test_padded_positions_are_ignored():
    """Masked-out slots must not penalise the token id they happen to hold."""
    logits = torch.tensor([[4.0, 4.0]])
    out = apply_repetition_penalty(logits, _span([[0, 1]], [[True, False]]), 2.0)
    assert out[0, 0].item() == pytest.approx(2.0)
    assert out[0, 1].item() == pytest.approx(4.0)


def test_padding_does_not_cancel_a_real_hit_of_the_same_id():
    """A padded slot holding an id that also occurs for real must not clear it.

    This is why padding is redirected to a scratch column rather than scattered
    as ``False``: a plain scatter would overwrite the real ``True`` and silently
    drop that token's penalty.
    """
    logits = torch.tensor([[4.0, 1.0]])
    span = _span([[0, 0]], [[True, False]])  # same id, one real one padded
    out = apply_repetition_penalty(logits, span, 2.0)
    assert out[0, 0].item() == pytest.approx(2.0)


def test_rows_are_penalised_independently():
    """Sequence 0's history must not affect sequence 1's logits."""
    logits = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    span = _span([[0], [1]], [[True], [True]])
    out = apply_repetition_penalty(logits, span, 2.0)
    assert out[0].tolist() == pytest.approx([2.0, 4.0])
    assert out[1].tolist() == pytest.approx([4.0, 2.0])


def test_matches_a_naive_per_row_reference():
    """Cross-check the vectorised path against the obvious loop it replaced."""
    torch.manual_seed(0)
    batch, vocab, span_len = 4, 50, 6
    logits = torch.randn(batch, vocab)
    ids = torch.randint(0, vocab, (batch, span_len))
    mask = torch.rand(batch, span_len) > 0.3

    expected = logits.clone()
    for row in range(batch):
        for tid in ids[row][mask[row]].unique():
            value = expected[row, tid]
            expected[row, tid] = value / 1.5 if value > 0 else value * 1.5

    out = apply_repetition_penalty(logits.clone(), GeneratedSpan(token_ids=ids, mask=mask), 1.5)
    torch.testing.assert_close(out, expected)


def test_sampler_applies_penalty_before_choosing():
    """Greedy must pick a different token once the leader is penalised.

    End-to-end through ``Sampler.sample``: this is what verifies the penalty is
    actually wired into the decision rather than computed and discarded.
    """
    sampler = Sampler()
    logits = torch.tensor([[10.0, 9.0]])
    params = SamplingParams(temperature=0.0, repetition_penalty=5.0)
    span = GeneratedSpan(token_ids=torch.tensor([[0]]), mask=torch.tensor([[True]]))

    # Without a span there is nothing to penalise: the raw leader wins.
    assert sampler.sample(logits.clone(), params, generated=None).item() == 0
    # 10 / 5 = 2 now loses to the untouched 9, so the choice must flip.
    assert sampler.sample(logits.clone(), params, generated=span).item() == 1


# --------------------------------------------------------------------------- #
# apply_repetition_penalty under vocabulary parallelism
#
# With a sharded head each rank sees ``[batch, vocab / tp]`` and must penalise only the
# ids it owns, translated into local columns. No process group is needed to test that:
# the offset is an argument, so the sharded behaviour is reproducible on one CPU.
# --------------------------------------------------------------------------- #
def test_a_token_owned_by_another_rank_is_left_alone():
    """Rank 0 of a 4-wide split must not touch its columns because rank 1 generated a
    token: an offset ignored here penalises ``id`` instead of ``id - offset``, which is a
    penalty applied to an unrelated token."""
    logits = torch.tensor([[4.0, 4.0]])
    span = _span([[3]], [[True]])  # rank 1's territory when the offset is 0..1

    torch.testing.assert_close(
        apply_repetition_penalty(logits.clone(), span, 2.0, vocab_offset=0), logits
    )


def test_the_offset_translates_global_ids_into_local_columns():
    """Rank 1 holds ids 2..3, so generating id 3 must move its *second* column."""
    logits = torch.tensor([[4.0, 4.0]])
    out = apply_repetition_penalty(logits, _span([[3]], [[True]]), 2.0, vocab_offset=2)

    assert out[0].tolist() == pytest.approx([4.0, 2.0])


def test_the_shards_together_reproduce_the_unsharded_penalty():
    """The property that matters: concatenating every rank's penalised slice must equal
    penalising the full row once. Each id has to be claimed by exactly one rank."""
    torch.manual_seed(0)
    batch, vocab, tp_size = 4, 32, 4
    local = vocab // tp_size
    logits = torch.randn(batch, vocab)
    span = GeneratedSpan(
        token_ids=torch.randint(0, vocab, (batch, 6)), mask=torch.rand(batch, 6) > 0.3
    )

    expected = apply_repetition_penalty(logits.clone(), span, 1.5)
    shards = [
        apply_repetition_penalty(
            logits[:, rank * local : (rank + 1) * local].clone(), span, 1.5, rank * local
        )
        for rank in range(tp_size)
    ]

    torch.testing.assert_close(torch.cat(shards, dim=-1), expected)


# --------------------------------------------------------------------------- #
# Logprob reporting (ROADMAP F6)
#
# ``sample_with_logprobs`` must describe the distribution actually drawn from:
# penalised, temperature-scaled — except greedy rows, whose clamped temperature
# (1.0) makes their records the raw model distribution, matching HuggingFace
# under ``do_sample=False``. ``rows_logprobs`` scores known targets on the raw
# logits, which is what prompt-logprob reporting during prefill uses.
# --------------------------------------------------------------------------- #
def test_sampling_params_rejects_negative_logprobs():
    with pytest.raises(ValueError):
        SamplingParams(logprobs=-1)
    with pytest.raises(ValueError):
        SamplingParams(prompt_logprobs=-1)


def test_no_logprobs_requested_returns_no_records():
    """``None`` must skip the work entirely — the top-k over the vocabulary is
    the only part of sampling whose cost scales with the vocabulary size."""
    ids, records = Sampler().sample_with_logprobs(
        torch.randn(2, 30), SamplingParams(temperature=0.0)
    )
    assert ids.shape == (2, 1)
    assert records is None


def test_greedy_records_match_a_plain_log_softmax():
    """A greedy row divides by the clamped 1.0, so its record must equal
    ``log_softmax`` of the raw logits — what HF reports under ``do_sample=False``."""
    torch.manual_seed(0)
    logits = torch.randn(3, 50)
    ids, records = Sampler().sample_with_logprobs(
        logits, SamplingParams(temperature=0.0, repetition_penalty=1.0, logprobs=4)
    )

    reference = torch.log_softmax(logits.float(), dim=-1)
    top_values, top_ids = reference.topk(4, dim=-1)
    for row in range(3):
        record = records[row]
        assert record is not None
        assert record.token_id == ids[row].item() == logits[row].argmax().item()
        assert record.logprob == pytest.approx(reference[row, record.token_id].item())
        assert record.top_token_ids == tuple(top_ids[row].tolist())
        assert list(record.top_logprobs) == pytest.approx(top_values[row].tolist())


def test_logprobs_zero_reports_only_the_chosen_token():
    """k=0 keeps the sampled token's own logprob but skips the top-k entirely."""
    torch.manual_seed(0)
    logits = torch.randn(2, 30)
    _, records = Sampler().sample_with_logprobs(
        logits, SamplingParams(temperature=0.0, repetition_penalty=1.0, logprobs=0)
    )

    reference = torch.log_softmax(logits.float(), dim=-1)
    for row, record in enumerate(records):
        assert record is not None
        assert record.top_token_ids == ()
        assert record.top_logprobs == ()
        assert record.logprob == pytest.approx(reference[row, record.token_id].item())


def test_records_describe_the_temperature_scaled_distribution():
    """The sampled token's logprob must come from the scaled distribution, not
    the raw one: ``temperature=2.0`` flattens it, and the record has to follow."""
    torch.manual_seed(0)
    logits = torch.randn(2, 40)
    params = SamplingParams(temperature=2.0, top_p=1.0, repetition_penalty=1.0, logprobs=3)
    ids, records = Sampler().sample_with_logprobs(logits, params)

    reference = torch.log_softmax(logits.float() / 2.0, dim=-1)
    for row in range(2):
        record = records[row]
        assert record.token_id == ids[row].item()
        assert record.logprob == pytest.approx(reference[row, record.token_id].item())


def test_records_describe_the_penalised_distribution():
    """The record follows the penalised logits: the de-moted leader's logprob
    drops, and the draw moves to the token the penalty made the winner."""
    logits = torch.tensor([[10.0, 9.0, 1.0]])
    span = GeneratedSpan(token_ids=torch.tensor([[0]]), mask=torch.tensor([[True]]))
    params = SamplingParams(temperature=0.0, repetition_penalty=2.0, logprobs=3)

    ids, records = Sampler().sample_with_logprobs(logits, params, generated=span)

    reference = torch.log_softmax(apply_repetition_penalty(logits, span, 2.0), dim=-1)
    record = records[0]
    assert record.token_id == ids[0].item() == 1  # 10/2 = 5 now loses to 9
    assert record.logprob == pytest.approx(reference[0, 1].item(), abs=1e-6)
    assert list(record.top_logprobs) == pytest.approx(
        reference.topk(3, dim=-1).values[0].tolist(), abs=1e-6
    )


def test_batched_rows_get_records_only_where_asked():
    """Rows are independent: a row that opted out gets a ``None`` entry, and a
    k=0 row gets its chosen token alone — while the draw itself stays batched."""
    torch.manual_seed(0)
    logits = torch.randn(3, 30)
    params = BatchedSamplingParams.build(
        [
            SamplingParams(temperature=0.0, repetition_penalty=1.0, logprobs=2),
            SamplingParams(temperature=0.0, repetition_penalty=1.0),
            SamplingParams(temperature=0.0, repetition_penalty=1.0, logprobs=0),
        ],
        "cpu",
    )

    ids, records = Sampler().sample_batched_with_logprobs(logits, params)

    assert records[0] is not None
    assert len(records[0].top_token_ids) == 2
    assert records[0].token_id == ids[0].item()
    assert records[1] is None
    assert records[2] is not None
    assert records[2].top_token_ids == ()
    assert records[2].token_id == ids[2].item()


def test_batched_records_of_greedy_rows_use_the_clamped_temperature():
    """Batched greedy rows were clamped to 1.0 at build time; their records
    must describe the raw distribution, not a division by zero."""
    torch.manual_seed(0)
    logits = torch.randn(2, 20)
    params = BatchedSamplingParams.build(
        [SamplingParams(temperature=0.0, repetition_penalty=1.0, logprobs=2)] * 2, "cpu"
    )

    _, records = Sampler().sample_batched_with_logprobs(logits, params)

    reference = torch.log_softmax(logits.float(), dim=-1)
    for row, record in enumerate(records):
        assert record.logprob == pytest.approx(reference[row, record.token_id].item())


def test_rows_logprobs_scores_known_targets_on_raw_logits():
    """Prompt scoring: each row's record carries its own target token's logprob
    under the raw distribution, plus the top-k — no temperature, no penalty."""
    torch.manual_seed(0)
    logits = torch.randn(4, 30)
    targets = torch.tensor([0, 5, 12, 29])

    records = rows_logprobs(logits, targets, 3)

    reference = torch.log_softmax(logits.float(), dim=-1)
    top_values, top_ids = reference.topk(3, dim=-1)
    for row in range(4):
        assert records[row].token_id == targets[row].item()
        assert records[row].logprob == pytest.approx(reference[row, targets[row]].item())
        assert records[row].top_token_ids == tuple(top_ids[row].tolist())
        assert list(records[row].top_logprobs) == pytest.approx(top_values[row].tolist())


def test_sharded_distribution_records_reproduce_the_full_vocabulary():
    """The TP branch's collectives, simulated by hand on one CPU.

    The properties the wire protocol relies on: MAX then SUM over the shards
    gives the global logsumexp; the masked gather hands the chosen id's logit
    to its owning rank alone, so a SUM reduce recovers it; and the union of
    per-rank top-k's contains the global top-k, so gathering ``O(k * tp)``
    candidates is enough.
    """
    torch.manual_seed(0)
    rows, vocab, tp_size, k = 3, 32, 4, 5
    local = vocab // tp_size
    logits = torch.randn(rows, vocab)
    ids = torch.randint(0, vocab, (rows, 1))

    chosen_ref, top_v_ref, top_i_ref = _distribution_records(logits, ids, None, k)

    shards = [logits[:, r * local : (r + 1) * local] for r in range(tp_size)]
    row_max = torch.stack([s.amax(dim=-1, keepdim=True) for s in shards]).amax(dim=0)
    log_z = (
        row_max
        + torch.stack([(s - row_max).exp().sum(dim=-1, keepdim=True) for s in shards])
        .sum(dim=0)
        .log()
    )

    chosen = torch.zeros(rows, 1)
    pool_values, pool_ids = [], []
    for r, shard in enumerate(shards):
        local_ids = ids - r * local
        valid = (local_ids >= 0) & (local_ids < local)
        gathered = shard.gather(-1, local_ids.clamp(0, local - 1))
        chosen += torch.where(valid, gathered, torch.zeros_like(gathered))
        values, indices = (shard - log_z).topk(k, dim=-1)
        pool_values.append(values)
        pool_ids.append(indices + r * local)
    chosen -= log_z
    top_v_shard, order = torch.cat(pool_values, dim=-1).topk(k, dim=-1)
    top_i_shard = torch.cat(pool_ids, dim=-1).gather(-1, order)

    torch.testing.assert_close(chosen, chosen_ref)
    torch.testing.assert_close(top_v_shard, top_v_ref)
    assert (top_i_shard == top_i_ref).all()
