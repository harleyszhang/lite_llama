"""CPU-only tests for the Sampler + SamplingParams pair."""

from __future__ import annotations

import pytest
import torch

from lite_llama.engine.sampler import (
    GeneratedSpan,
    Sampler,
    SamplingParams,
    apply_repetition_penalty,
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
