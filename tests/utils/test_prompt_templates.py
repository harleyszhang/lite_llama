"""Tests for the prompter registry and its per-family dispatch rules.

``get_prompter`` turns a ``model_type`` plus a checkpoint *path* into the chat
template to wrap prompts in. The path matters as much as the type: within the
``llama`` family the same ``model_type`` maps to Vicuna, Llama-2, Llama-3 or
LLaVA templates purely on filename substrings. Those rules are pure string
logic, so they are cheap to test and easy to get wrong -- and getting them wrong
degrades output quality without raising anything, because the model still
receives a syntactically valid prompt, just the wrong one.

The registry replaced an if-elif chain, so what is pinned here is the dispatch
table's behaviour, plus the boundary cases the old chain's ordering encoded
(``llama-3.2`` counting as Llama-3, the 30B variant deliberately not).
"""

from __future__ import annotations

import pytest

from lite_llama.utils.prompt_templates import (
    EmptyPrompter,
    Llama2Prompter,
    Llama3Prompter,
    LlavaLlama3Prompter,
    LlavaLlamaPrompter,
    Qwen2Prompter,
    VicunaPrompter,
    get_prompter,
    get_stop_token_ids,
    register_prompter,
)


# --------------------------------------------------------------------------- #
# Dispatch by model_type
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_qwen_families_share_the_chatml_template(model_type):
    """Qwen2 and Qwen3 both use ChatML, so they must resolve to one prompter."""
    assert isinstance(get_prompter(model_type), Qwen2Prompter)


def test_model_type_lookup_is_case_insensitive():
    assert isinstance(get_prompter("QWEN2"), Qwen2Prompter)


def test_unknown_model_type_is_rejected():
    """Falling back to a default template would silently mis-prompt a new model."""
    with pytest.raises(ValueError, match="not supported"):
        get_prompter("no_such_model")


def test_empty_prompt_bypasses_the_registry():
    """Base (non-instruct) checkpoints take the raw prompt with no chat wrapper."""
    assert isinstance(get_prompter("llama", empty_prompt=True), EmptyPrompter)
    # Even for a type that is not registered at all.
    assert isinstance(get_prompter("no_such_model", empty_prompt=True), EmptyPrompter)


# --------------------------------------------------------------------------- #
# Dispatch by checkpoint path within the llama family
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path,expected",
    [
        pytest.param("my_weight/vicuna-7b-v1.5", VicunaPrompter, id="vicuna"),
        pytest.param("my_weight/Llama-3.2-1B-Instruct", Llama3Prompter, id="llama-3.2"),
        pytest.param("my_weight/llama3-8b", Llama3Prompter, id="llama3-no-dot"),
        pytest.param("my_weight/llava-1.5-7b-hf", LlavaLlamaPrompter, id="llava-1.5"),
        pytest.param("my_weight/Llama-3-VILA1.5-8b", LlavaLlama3Prompter, id="llama3-vila"),
        pytest.param("my_weight/llama3-llava-next-8b", Llama3Prompter, id="llama3-llava-next"),
        pytest.param("my_weight/Llama-2-7b-chat", Llama2Prompter, id="llama-2"),
        pytest.param("my_weight/some-unknown-llama", Llama2Prompter, id="fallback-llama2"),
    ],
)
def test_llama_family_dispatches_on_path(path, expected):
    """One ``model_type``, four templates, chosen by filename substrings."""
    assert isinstance(get_prompter("llama", model_path=path), expected)


def test_llama3_30b_variant_uses_the_llama2_template():
    """A deliberate carve-out: the 30B "llama-3" build ships Llama-2 formatting.

    Encoded in ``_is_llama3_name`` as an explicit ``30b`` exclusion, so it is
    pinned here rather than left to be "simplified" away.
    """
    prompter = get_prompter("llama", model_path="my_weight/llama-3-30b")
    assert isinstance(prompter, Llama2Prompter)
    assert not isinstance(prompter, Llama3Prompter)


def test_path_matching_ignores_case():
    assert isinstance(get_prompter("llama", model_path="MY_WEIGHT/VICUNA-7B"), VicunaPrompter)


# --------------------------------------------------------------------------- #
# Stop-token rules
# --------------------------------------------------------------------------- #
def test_llama3_declares_its_stop_tokens():
    """Llama-3 needs <|end_of_text|> and <|eot_id|>; Llama-2 relies on the tokenizer."""
    assert get_stop_token_ids("llama", "my_weight/Llama-3.2-1B") == [128001, 128009]
    assert get_stop_token_ids("llama", "my_weight/Llama-2-7b") == []


def test_mpt_chat_variant_has_its_own_stop_tokens():
    assert get_stop_token_ids("mpt", "my_weight/mpt-7b-chat") == [50278, 0]
    assert get_stop_token_ids("mpt", "my_weight/mpt-7b") == []


def test_model_type_without_a_stop_rule_is_reported():
    """Qwen registers no stop rule -- the caller must be told, not given []."""
    with pytest.raises(ValueError, match="not supported"):
        get_stop_token_ids("qwen2")


# --------------------------------------------------------------------------- #
# Templating behaviour
# --------------------------------------------------------------------------- #
def test_insert_prompt_embeds_the_user_text():
    prompter = get_prompter("qwen2")
    prompter.insert_prompt("What is 2 + 2?")
    assert "What is 2 + 2?" in prompter.model_input


def test_chatml_template_carries_the_role_markers():
    """A ChatML prompt must actually be ChatML, or the model sees plain text."""
    prompter = get_prompter("qwen2")
    prompter.insert_prompt("hello")
    assert "<|im_start|>" in prompter.model_input
    assert "<|im_end|>" in prompter.model_input


def test_empty_prompter_is_not_a_passthrough():
    """``EmptyPrompter`` still wraps the prompt in BasePrompter's colons.

    Its roles are empty strings but ``colon`` defaults to ``":"``, so the result
    is ``": <prompt>:"`` rather than the prompt itself. This is why
    ``cli.PrompterResolver.build`` returns ``None`` for the ``"empty"`` style and
    bypasses the prompter layer entirely for base checkpoints. Pinned here so
    the bypass is not "simplified" away on the assumption that this class is
    already transparent.
    """
    prompter = EmptyPrompter()
    prompter.insert_prompt("The capital of France is")

    assert prompter.model_input != "The capital of France is"
    assert prompter.model_input == ": The capital of France is:"


def test_cli_bypasses_the_prompter_for_base_checkpoints():
    """The counterpart to the trap above: base models get no prompter at all."""
    from lite_llama.cli import PrompterResolver

    assert PrompterResolver.build("empty", "my_weight/Qwen2.5-0.5B", 2048) is None


def test_qwen_update_template_drops_prior_turns():
    """Qwen's ``update_template`` rebuilds a single-turn prompt, discarding history.

    The implementation says so outright ("简单起见,不做复杂处理"), so each turn is
    sent standalone: the model gets no conversation context. That is a real
    functional limitation rather than a bug in this test, and pinning it here
    means any future attempt to add history has to update this expectation
    deliberately instead of discovering the behaviour by accident.
    """
    prompter = get_prompter("qwen2")
    prompter.insert_prompt("first question")
    prompter.update_template("first answer")
    prompter.insert_prompt("second question")

    assert "second question" in prompter.model_input
    assert "first answer" not in prompter.model_input
    assert "first question" not in prompter.model_input


def test_vicuna_update_template_does_carry_prior_turns():
    """Contrast with Qwen: the BasePrompter implementation does thread history.

    Shows the inconsistency is per-family, not global -- BasePrompter folds the
    previous answer into the next template, Qwen2Prompter overrides that away.
    """
    prompter = get_prompter("llama", model_path="my_weight/vicuna-7b-v1.5")
    prompter.insert_prompt("first question")
    prompter.update_template("first answer")
    prompter.insert_prompt("second question")

    assert "first answer" in prompter.model_input
    assert "second question" in prompter.model_input


# --------------------------------------------------------------------------- #
# Registry extension
# --------------------------------------------------------------------------- #
def test_a_new_family_can_be_registered(monkeypatch):
    """Adding a model must not require editing a dispatch chain.

    ``monkeypatch`` is not enough on its own here because the registry is module
    global, so the entry is removed explicitly afterwards to keep the test
    isolated from ordering.
    """
    from lite_llama.utils import prompt_templates

    register_prompter("test_family", lambda path, short: EmptyPrompter())
    try:
        assert isinstance(get_prompter("test_family"), EmptyPrompter)
    finally:
        prompt_templates._PROMPTER_FACTORIES.pop("test_family", None)

    with pytest.raises(ValueError, match="not supported"):
        get_prompter("test_family")
