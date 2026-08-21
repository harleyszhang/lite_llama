"""Tests for chat prompt formatting via the tokenizer's own chat template.

``get_prompter`` returns a :class:`ChatPrompter` that defers to
``tokenizer.apply_chat_template`` when the checkpoint ships a template, or ``None``
when it does not (base models, whose prompts are sent verbatim). The instruct-vs-base
decision lives in ``cli.PrompterResolver``; here we pin the formatting contract.
"""

from __future__ import annotations

from lite_llama.utils.prompt_templates import ChatPrompter, get_prompter, has_chat_template


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer, with or without a chat template."""

    def __init__(self, chat_template: str | None = "<jinja>") -> None:
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)


# --------------------------------------------------------------------------- #
# Template detection
# --------------------------------------------------------------------------- #
def test_has_chat_template_distinguishes_instruct_from_base():
    assert has_chat_template(_FakeTokenizer())
    assert not has_chat_template(_FakeTokenizer(chat_template=None))


def test_get_prompter_returns_none_without_a_template():
    """No template means a base checkpoint -> the caller must send prompts verbatim."""
    assert get_prompter(_FakeTokenizer(chat_template=None)) is None


def test_get_prompter_builds_a_chat_prompter_with_a_template():
    assert isinstance(get_prompter(_FakeTokenizer()), ChatPrompter)


# --------------------------------------------------------------------------- #
# Formatting behaviour
# --------------------------------------------------------------------------- #
def test_insert_prompt_defers_to_the_tokenizer_template():
    prompter = get_prompter(_FakeTokenizer())
    out = prompter.insert_prompt("hello")
    assert out == prompter.model_input == "<|user|>hello<|assistant|>"


def test_insert_prompt_prepends_the_system_message_when_set():
    prompter = get_prompter(_FakeTokenizer(), system_prompt="be brief")
    prompter.insert_prompt("hi")
    assert prompter.model_input == "<|system|>be brief<|user|>hi<|assistant|>"


def test_insert_prompt_omits_the_system_message_when_unset():
    prompter = ChatPrompter(_FakeTokenizer())
    prompter.insert_prompt("hi")
    assert "<|system|>" not in prompter.model_input


# --------------------------------------------------------------------------- #
# CLI integration (base-vs-instruct routing)
# --------------------------------------------------------------------------- #
def test_cli_build_returns_none_for_the_base_style():
    """Base models take the ``"empty"`` style and get no prompter at all (verbatim)."""
    from lite_llama.cli import PrompterResolver

    assert PrompterResolver.build("empty", _FakeTokenizer()) is None


def test_cli_build_uses_a_chat_prompter_for_an_instruct_style():
    from lite_llama.cli import PrompterResolver

    assert isinstance(PrompterResolver.build("qwen2", _FakeTokenizer()), ChatPrompter)
