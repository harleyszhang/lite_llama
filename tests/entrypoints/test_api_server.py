"""Tests for the OpenAI-compatible HTTP layer.

The point of a wire protocol is that somebody else's client works against it, so
what matters here is the shape of the JSON and the SSE framing -- not what the
model says. A fake engine therefore stands in for the real one, which is what
lets this whole file run on CPU with no checkpoint: protocol regressions are
cheap to catch and should not need a GPU.

The fake records what it was asked for, so the tests can also pin down the
translation in the other direction: that ``max_tokens`` and friends actually
reach :class:`~lite_llama.engine.sampler.SamplingParams`, and that chat messages
go through the tokenizer's template.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the `serve` extra")

from fastapi.testclient import TestClient

from lite_llama.engine.async_engine import StreamedOutput
from lite_llama.entrypoints.api_server import (
    ServerConfig,
    build_app,
    parse_sse,
)

pytestmark = pytest.mark.serving

_REPLY = "Paris is the capital."
_MODEL = "fake-model"


class FakeTokenizer:
    """Just enough tokenizer for templating and token counting.

    ``apply_chat_template`` wraps each turn in visible markers so a test can
    assert the template ran, rather than inferring it from a token count.
    """

    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.templated: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.templated.append(list(messages))
        turns = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return f"{turns}<|assistant|>"

    def encode(self, text, add_special_tokens=True):
        return text.split()


class FakeEngine:
    """Streams a fixed reply word by word and records every request it saw."""

    def __init__(self, reply: str = _REPLY) -> None:
        self.tokenizer = FakeTokenizer()
        self._reply = reply
        self.seen: list[tuple[str, object]] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.started = False

    async def generate(self, prompt, sampling_params=None, request_id=None):
        self.seen.append((prompt, sampling_params))
        text = ""
        pieces = self._reply.split(" ")
        for index, piece in enumerate(pieces):
            delta = piece if index == 0 else " " + piece
            text += delta
            last = index == len(pieces) - 1
            yield StreamedOutput(
                request_id=request_id or "fake",
                delta=delta,
                text=text,
                finish_reason="eos" if last else None,
                prompt_tokens=len(self.tokenizer.encode(prompt)),
                completion_tokens=index + 1,
            )

    async def generate_text(self, prompt, sampling_params=None, request_id=None):
        final = None
        async for chunk in self.generate(prompt, sampling_params, request_id):
            final = chunk
        return final


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


def make_client(engine: FakeEngine, **config_kwargs) -> TestClient:
    config = ServerConfig(model_dir="/nonexistent", served_model_name=_MODEL, **config_kwargs)
    return TestClient(build_app(config, engine=engine))


@pytest.fixture
def client(engine) -> TestClient:
    with make_client(engine) as started:
        yield started


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_health_reports_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_the_engine_is_started_by_the_app_lifespan(engine):
    with make_client(engine):
        assert engine.started


def test_models_lists_the_served_name(client):
    body = client.get("/v1/models").json()

    assert body["object"] == "list"
    assert [card["id"] for card in body["data"]] == [_MODEL]


# --------------------------------------------------------------------------- #
# /v1/completions
# --------------------------------------------------------------------------- #
def test_completion_returns_the_openai_shape(client):
    body = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "max_tokens": 8}
    ).json()

    assert body["object"] == "text_completion"
    assert body["model"] == _MODEL
    assert body["id"].startswith("cmpl-")
    assert body["choices"] == [{"index": 0, "text": _REPLY, "finish_reason": "eos"}]


def test_completion_reports_token_usage(client):
    body = client.post("/v1/completions", json={"model": _MODEL, "prompt": "one two three"}).json()
    usage = body["usage"]

    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == len(_REPLY.split())
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_usage_counts_come_from_the_engine_not_a_reencode():
    """Decode does not round-trip through encode, so usage must be the engine's
    own counts rather than whatever encoding the texts again happens to give.
    The fixed counts below deliberately disagree with both texts' word counts.
    """

    class FixedCountsEngine(FakeEngine):
        async def generate(self, prompt, sampling_params=None, request_id=None):
            self.seen.append((prompt, sampling_params))
            yield StreamedOutput(
                request_id=request_id or "fake",
                delta=self._reply,
                text=self._reply,
                finish_reason="eos",
                prompt_tokens=7,
                completion_tokens=11,
            )

    with make_client(FixedCountsEngine()) as client:
        usage = client.post(
            "/v1/completions", json={"model": _MODEL, "prompt": "Hello"}
        ).json()["usage"]

    assert usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}


def test_completion_passes_the_prompt_through_untemplated(client, engine):
    client.post("/v1/completions", json={"model": _MODEL, "prompt": "raw prompt"})

    assert engine.seen[0][0] == "raw prompt"
    assert engine.tokenizer.templated == [], "/v1/completions must not apply a chat template"


def test_sampling_fields_reach_the_engine(client, engine):
    client.post(
        "/v1/completions",
        json={
            "model": _MODEL,
            "prompt": "Hello",
            "max_tokens": 11,
            "temperature": 0.25,
            "top_p": 0.5,
            "repetition_penalty": 1.3,
        },
    )
    params = engine.seen[0][1]

    assert params.max_gen_len == 11
    assert params.temperature == pytest.approx(0.25)
    assert params.top_p == pytest.approx(0.5)
    assert params.repetition_penalty == pytest.approx(1.3)


def test_the_protocol_defaults_match_openai_not_the_cli(client, engine):
    """A wire client expects OpenAI's 1.0/1.0, not lite_llama's own 0.6/0.9."""
    client.post("/v1/completions", json={"model": _MODEL, "prompt": "Hello"})
    params = engine.seen[0][1]

    assert params.temperature == pytest.approx(1.0)
    assert params.top_p == pytest.approx(1.0)
    assert params.repetition_penalty == pytest.approx(1.0)


def test_streamed_completion_deltas_rebuild_the_reply(client):
    response = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "stream": True}
    )
    frames = parse_sse(response.text)

    assert "".join(f["choices"][0]["text"] for f in frames) == _REPLY
    assert frames[-1]["choices"][0]["finish_reason"] == "eos"
    assert all(f["choices"][0]["finish_reason"] is None for f in frames[:-1])


def test_a_stream_ends_with_the_done_sentinel(client):
    response = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "stream": True}
    )

    assert response.text.rstrip().endswith("data: [DONE]")
    assert response.headers["content-type"].startswith("text/event-stream")


def test_every_frame_of_one_stream_shares_an_id(client):
    response = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "stream": True}
    )
    ids = {f["id"] for f in parse_sse(response.text)}

    assert len(ids) == 1, "a client correlates frames by id"


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #
def test_chat_completion_returns_an_assistant_message(client):
    body = client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": [{"role": "user", "content": "Hi"}]},
    ).json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"] == {"role": "assistant", "content": _REPLY}
    assert body["choices"][0]["finish_reason"] == "eos"


def test_chat_applies_the_tokenizer_template_to_every_turn(client, engine):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Again"},
    ]
    client.post("/v1/chat/completions", json={"model": _MODEL, "messages": messages})

    assert engine.tokenizer.templated == [messages], "all turns must reach the template"
    assert engine.seen[0][0].endswith("<|assistant|>")


def test_chat_without_a_template_concatenates_the_turns(engine):
    """Base models have no template, and inventing one makes them echo role names."""
    with make_client(engine, chat_template=False) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )

    assert engine.seen[0][0] == "Be brief.\nHi"
    assert engine.tokenizer.templated == []


def test_streamed_chat_opens_with_a_role_only_delta(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )
    frames = parse_sse(response.text)

    assert frames[0]["object"] == "chat.completion.chunk"
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    content = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)
    assert content == _REPLY
    assert frames[-1]["choices"][0]["finish_reason"] == "eos"


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
def test_more_than_one_completion_is_refused_rather_than_ignored(client):
    """Silently returning one completion for ``n=4`` is undetectable by a client."""
    response = client.post("/v1/completions", json={"model": _MODEL, "prompt": "Hello", "n": 4})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"model": _MODEL, "prompt": ""}, id="empty-prompt"),
        pytest.param({"model": _MODEL, "prompt": "x", "temperature": -1}, id="negative-temp"),
        pytest.param({"model": _MODEL, "prompt": "x", "top_p": 0}, id="zero-top-p"),
        pytest.param({"model": _MODEL, "prompt": "x", "top_p": 1.5}, id="top-p-above-one"),
        pytest.param({"model": _MODEL, "prompt": "x", "max_tokens": 0}, id="zero-max-tokens"),
        pytest.param({"prompt": "x"}, id="missing-model"),
    ],
)
def test_invalid_bodies_are_rejected(client, body):
    assert client.post("/v1/completions", json=body).status_code == 422


def test_chat_requires_at_least_one_message(client):
    response = client.post("/v1/chat/completions", json={"model": _MODEL, "messages": []})
    assert response.status_code == 422
