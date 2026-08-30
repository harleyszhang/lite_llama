"""What the CLI hands the engine, and what it does with the engine's output.

The CLI is where a typo becomes a silent behaviour change: a flag that stops
being forwarded does not raise, it just runs the wrong configuration — decode
graphs off, one GPU instead of four — and only shows up as a number nobody
notices. So the assertions here are on the wiring: the options a subcommand
derives, the engine call it makes, and the release it owes on the way out.

Nothing loads a checkpoint. The engine is a fake, ``--model-dir`` is a temporary
directory (the CLI only checks that it exists), and every test runs on CPU in
milliseconds, which is what makes this tier worth running on every commit.

Usage:
    pytest tests/entrypoints/test_cli_wiring.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite_llama import cli
from lite_llama.cli import TextEngineOptions, build_parser


@pytest.fixture
def model_dir(tmp_path):
    """Any existing directory satisfies the CLI; no weights are read."""
    return tmp_path


def options_for(argv: list[str], model_dir) -> TextEngineOptions:
    args = build_parser().parse_args([*argv, "--model-dir", str(model_dir)])
    return TextEngineOptions.from_args(args)


class FakeEngine:
    """One request, one step, then finished — the shape ``chat`` consumes.

    Mimics the three methods the REPL uses (``add_request``, ``has_unfinished_requests``,
    ``step``) plus ``shutdown``, so a test can watch the release happen.
    """

    def __init__(self, finish_reason: str = "length", reject: bool = False) -> None:
        self.tokenizer = None
        self.released = False
        self.prompts: list[str] = []
        self._finish_reason = finish_reason
        self._reject = reject
        self._request = None

    def add_request(self, prompt, params=None):
        if self._reject:
            raise ValueError("prompt is longer than the context window")
        self.prompts.append(prompt)
        self._request = SimpleNamespace(delta=f"reply to {prompt}", finish_reason=None)
        return self._request

    def has_unfinished_requests(self) -> bool:
        return self._request is not None

    def step(self) -> list:
        request, self._request = self._request, None
        request.finish_reason = self._finish_reason
        return [request]

    def shutdown(self) -> None:
        self.released = True


def run_chat(monkeypatch, model_dir, engine: FakeEngine, typed: list[str], *argv: str) -> int:
    """Drive the chat REPL over ``engine``, feeding ``typed`` at the prompt."""
    monkeypatch.setattr(TextEngineOptions, "build_engine", lambda self, **_kwargs: engine)
    lines = iter(typed)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(lines))
    args = build_parser().parse_args(["chat", "--model-dir", str(model_dir), *argv])
    return args.handler.run(args)


class TestCudaGraphDefaults:
    """The right default differs by command, so both flag spellings must work."""

    def test_chat_captures_graphs_only_when_asked(self, model_dir):
        assert options_for(["chat"], model_dir).use_cuda_graph is False
        assert options_for(["chat", "--cuda-graph"], model_dir).use_cuda_graph is True

    def test_a_throughput_command_captures_unless_refused(self, model_dir):
        """``batch`` and ``serve`` exist for tokens per second; eager decode is opt-in."""
        assert options_for(["batch"], model_dir).use_cuda_graph is True
        assert options_for(["batch", "--no-cuda-graph"], model_dir).use_cuda_graph is False
        assert options_for(["serve"], model_dir).use_cuda_graph is True
        assert options_for(["serve", "--no-cuda-graph"], model_dir).use_cuda_graph is False


class TestEngineConstruction:
    """One factory builds the text engine, and it forwards what it was given."""

    def test_the_factory_forwards_the_process_grid_and_concurrency(self, model_dir, monkeypatch):
        seen: dict = {}

        def fake_from_pretrained(model, **kwargs):
            seen.update(model=model, **kwargs)
            return object()

        monkeypatch.setattr(cli.ContinuousBatchingEngine, "from_pretrained", fake_from_pretrained)
        options_for(["batch", "--tensor-parallel-size", "2"], model_dir).build_engine(
            max_num_seqs=4
        )

        assert seen["model"] == str(model_dir)
        assert seen["tensor_parallel_size"] == 2
        assert seen["max_num_seqs"] == 4

    def test_a_repl_asks_for_a_single_slot(self, model_dir, monkeypatch):
        """One turn is in flight at a time; 32 slots would only shrink each one's cache."""
        seen: dict = {}
        monkeypatch.setattr(
            TextEngineOptions,
            "build_engine",
            lambda self, **kwargs: seen.update(kwargs) or FakeEngine(),
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": "exit")
        args = build_parser().parse_args(["chat", "--model-dir", str(model_dir)])
        args.handler.run(args)

        assert seen["max_num_seqs"] == 1


class TestChatRepl:
    """The REPL streams the engine's deltas and always hands the engine back."""

    def test_a_turn_prints_the_deltas_it_was_given(self, monkeypatch, model_dir, capsys):
        engine = FakeEngine()

        assert run_chat(monkeypatch, model_dir, engine, ["hello", "exit"]) == 0
        assert "reply to hello" in capsys.readouterr().out
        assert engine.prompts == ["hello"]

    def test_the_engine_is_released_when_the_session_ends(self, monkeypatch, model_dir):
        """Under tensor parallelism the follower ranks exit on this and nothing else."""
        engine = FakeEngine()

        run_chat(monkeypatch, model_dir, engine, ["hi", "exit"])

        assert engine.released

    def test_a_rejected_prompt_does_not_end_the_session(self, monkeypatch, model_dir, capsys):
        """An over-long prompt is the user's problem, not a reason to drop the REPL."""
        engine = FakeEngine(reject=True)

        assert run_chat(monkeypatch, model_dir, engine, ["war and peace", "exit"]) == 0
        assert "context window" in capsys.readouterr().err
        assert engine.released

    def test_a_repetition_stop_is_explained(self, monkeypatch, model_dir, capsys):
        engine = FakeEngine(finish_reason="repeat")

        run_chat(monkeypatch, model_dir, engine, ["hi", "exit"])

        assert "repetition" in capsys.readouterr().err


class TestTensorParallelSurface:
    """Which commands take extra GPUs, and which say so plainly."""

    def test_serve_passes_the_process_grid_to_the_server(self, model_dir, monkeypatch):
        captured: dict = {}
        from lite_llama.entrypoints import api_server

        monkeypatch.setattr(
            api_server, "run_server", lambda config, host, port: captured.update(config=config)
        )
        args = build_parser().parse_args(
            ["serve", "--model-dir", str(model_dir), "--tensor-parallel-size", "2"]
        )

        assert args.handler.run(args) == 0
        assert captured["config"].tensor_parallel_size == 2

    def test_vl_chat_refuses_tensor_parallelism_instead_of_faking_it(self, model_dir):
        """Vision has no sharded path yet, and the mirror process that pretended
        otherwise is gone: refusing beats a run whose ranks quietly disagree."""
        args = build_parser().parse_args(
            [
                "vl-chat",
                "--model-dir",
                str(model_dir),
                "--image",
                "cat.png",
                "--tensor-parallel-size",
                "2",
            ]
        )

        with pytest.raises(SystemExit, match="single-GPU"):
            args.handler.run(args)

    def test_the_mirror_process_scheme_stays_retired(self):
        """Every rank used to re-derive the batch from a broadcast prompt string.

        Two encodings of the same decision, kept in step by hand, whose failure
        mode was an NCCL deadlock rather than a wrong answer. The schedule is now
        computed once and broadcast as a plan; these names must not come back.
        """
        for retired in ("_tp_mirror_worker", "_pack_sampling_params", "_unpack_sampling_params"):
            assert not hasattr(cli, retired), f"{retired} is back"


class TestDataParallelSurface:
    """``serve`` is the one command that wants more GPUs than one replica needs."""

    @pytest.fixture
    def captured_server(self, monkeypatch):
        captured: dict = {}
        from lite_llama.entrypoints import api_server

        monkeypatch.setattr(
            api_server, "run_server", lambda config, host, port: captured.update(config=config)
        )
        return captured

    def test_serve_passes_the_data_parallel_grid_to_the_server(self, model_dir, captured_server):
        args = build_parser().parse_args(
            [
                "serve",
                "--model-dir",
                str(model_dir),
                "--data-parallel-size",
                "2",
                "--load-balancer",
                "total_tokens",
            ]
        )

        assert args.handler.run(args) == 0
        assert captured_server["config"].data_parallel_size == 2
        assert captured_server["config"].load_balancer == "total_tokens"
        assert captured_server["config"].tensor_parallel_size == 1

    def test_serve_defaults_to_one_replica(self, model_dir, captured_server):
        """Without the flags a server run must stay exactly the engine it was."""
        args = build_parser().parse_args(["serve", "--model-dir", str(model_dir)])

        assert args.handler.run(args) == 0
        assert captured_server["config"].data_parallel_size == 1
        assert captured_server["config"].load_balancer == "round_robin"
