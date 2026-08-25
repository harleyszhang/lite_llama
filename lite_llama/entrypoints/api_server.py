"""OpenAI-compatible HTTP server over :class:`AsyncLLMEngine`.

Endpoints: ``/v1/models``, ``/v1/completions``, ``/v1/chat/completions`` (both
streaming and not) and ``/health``. Being wire-compatible means the official
``openai`` client, ``curl`` and anything built for vLLM's server work unchanged.

The layer is deliberately thin. It translates JSON to
:class:`~lite_llama.engine.sampler.SamplingParams`, applies the chat template, and
turns the engine's chunks into SSE frames — no batching, queuing or scheduling
logic lives here, because all of that is the engine's job and duplicating any of
it would mean two policies to keep in agreement.

Usage:
    lite-llama serve --model-dir my_weight/Qwen2.5-0.5B --port 8000
    curl localhost:8000/v1/completions -H 'Content-Type: application/json' \\
         -d '{"model":"qwen","prompt":"Hello","max_tokens":32}'
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from ..engine.async_data_parallel import AsyncDataParallelEngine
from ..engine.async_engine import AsyncLLMEngine, StreamedOutput
from ..engine.sampler import SamplingParams
from ..utils.logger import get_logger
from ..utils.prompt_templates import get_prompter
from .protocol import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    UsageInfo,
    _request_id,
)

logger = get_logger(__name__)

# Terminator every OpenAI-compatible stream ends with; clients watch for it.
_SSE_DONE = "data: [DONE]\n\n"


def _require_fastapi() -> Any:
    """Import FastAPI, explaining the extra when it is missing.

    Serving is an optional extra, so the dependency is imported here rather than
    at module import: ``import lite_llama`` must keep working on a machine that
    only ever runs offline generation.
    """
    try:
        import fastapi
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "the API server needs FastAPI and uvicorn; install them with "
            "`pip install 'lite-llama[serve]'`"
        ) from exc
    return fastapi


@dataclass
class ServerConfig:
    """How to build the engine a server run sits on.

    Attributes:
        model_dir: Checkpoint directory.
        served_model_name: Name reported by ``/v1/models`` and echoed in
            responses; defaults to the checkpoint directory's name.
        max_seq_len: Context window, and the per-slot KV cache size.
        max_num_seqs: Concurrency ceiling — how many requests may decode together.
        max_num_batched_tokens: Padded token budget for one prefill group.
        max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
        device: Torch device string.
        use_cuda_graph: Capture decode CUDA graphs.
        quantization: Runtime weight quantisation for fp16 checkpoints.
        tensor_parallel_size: GPUs this replica's weights are split over. Above 1
            the engine spawns the follower ranks itself and the server is still
            one process with one scheduler.
        kv_cache_dtype: KV-cache element type (``"auto"`` or an fp8 spelling).
        data_parallel_size: Whole-model replicas serving this one endpoint,
            combined with ``tensor_parallel_size`` into the usual
            ``dp x tp`` GPU grid. Above 1 the lifespan builds an
            :class:`~lite_llama.engine.async_data_parallel.AsyncDataParallelEngine`
            instead — the replicas multiply concurrent decode, which is what a
            server is for; the fields above apply *per replica*.
        load_balancer: Which replica each request is routed to, one of
            :data:`~lite_llama.engine.dp_load_balancer.LOAD_BALANCERS`.
        chat_template: ``True`` applies the tokenizer's chat template to
            ``/v1/chat/completions`` messages. Turn it off for base models, which
            have no template and degenerate when given one.
    """

    model_dir: str
    served_model_name: str | None = None
    max_seq_len: int = 2048
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    max_gpu_num_blocks: int | None = None
    device: str = "cuda"
    use_cuda_graph: bool = True
    quantization: str | None = None
    tensor_parallel_size: int = 1
    kv_cache_dtype: str = "auto"
    data_parallel_size: int = 1
    load_balancer: str = "round_robin"
    chat_template: bool = True

    @property
    def model_name(self) -> str:
        from pathlib import Path

        return self.served_model_name or Path(self.model_dir).name


class OpenAIServer:
    """Wire-protocol adapter: OpenAI JSON in, engine calls out.

    Args:
        engine: The async engine to serve from — one replica's
            :class:`~lite_llama.engine.async_engine.AsyncLLMEngine` or the
            data-parallel front end over several. Either answers with the same
            streamed chunks, which is all this layer reads.
        model_name: Name echoed back in responses.
        chat_template: Whether to apply the tokenizer's chat template.
    """

    def __init__(
        self,
        engine: AsyncLLMEngine | AsyncDataParallelEngine,
        model_name: str,
        *,
        chat_template: bool = True,
    ) -> None:
        self.engine = engine
        self.model_name = model_name
        self._prompter = get_prompter(engine.tokenizer) if chat_template else None

    # ----------------------------------------------------------------- models #
    def list_models(self) -> ModelList:
        return ModelList(data=[ModelCard(id=self.model_name)])

    # ------------------------------------------------------------ completions #
    async def completions(self, body: CompletionRequest):
        params = body.to_sampling_params()
        if body.stream:
            return self._stream_completion(body.prompt, params)
        return await self._full_completion(body.prompt, params)

    async def _full_completion(self, prompt: str, params: SamplingParams):
        final = await self.engine.generate_text(prompt, params)
        return CompletionResponse(
            model=self.model_name,
            choices=[CompletionChoice(text=final.text, finish_reason=final.finish_reason)],
            usage=self._usage(final),
        )

    async def _stream_completion(self, prompt: str, params: SamplingParams) -> AsyncIterator[str]:
        response_id = _request_id("cmpl")
        async for chunk in self.engine.generate(prompt, params):
            frame = CompletionChunk(
                id=response_id,
                model=self.model_name,
                choices=[CompletionChoice(text=chunk.delta, finish_reason=chunk.finish_reason)],
            )
            yield f"data: {frame.model_dump_json()}\n\n"
        yield _SSE_DONE

    # ------------------------------------------------------- chat completions #
    async def chat_completions(self, body: ChatCompletionRequest):
        prompt = self._render_chat(body)
        params = body.to_sampling_params()
        if body.stream:
            return self._stream_chat(prompt, params)
        return await self._full_chat(prompt, params)

    async def _full_chat(self, prompt: str, params: SamplingParams):
        final = await self.engine.generate_text(prompt, params)
        return ChatCompletionResponse(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=final.text),
                    finish_reason=final.finish_reason,
                )
            ],
            usage=self._usage(final),
        )

    async def _stream_chat(self, prompt: str, params: SamplingParams) -> AsyncIterator[str]:
        response_id = _request_id("chatcmpl")

        def frame(delta: ChatCompletionDelta, reason: str | None) -> str:
            chunk = ChatCompletionChunk(
                id=response_id,
                model=self.model_name,
                choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=reason)],
            )
            return f"data: {chunk.model_dump_json()}\n\n"

        # OpenAI opens with a role-only delta, before any text exists.
        yield frame(ChatCompletionDelta(role="assistant"), None)
        async for chunk in self.engine.generate(prompt, params):
            yield frame(ChatCompletionDelta(content=chunk.delta), chunk.finish_reason)
        yield _SSE_DONE

    def _render_chat(self, body: ChatCompletionRequest) -> str:
        """Turn chat messages into a single prompt string.

        Without a template (base models) the turns are concatenated verbatim,
        which is the only honest thing to do: inventing ``<|im_start|>`` markers a
        base model never saw makes it echo the role names back.
        """
        if self._prompter is None:
            return "\n".join(message.content for message in body.messages)
        return self._prompter.apply([{"role": m.role, "content": m.content} for m in body.messages])

    def _usage(self, final: StreamedOutput) -> UsageInfo:
        """Report the token counts the engine already keeps.

        Re-encoding the texts would be both slower and subtly wrong: decode does
        not round-trip through encode at token boundaries, so the honest numbers
        are the ones the engine tokenised and sampled.
        """
        return UsageInfo(
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
            total_tokens=final.prompt_tokens + final.completion_tokens,
        )


def build_app(config: ServerConfig, engine: AsyncLLMEngine | AsyncDataParallelEngine | None = None):
    """Assemble the FastAPI application.

    Args:
        config: Engine and serving options.
        engine: Pre-built engine; when omitted one is loaded on startup — one
            replica's async engine, or the data-parallel front end when
            ``config.data_parallel_size`` is above 1. Injecting a fake here is
            what lets the protocol layer be tested without a GPU or a checkpoint.
    """
    fastapi = _require_fastapi()
    from fastapi.responses import JSONResponse, StreamingResponse

    owns_engine = engine is None
    state: dict[
        str, AsyncLLMEngine | AsyncDataParallelEngine | OpenAIServer | None
    ] = {"engine": engine, "server": None}

    @asynccontextmanager
    async def lifespan(_app):
        if state["engine"] is None:
            if config.data_parallel_size > 1:
                # ``device`` is deliberately absent: a replica's device is its
                # position in the grid, and the coordinator loads no model.
                logger.info(
                    "loading %s for serving on %d replicas",
                    config.model_dir,
                    config.data_parallel_size,
                )
                state["engine"] = AsyncDataParallelEngine(
                    model=config.model_dir,
                    data_parallel_size=config.data_parallel_size,
                    load_balancer=config.load_balancer,
                    max_seq_len=config.max_seq_len,
                    max_num_seqs=config.max_num_seqs,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    max_gpu_num_blocks=config.max_gpu_num_blocks,
                    use_cuda_graph=config.use_cuda_graph,
                    quantization=config.quantization,
                    tensor_parallel_size=config.tensor_parallel_size,
                    kv_cache_dtype=config.kv_cache_dtype,
                )
            else:
                logger.info("loading %s for serving", config.model_dir)
                state["engine"] = AsyncLLMEngine.from_pretrained(
                    config.model_dir,
                    max_seq_len=config.max_seq_len,
                    max_num_seqs=config.max_num_seqs,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    max_gpu_num_blocks=config.max_gpu_num_blocks,
                    device=config.device,
                    use_cuda_graph=config.use_cuda_graph,
                    quantization=config.quantization,
                    tensor_parallel_size=config.tensor_parallel_size,
                    kv_cache_dtype=config.kv_cache_dtype,
                )
        active: AsyncLLMEngine | AsyncDataParallelEngine = state["engine"]
        active.start()
        state["server"] = OpenAIServer(
            active, config.model_name, chat_template=config.chat_template
        )
        yield
        if owns_engine:
            await active.shutdown()

    app = fastapi.FastAPI(title="lite_llama", lifespan=lifespan)

    def server() -> OpenAIServer:
        return state["server"]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelList)
    async def models() -> ModelList:
        return server().list_models()

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        result = await server().completions(body)
        if body.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return result

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        result = await server().chat_completions(body)
        if body.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return result

    @app.exception_handler(ValueError)
    async def value_error_handler(_request, exc: ValueError):
        # Prompts the engine refuses (empty, or past the context window) are
        # client errors; without this they would surface as a 500.
        return JSONResponse(
            status_code=400, content={"error": {"message": str(exc), "type": "invalid_request"}}
        )

    return app


def run_server(config: ServerConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve until interrupted."""
    _require_fastapi()
    import uvicorn

    uvicorn.run(build_app(config), host=host, port=port, log_level="info")


def parse_sse(payload: str) -> list[dict[str, Any]]:
    """Decode an SSE body into its JSON frames, dropping the ``[DONE]`` sentinel.

    Lives here rather than in the tests so that the framing rules have exactly one
    definition; a test that re-implemented the split would pass while the server
    emitted subtly malformed frames.
    """
    frames = []
    for line in payload.splitlines():
        if not line.startswith("data: "):
            continue
        body = line.removeprefix("data: ").strip()
        if body == "[DONE]":
            continue
        frames.append(json.loads(body))
    return frames
