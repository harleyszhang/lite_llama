"""Online serving matrix: quantisation x TP/DP x CUDA graph, over the HTTP API.

The offline matrix (``benchmarks/bench_quant.py``) drives the engine in-process, so
it measures the model and nothing else. This one goes through ``lite-llama serve``:
a real subprocess, a real socket, the async engine and the continuous-batching
scheduler. Those layers are where the numbers a served deployment sees come from,
and they are exactly what an in-process benchmark cannot see.

Three things are measured per configuration, at each concurrency level:

* **TTFT** from the first SSE frame, per request, reported as mean and p99 rather
  than mean alone. Under load the tail is the number a user notices, and it is the
  one the scheduler's admission order moves.
* **TPOT** from the intervals between a request's own frames, so a request that
  waited in the queue is not charged for that wait twice.
* **Aggregate throughput** over the whole wave, which is the quantity concurrency
  is supposed to buy. Per-request latency gets *worse* as it rises; that is not a
  regression, and the two are reported side by side so the trade is visible.

Accuracy is checked three ways, all at ``temperature=0`` with every sampling field
pinned (the CLI's ``SamplingParams`` defaults and the wire protocol's are *not* the
same, so an implicit field would make these columns measure that disagreement):

* ``batch`` compares each concurrency level against the same server answering the
  same prompts one at a time. Not against the concurrency-1 *wave*: that wave holds
  a single prompt, so comparing to it scored one completion and called it a rate.
* ``dup`` compares the repeated copies of a prompt *inside one wave* against each
  other. Copies do not necessarily share a batch: the scheduler admits from its
  waiting queue under ``max_num_seqs`` and a padded token budget as requests arrive,
  so two copies can begin decoding on different steps and never see the same batch
  again. ``dup`` below 1.000 therefore does not by itself mean requests can see each
  other; the offline duplicate control below is what separates the two.
* ``offline`` compares the server's batch-of-one answers against the same checkpoint
  and scheme run in-process, one prompt per call, before the server starts. Both
  sides at batch one, so what is left is the serving path itself (chat templating,
  stop handling, the async harvest). One number per configuration, not per wave.

The same offline run also submits one prompt duplicated N times in a *single* call.
Every copy is in the waiting queue before the first step and every copy decodes the
same number of tokens, so the copies do share a batch trajectory — which the HTTP
wave cannot guarantee. If those copies agree, no state is shared between concurrent
requests, and a low ``batch`` or ``dup`` is batch-size-dependent arithmetic (a GEMM
config chosen per M, a padded CUDA-graph bucket, a split-K reduction order): bf16
argmax ties turn on the last bit, so a 1e-3 logit shift rewrites a completion. If
they disagree, state is leaking, which is a bug in a different league.

All three are *prefix* rates: the fraction of a completion reproduced before the
first differing token. Greedy decoding is chaotic, so once one token differs the
suffixes are unrelated and a position-wise rate decays toward chance regardless of
how small the original divergence was.

Every configuration gets a fresh server process. A server owns its GPUs for its
lifetime, TP followers and DP replicas are its children, and a configuration that
fails to come up is reported as a failed row rather than poisoning the next one.

Usage:
    # one configuration, three concurrency levels
    python benchmarks/bench_serving.py \\
        --model-dir $LITE_LLAMA_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 --schemes fp8

    # the matrix the offline run pairs with
    python benchmarks/bench_serving.py --model-dir ... \\
        --schemes fp16 fp8 int4 --tp 1 2 --concurrency 1 8 32
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import queue as queue_module
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.multiprocessing as mp

from benchmarks.common import PROMPTS, expand_prompts, gpu_tag

_MAX_TOKENS = 64
_CONCURRENCY = (1, 8, 32)

#: A checkpoint load, a KV profile and a graph capture, on every rank.
_READY_TIMEOUT_S = 600.0

#: One wave of requests. Generous: at concurrency 32 the scheduler admits in waves.
_WAVE_TIMEOUT_S = 600.0

_MODELZOO_ENV = "LITE_LLAMA_MODELZOO"


@dataclass(frozen=True)
class ServeSpec:
    """One served configuration. Every field appears in :attr:`label`."""

    scheme: str | None = None
    kv_cache_dtype: str = "auto"
    tp: int = 1
    dp: int = 1
    graph: bool = True

    @property
    def label(self) -> str:
        parts = [self.scheme or "bf16"]
        if self.kv_cache_dtype != "auto":
            parts.append("kv" + self.kv_cache_dtype.replace("_", "").replace("e4m3", "fp8"))
        parts.append(f"tp{self.tp}")
        if self.dp > 1:
            parts.append(f"dp{self.dp}")
        parts.append("graph" if self.graph else "eager")
        return "+".join(parts)

    @property
    def gpus_needed(self) -> int:
        return self.tp * self.dp

    def serve_argv(
        self, model_dir: str, port: int, max_seq_len: int, max_num_seqs: int
    ) -> list[str]:
        """The ``lite-llama serve`` command line for this configuration."""
        argv = [
            sys.executable,
            "-m",
            "lite_llama.cli",
            "serve",
            "--model-dir",
            model_dir,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--max-seq-len",
            str(max_seq_len),
            "--max-num-seqs",
            str(max_num_seqs),
            "--tensor-parallel-size",
            str(self.tp),
            "--data-parallel-size",
            str(self.dp),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--cuda-graph" if self.graph else "--no-cuda-graph",
        ]
        if self.scheme:
            argv += ["--quantization", self.scheme]
        return argv


@dataclass
class ServeRow:
    """One (configuration, concurrency) measurement, or the record of its failure."""

    config: str
    model: str
    concurrency: int = 0
    ttft_mean_ms: float | None = None
    ttft_p99_ms: float | None = None
    tpot_ms: float | None = None
    throughput_tps: float = 0.0
    wave_s: float = 0.0
    completed: int = 0
    issued: int = 0
    #: Prefix agreement with this server answering the same prompts one at a time.
    #: Below 1.000 means batch composition changed the tokens; ``dup_prefix`` says
    #: whether that is arithmetic or leaked state.
    batch_prefix: float | None = None
    #: Prefix agreement between repeated copies of one prompt within this wave.
    #: ``None`` when the wave was too small to contain a duplicate.
    dup_prefix: float | None = None
    #: Prefix agreement with the same scheme run offline, in-process, both sides at
    #: batch one. Per configuration: it does not vary with concurrency by construction.
    offline_prefix: float | None = None
    #: Agreement between copies of one prompt submitted offline in a single batch, so
    #: they share a batch trajectory. Below 1.000 means state is shared between
    #: concurrent requests; at 1.000 a low ``batch``/``dup`` is arithmetic.
    #: Per configuration, like ``offline_prefix``.
    dup_batch_prefix: float | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.completed > 0

    @property
    def success_rate(self) -> float:
        return self.completed / self.issued if self.issued else 0.0


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    """A port the OS says is free, so a crashed run's socket cannot break the next."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(port: int, process: subprocess.Popen, timeout_s: float) -> None:
    """Poll ``/health`` until the server answers, or explain why it never will.

    The process is checked on every pass, not only the clock: a server that dies
    during startup (an OOM, an unsupported scheme, a checkpoint whose shards do not
    divide by the TP size) would otherwise be indistinguishable from a slow one
    until the timeout expired.
    """
    import httpx

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode} during startup")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0).status_code == 200:
                return
        except Exception:
            pass  # not listening yet
        time.sleep(1.0)
    raise TimeoutError(f"server was not ready within {timeout_s:.0f}s")


def _stop(process: subprocess.Popen) -> None:
    """Terminate the server and everything it spawned.

    ``killpg`` rather than ``terminate``: a TP server's followers and a DP server's
    replicas are separate processes holding GPU memory, and signalling only the
    parent leaves them resident, which the next configuration discovers as an OOM.
    """
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), 15)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=60.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=30.0)


# --------------------------------------------------------------------------- #
# Load generation
# --------------------------------------------------------------------------- #
@dataclass
class _Result:
    """One request's timeline, as its own frames reported it."""

    ok: bool = False
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    tokens: int = 0
    text: str = ""
    error: str = ""


async def _one_request(client, url: str, model: str, prompt: str, max_tokens: int) -> _Result:
    """Stream one completion, timing it from the caller's side of the socket.

    Timing starts before the request is sent, so TTFT includes queueing — which is
    what the client experiences and what rises with concurrency. TPOT is taken from
    the gaps *between this request's own frames*, so the queue wait is charged to
    TTFT alone rather than smeared across both.
    """
    # top_p and repetition_penalty are the protocol's own defaults, sent anyway so
    # this request and the offline reference stay pinned to the same sampler even
    # if either side's default moves.
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "stream": True,
    }
    result = _Result()
    frame_times: list[float] = []
    pieces: list[str] = []
    start = time.perf_counter()
    try:
        async with client.stream("POST", url, json=body, timeout=_WAVE_TIMEOUT_S) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ").strip()
                if payload == "[DONE]":
                    break
                frame_times.append(time.perf_counter())
                chunk = json.loads(payload)
                pieces.append(chunk["choices"][0].get("text") or "")
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    if not frame_times:
        result.error = "no frames"
        return result
    result.ok = True
    result.ttft_s = frame_times[0] - start
    # One frame per decode step, so the count is the token count and the gaps
    # between them are the per-token latency.
    gaps = [b - a for a, b in itertools.pairwise(frame_times)]
    result.tpot_s = statistics.mean(gaps) if gaps else 0.0
    result.tokens = len(frame_times)
    result.text = "".join(pieces)
    return result


async def _wave(
    port: int, model: str, prompts: list[str], max_tokens: int
) -> tuple[list[_Result], float]:
    """Fire every prompt at once and wait for all of them. Returns results and wall time."""
    import httpx

    url = f"http://127.0.0.1:{port}/v1/completions"
    limits = httpx.Limits(max_connections=len(prompts) + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        start = time.perf_counter()
        results = await asyncio.gather(
            *(_one_request(client, url, model, p, max_tokens) for p in prompts)
        )
        return list(results), time.perf_counter() - start


async def _serial(port: int, model: str, prompts: list[str], max_tokens: int) -> list[str]:
    """Answer each prompt with nothing else in flight — the batch-of-one reference.

    Sent over the same socket to the same server, so the only difference from a wave
    is who else was decoding. That is what makes it a reference for batch invariance
    rather than a second measurement of the model.
    """
    import httpx

    url = f"http://127.0.0.1:{port}/v1/completions"
    texts: list[str] = []
    async with httpx.AsyncClient() as client:
        for prompt in prompts:
            result = await _one_request(client, url, model, prompt, max_tokens)
            texts.append(result.text if result.ok else "")
    return texts


# --------------------------------------------------------------------------- #
# Offline reference, in a child process so the server gets a clean GPU
# --------------------------------------------------------------------------- #
def _offline_child(payload: dict[str, Any], out: mp.Queue) -> None:
    try:
        from lite_llama import SamplingParams
        from lite_llama.engine import ContinuousBatchingEngine
        from lite_llama.engine.scheduler import DEFAULT_MAX_NUM_SEQS

        spec: ServeSpec = payload["spec"]
        kwargs: dict[str, Any] = {
            "max_seq_len": payload["max_seq_len"],
            "use_cuda_graph": spec.graph,
            "tensor_parallel_size": spec.tp,
        }
        if spec.scheme:
            kwargs["quantization"] = spec.scheme
        if spec.kv_cache_dtype != "auto":
            kwargs["kv_cache_dtype"] = spec.kv_cache_dtype
        engine = ContinuousBatchingEngine.from_pretrained(payload["model"], **kwargs)
        try:
            # Every field spelled out, because the two sides disagree on defaults:
            # SamplingParams is the CLI's (top_p 0.9, repetition_penalty 1.1) while
            # the wire protocol is OpenAI's (1.0 / 1.0). Left implicit, the offline
            # column would be measuring that disagreement instead of the serving path.
            params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_gen_len=payload["max_tokens"],
                repetition_penalty=1.0,
            )
            # One prompt per call, not one batch of all of them. This reference is
            # compared against the server's batch-of-one pass, so batch size has to be
            # held equal across the comparison or the column measures batch size too.
            texts = [
                engine.generate([prompt], params)[0].outputs[0].text
                for prompt in payload["prompts"]
            ]
            # The control for "can requests see each other". One prompt duplicated in a
            # single call: every copy is queued before the first step and every copy
            # runs the same length, so the copies share a batch trajectory, which HTTP
            # arrivals do not. Capped at the scheduler's default concurrency ceiling,
            # past which the engine admits in waves and the copies drift apart again.
            copies = min(payload["dup_copies"], DEFAULT_MAX_NUM_SEQS)
            dup_texts = (
                [
                    out.outputs[0].text
                    for out in engine.generate([payload["prompts"][0]] * copies, params)
                ]
                if copies > 1
                else []
            )
        finally:
            engine.shutdown()
    except Exception:
        out.put(("error", traceback.format_exc()))
    else:
        out.put(("ok", {"serial": texts, "dup": dup_texts}))


def _offline_texts(payload: dict[str, Any], timeout_s: float) -> tuple[str, Any]:
    """Run the same scheme in-process, before any server holds the GPUs.

    A separate process, and a *finished* one: the reference engine must have
    released its KV cache before the server profiles for its own, or the server
    sizes itself against memory that is about to be freed.
    """
    context = mp.get_context("spawn")
    out: mp.Queue = context.Queue()
    process = context.Process(target=_offline_child, args=(payload, out), daemon=False)
    process.start()
    try:
        try:
            return out.get(timeout=timeout_s)
        except queue_module.Empty:
            return "error", f"offline reference produced nothing in {timeout_s:.0f}s"
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()
            process.join(timeout=30.0)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _prefix_rate(tokenizer, want: list[str], got: list[str]) -> float | None:
    """Fraction of each completion reproduced before its first differing token.

    Position-wise over token ids rather than characters: two schemes can spell the
    same prose with different tokens, and it is the tokens that were sampled.
    """
    leading = total = 0
    for want_text, got_text in zip(want, got, strict=False):
        want_ids = tokenizer(want_text, add_special_tokens=False).input_ids
        got_ids = tokenizer(got_text, add_special_tokens=False).input_ids
        length = min(len(want_ids), len(got_ids))
        pairs = list(zip(want_ids[:length], got_ids[:length], strict=True))
        leading += next((i for i, (a, b) in enumerate(pairs) if a != b), length)
        total += length
    return leading / total if total else None


def _copies_rate(tokenizer, copies: list[str]) -> float | None:
    """Agreement between answers to one prompt that ran together."""
    if len(copies) < 2:
        return None
    return _prefix_rate(tokenizer, [copies[0]] * (len(copies) - 1), copies[1:])


def _dup_rate(tokenizer, prompts: list[str], texts: list[str]) -> float | None:
    """Agreement between copies of one prompt that shared a wave.

    The load generator cycles a fixed prompt list, so a wave wider than that list
    contains the same prompt several times. Those copies were issued together, but
    the scheduler admits as requests arrive, so they need not have shared a batch at
    every step: this bounds how far batch-dependent arithmetic moved an answer. The
    offline duplicate batch, where arrival is simultaneous by construction, is what
    tells apart arithmetic from state shared between requests.

    Reports the worst prompt's rate, not the mean: one prompt whose copies diverged
    is the finding, and averaging over the prompts that agreed hides it.
    """
    groups: dict[str, list[str]] = {}
    for prompt, text in zip(prompts, texts, strict=True):
        groups.setdefault(prompt, []).append(text)
    rates = [
        rate
        for copies in groups.values()
        for rate in [_copies_rate(tokenizer, copies)]
        if rate is not None
    ]
    return min(rates) if rates else None


def render(rows: list[ServeRow]) -> str:
    header = (
        "| Config | Conc | TTFT mean (ms) | TTFT p99 (ms) | TPOT (ms) | TPS "
        "| ok | batch | dup | offline |"
    )
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        if not r.ok:
            lines.append(f"| {r.config} | {r.concurrency} | FAILED: {r.note} |" + " |" * 7)
            continue

        def num(value, spec="{:.2f}"):
            return "—" if value is None else spec.format(value)

        lines.append(
            f"| {r.config} | {r.concurrency} | {num(r.ttft_mean_ms, '{:.1f}')} "
            f"| {num(r.ttft_p99_ms, '{:.1f}')} | {num(r.tpot_ms)} "
            f"| {r.throughput_tps:.1f} | {r.completed}/{r.issued} "
            f"| {num(r.batch_prefix, '{:.3f}')} | {num(r.dup_prefix, '{:.3f}')} "
            f"| {num(r.offline_prefix, '{:.3f}')} |"
        )
    return "\n".join(lines)


def _meta(model_dir: str, max_tokens: int) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "model_dir": model_dir,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "commit": commit,
        "command": " ".join(sys.argv),
        "max_tokens": max_tokens,
        "date": date.today().isoformat(),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class _Plan:
    """Everything one configuration's measurement needs, so the loop stays readable."""

    model_dir: str
    max_tokens: int = _MAX_TOKENS
    max_seq_len: int = 1024
    concurrency: tuple[int, ...] = _CONCURRENCY
    offline_check: bool = True
    #: Answer the prompt set one request at a time before the waves, as the reference
    #: for batch invariance. Costs one serial pass per configuration.
    batch_check: bool = True
    ready_timeout_s: float = _READY_TIMEOUT_S
    server_log_dir: Path | None = None
    rows: list[ServeRow] = field(default_factory=list)


def _measure_spec(spec: ServeSpec, plan: _Plan, tokenizer) -> list[ServeRow]:
    """Bring one configuration up, run every concurrency level, tear it down."""
    model_name = Path(plan.model_dir).name
    rows: list[ServeRow] = []

    def failed(note: str, concurrency: int = 0) -> list[ServeRow]:
        print(f"  FAILED: {note}")
        return [ServeRow(config=spec.label, model=model_name, concurrency=concurrency, note=note)]

    offline: list[str] | None = None
    dup_batch: float | None = None
    if plan.offline_check:
        print("  offline reference ...", flush=True)
        status, result = _offline_texts(
            {
                "spec": spec,
                "model": plan.model_dir,
                "prompts": expand_prompts(PROMPTS, max(plan.concurrency)),
                "dup_copies": max(plan.concurrency),
                "max_tokens": plan.max_tokens,
                "max_seq_len": plan.max_seq_len,
            },
            plan.ready_timeout_s,
        )
        if status == "ok":
            offline = result["serial"]
            dup_batch = _copies_rate(tokenizer, result["dup"])
            if dup_batch is not None:
                print(f"  offline duplicate batch (shared trajectory): {dup_batch:.3f}")
        else:
            print(f"  offline reference unavailable: {_first_line(result)}")

    port = _free_port()
    argv = spec.serve_argv(plan.model_dir, port, plan.max_seq_len, max(plan.concurrency))
    log_path = None
    log_handle = None
    if plan.server_log_dir is not None:
        plan.server_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = plan.server_log_dir / f"serve_{spec.label.replace('+', '_')}.log"
        log_handle = log_path.open("w")
    # Its own process group, so tearing the server down takes its followers and
    # replicas with it rather than orphaning them on the GPUs.
    process = subprocess.Popen(
        argv,
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_ready(port, process, plan.ready_timeout_s)
    except Exception as exc:
        _stop(process)
        if log_handle:
            log_handle.close()
        hint = f" (see {log_path})" if log_path else ""
        return failed(f"{type(exc).__name__}: {exc}{hint}")

    # The batch-of-one reference, taken from this server before any wave loads it:
    # same weights, same socket, same scheme, one request in flight. Comparing a wave
    # against the concurrency-1 *wave* instead would score a single completion, since
    # that wave holds one prompt.
    serial: list[str] | None = None
    offline_prefix: float | None = None
    try:
        if plan.batch_check:
            print("  batch-of-one reference ...", flush=True)
            serial = asyncio.run(
                _serial(
                    port,
                    model_name,
                    expand_prompts(PROMPTS, max(plan.concurrency)),
                    plan.max_tokens,
                )
            )
            if offline is not None:
                offline_prefix = _prefix_rate(tokenizer, offline, serial)
                print(f"  offline parity (batch-of-one both sides): {offline_prefix:.3f}")
        for concurrency in plan.concurrency:
            prompts = expand_prompts(PROMPTS, concurrency)
            try:
                results, wall = asyncio.run(_wave(port, model_name, prompts, plan.max_tokens))
            except Exception as exc:
                rows += failed(f"{type(exc).__name__}: {exc}", concurrency)
                continue

            good = [r for r in results if r.ok]
            row = ServeRow(
                config=spec.label,
                model=model_name,
                concurrency=concurrency,
                issued=len(results),
                completed=len(good),
                wave_s=wall,
            )
            if not good:
                row.note = _first_line(results[0].error if results else "no results")
                rows.append(row)
                print(f"  conc {concurrency}: FAILED: {row.note}")
                continue

            ttfts = sorted(r.ttft_s * 1000 for r in good)
            row.ttft_mean_ms = statistics.mean(ttfts)
            # Highest observed rather than an interpolated quantile: at concurrency
            # 8 an interpolated p99 is a fiction, and the worst case is the fact.
            row.ttft_p99_ms = ttfts[-1]
            row.tpot_ms = statistics.mean(r.tpot_s * 1000 for r in good)
            row.throughput_tps = sum(r.tokens for r in good) / wall if wall else 0.0

            # Parity needs prompt-to-completion alignment, which a dropped request
            # breaks. A partial wave gets its timings and no parity claim.
            if len(good) == len(results):
                texts = [r.text for r in results]
                row.dup_prefix = _dup_rate(tokenizer, prompts, texts)
                if serial is not None:
                    row.batch_prefix = _prefix_rate(tokenizer, serial[: len(texts)], texts)
            # A per-configuration quantity, not a per-wave one: both sides of it are
            # batch-of-one, which is what makes it a statement about the serving path
            # rather than about batch size. Repeated on every row so each row reads
            # on its own.
            row.offline_prefix = offline_prefix
            row.dup_batch_prefix = dup_batch
            rows.append(row)

            def parity(name: str, value: float | None) -> str:
                return "" if value is None else f" | {name} {value:.3f}"

            print(
                f"  conc {concurrency}: TPS {row.throughput_tps:.1f} "
                f"| TTFT {row.ttft_mean_ms:.1f}/{row.ttft_p99_ms:.1f} ms "
                f"| TPOT {row.tpot_ms:.2f} ms | ok {row.completed}/{row.issued}"
                + parity("batch", row.batch_prefix)
                + parity("dup", row.dup_prefix)
                + parity("offline", row.offline_prefix)
            )
    finally:
        _stop(process)
        if log_handle:
            log_handle.close()
    return rows


def _first_line(text: str) -> str:
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    return lines[-1][:200] if lines else "unknown failure"


def benchmark(specs: list[ServeSpec], plan: _Plan) -> list[ServeRow]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(plan.model_dir)
    visible = torch.cuda.device_count()
    model_name = Path(plan.model_dir).name
    rows: list[ServeRow] = []
    for spec in specs:
        print(f"\n=== {model_name} — {spec.label} ===", flush=True)
        if spec.gpus_needed > visible:
            note = f"needs {spec.gpus_needed} GPUs, {visible} visible"
            rows.append(ServeRow(config=spec.label, model=model_name, note=note))
            print(f"  SKIPPED: {note}")
            continue
        rows += _measure_spec(spec, plan, tokenizer)
    _check_parity(rows)
    return rows


def _check_parity(rows: list[ServeRow]) -> None:
    """Say what the parity columns mean instead of leaving three numbers side by side.

    The offline duplicate batch decides the reading, because it is the only comparison
    in which the copies provably shared a batch trajectory:

    * duplicates agree — concurrent requests cannot see each other. A ``batch`` or
      ``dup`` below 1.000 then says the answer depends on how many sequences shared a
      step, which is arithmetic: a GEMM tile chosen per M, a padded graph bucket, a
      different reduction order. bf16 argmax ties are decided by the last bit, so a
      1e-3 logit shift rewrites a completion. Not fixable by the scheduler.
    * duplicates disagree — identical prompts, queued together, same length, and the
      answers still differ. Nothing about arithmetic can do that. Something is shared
      between concurrent requests (a scratch buffer, a position offset, a sampler
      state) and the row's throughput describes an engine answering the wrong question.
    """
    leaked = [
        r for r in rows if r.ok and r.dup_batch_prefix is not None and r.dup_batch_prefix < 1.0
    ]
    if leaked:
        print("\nWARNING: copies sharing one offline batch disagreed - state is shared:")
        for row in dict.fromkeys(r.config for r in leaked):
            rate = next(r.dup_batch_prefix for r in leaked if r.config == row)
            print(f"  {row}: offline duplicate batch {rate:.3f}")
    varying = [
        r
        for r in rows
        if r.ok
        and r.batch_prefix is not None
        and r.batch_prefix < 1.0
        and (r.dup_batch_prefix is None or r.dup_batch_prefix >= 1.0)
    ]
    if varying:
        print("\nNote: greedy answers vary with batch size (arithmetic, not shared state):")
        for row in varying:
            dup = "—" if row.dup_prefix is None else f"{row.dup_prefix:.3f}"
            print(
                f"  {row.config} @ concurrency {row.concurrency}: "
                f"batch {row.batch_prefix:.3f}, in-wave dup {dup}"
            )


def _specs_from_args(args: argparse.Namespace) -> list[ServeSpec]:
    schemes = [None if s in ("None", "none", "fp16", "bf16") else s for s in args.schemes]
    graphs: list[bool] = []
    if args.cuda_graph or not args.no_cuda_graph:
        graphs.append(True)
    if args.no_cuda_graph:
        graphs.append(False)
    return [
        ServeSpec(scheme=scheme, kv_cache_dtype=kv, tp=tp, dp=dp, graph=graph)
        for scheme in schemes
        for kv in args.kv_cache_dtype
        for tp in args.tp
        for dp in args.dp
        for graph in graphs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", help="Checkpoint directory to serve")
    parser.add_argument("--schemes", nargs="+", default=["fp16", "fp8", "int4"])
    parser.add_argument(
        "--kv-cache-dtype", nargs="+", default=["auto"], choices=["auto", "fp8", "fp8_e4m3"]
    )
    parser.add_argument("--tp", nargs="+", type=int, default=[1])
    parser.add_argument("--dp", nargs="+", type=int, default=[1])
    parser.add_argument("--cuda-graph", action="store_true", help="Capture decode graphs (default)")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Serve with eager decode")
    parser.add_argument("--concurrency", nargs="+", type=int, default=list(_CONCURRENCY))
    parser.add_argument("--max-tokens", type=int, default=_MAX_TOKENS)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--no-offline-check",
        action="store_true",
        help="Skip the in-process reference run (saves one model load per configuration)",
    )
    parser.add_argument(
        "--no-batch-check",
        action="store_true",
        help="Skip the serial batch-of-one pass (saves one prompt set per configuration; "
        "leaves the batch-invariance column empty)",
    )
    parser.add_argument("--ready-timeout", type=float, default=_READY_TIMEOUT_S)
    parser.add_argument(
        "--server-log-dir",
        default="/tmp/lite_llama_serving",
        help="Where each server's stdout goes; a failed startup is diagnosed from it",
    )
    parser.add_argument("--json", help="Output path; a default under docs/benchmark_logs is used")
    args = parser.parse_args()

    if not args.model_dir:
        root = os.environ.get(_MODELZOO_ENV)
        parser.print_help()
        if root:
            print(f"\n${_MODELZOO_ENV}={root}", file=sys.stderr)
        return 0
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1
    try:
        import httpx  # noqa: F401
    except ImportError:
        print("bench_serving needs httpx: pip install 'lite_llama[serve]'", file=sys.stderr)
        return 1

    plan = _Plan(
        model_dir=args.model_dir,
        max_tokens=args.max_tokens,
        max_seq_len=args.max_seq_len,
        concurrency=tuple(sorted(set(args.concurrency))),
        offline_check=not args.no_offline_check,
        batch_check=not args.no_batch_check,
        ready_timeout_s=args.ready_timeout,
        server_log_dir=Path(args.server_log_dir) if args.server_log_dir else None,
    )
    specs = _specs_from_args(args)
    print(f"{len(specs)} configurations x {len(plan.concurrency)} concurrency levels")

    rows = benchmark(specs, plan)
    print("\n" + render(rows))

    model_name = Path(args.model_dir).name
    gpu = gpu_tag()
    out_path = Path(
        args.json
        or Path(__file__).resolve().parent.parent
        / "docs"
        / "benchmark_logs"
        / f"bench_serving_{model_name}_{gpu}_{date.today():%Y%m%d}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "meta": _meta(args.model_dir, args.max_tokens),
                "rows": [asdict(row) for row in rows],
            },
            indent=2,
        )
    )
    print(f"\nJSON saved to {out_path}")
    return 0 if any(row.ok for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
