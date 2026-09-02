"""One decoder layer, built and run on its own.

:class:`SingleLayerHarness` builds just layer ``i`` of a checkpoint,
runs prefill and decode steps on it, times the dispatched kernels, and
diffs the outputs against an HF reference — layer-local evidence
without loading the whole model.

Usage:
    report = SingleLayerHarness(config, 0).run(batch=4)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ...executor.attention_metadata import AttentionMetadata
from ...executor.loader import init_empty_parameters, materialise_parameters
from ...executor.weight_utils import hf_weight_files, hf_weights_iterator
from ...models import weights as weight_keys
from ...models.config import ModelConfig
from ...models.registry import ModelRegistry
from ...modules.quantization import RawParameter
from ...utils.logger import get_logger

logger = get_logger(__name__)

#: Standard deviation of the random weights, matching the ``initializer_range`` every
#: Llama-family config ships. Random weights make the timing and dispatch numbers valid;
#: they say nothing about accuracy, which is why the accuracy path mirrors the reference
#: layer's weights instead of inventing its own.
RANDOM_STD = 0.02


@dataclass(frozen=True)
class OpTiming:
    """Device time one module of the layer accounted for across a measured run.

    Attributes:
        name: Dotted path under the layer, e.g. ``self_attn.qkv_proj``.
        calls: How many times the module ran.
        ms: Total device time over those calls.
    """

    name: str
    calls: int
    ms: float


@dataclass(frozen=True)
class Diff:
    """How far this layer's output sits from a reference implementation's.

    ``rel`` divides by the reference's own magnitude, because an absolute difference
    means nothing without it: 1e-2 is noise on activations that peak at 40 and a broken
    layer on activations that peak at 0.05.
    """

    max_abs: float
    mean_abs: float
    rel: float

    @classmethod
    def between(cls, actual: torch.Tensor, reference: torch.Tensor) -> Diff:
        """Compare two same-shaped outputs in fp32, whatever they were computed in.

        Raises:
            ValueError: On a shape mismatch, which means the two sides were fed
                different inputs rather than that the layer is inaccurate.
        """
        if actual.shape != reference.shape:
            raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}")
        delta = (actual.float() - reference.float()).abs()
        scale = reference.float().abs().max().item()
        max_abs = delta.max().item()
        return cls(
            max_abs=max_abs,
            mean_abs=delta.mean().item(),
            rel=max_abs / scale if scale > 0 else float("inf"),
        )


@dataclass(frozen=True)
class LayerReport:
    """Everything one harness run measured, renderable as a text block."""

    model_type: str
    layer_index: int
    mlp_kind: str
    device: str
    dtype: str
    weights: str
    batch: int
    seq_len: int
    decode_steps: int
    param_bytes: int
    peak_mem_gb: float
    prefill_ms: float
    decode_ms: float
    ops: tuple[OpTiming, ...] = ()
    kernels: tuple[str, ...] = ()
    prefill_diff: Diff | None = None
    decode_diff: Diff | None = None

    def render(self) -> str:
        """Render as aligned text: geometry, cost, dispatch, accuracy."""
        lines = [
            f"layer {self.layer_index} of {self.model_type} (mlp={self.mlp_kind})",
            f"  weights      {self.weights}",
            f"  parameters   {self.param_bytes / 2**30:.3f} GiB on {self.device} as {self.dtype}",
            f"  shape        batch={self.batch} seq_len={self.seq_len}"
            f" decode_steps={self.decode_steps}",
            f"  prefill      {self.prefill_ms:.3f} ms",
            f"  decode       {self.decode_ms:.3f} ms/step",
            f"  peak memory  {self.peak_mem_gb:.3f} GiB",
        ]
        if self.ops:
            lines.append("  per-module device time (a parent includes its children):")
            width = max(len(op.name) for op in self.ops)
            lines += [f"    {op.name:<{width}}  {op.ms:8.3f} ms  x{op.calls}" for op in self.ops]
        if self.kernels:
            lines.append("  dispatched   " + ", ".join(self.kernels))
        for phase, diff in (("prefill", self.prefill_diff), ("decode", self.decode_diff)):
            if diff is not None:
                lines.append(
                    f"  {phase} vs reference: max_abs={diff.max_abs:.3e}"
                    f" mean_abs={diff.mean_abs:.3e} rel={diff.rel:.3e}"
                )
        return "\n".join(lines)


class SingleLayerCache:
    """The paged-KV bookkeeping a lone layer needs, built by hand.

    :class:`~lite_llama.executor.model_runner.ModelRunner` derives all of this from a
    profiled :class:`~lite_llama.executor.kv_cache_manager.KVCacheManager`. Here the
    layer is the only consumer and every sequence has the same length, so request ``i``
    owns rows ``[i * max_seq, (i + 1) * max_seq)`` and the request table is that
    arithmetic written out. Laying the rows out over the *padded* grid is not cosmetic:
    the layer flattens ``[batch, seq_len]`` row-major, and a packed (sum-of-lengths)
    layout would point sequence ``i`` at rows written by sequence ``i - 1``.

    Args:
        batch: Number of sequences.
        seq_len: Prompt length, identical for every sequence.
        decode_steps: How many positions past the prompt to reserve.
        kv_row: Shape of one token's cache row — ``(2 * num_kv_heads, head_dim)``
            for the paged buffer, ``(1, kv_lora_rank + qk_rope_head_dim)`` for the
            MLA latent pool — sized exactly as :class:`ModelRunner` sizes its own,
            so a layer proven here drops into service unchanged.
        dtype: KV-cache element type.
        device: Where the cache lives.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        decode_steps: int,
        *,
        kv_row: tuple[int, int],
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.max_seq = seq_len + decode_steps
        rows = batch * self.max_seq
        # One entry: the layer under test is always addressed as layer 0.
        kv_buffer = torch.zeros((rows, *kv_row), dtype=dtype, device=device)
        self.table = torch.arange(rows, dtype=torch.int32, device=device).view(batch, self.max_seq)
        self.meta = AttentionMetadata(kv_buffer=[kv_buffer], b_req_tokens_table=self.table)
        self._filled = 0

    def begin_prefill(self) -> AttentionMetadata:
        """Reserve the prompt's rows and mark the metadata as a prefill step."""
        device = self.table.device
        meta = self.meta
        meta.is_prefill = True
        meta.b_req_idx = torch.arange(self.batch, dtype=torch.int32, device=device)
        meta.b_seq_len = torch.full((self.batch,), self.seq_len, dtype=torch.int32, device=device)
        meta.max_actual_seq_len = self.seq_len
        # Offset of each sequence in the flattened token batch, not in cache rows.
        meta.b_start_loc = (torch.arange(self.batch, device=device) * self.seq_len).to(torch.int32)
        meta.cur_select_index = self.table[:, : self.seq_len].reshape(-1)
        self._filled = self.seq_len
        return meta

    def step_decode(self) -> AttentionMetadata:
        """Reserve one row per sequence for the next decode step.

        ``b_seq_len`` grows before the kernel runs, matching
        :meth:`~lite_llama.executor.model_runner.ModelRunner.decode_alloc_kv_cache`: the
        decode kernel reads history up to ``b_seq_len``, so incrementing afterwards
        would hide the token just written.

        Raises:
            RuntimeError: When the reserved ``decode_steps`` are used up.
        """
        if self._filled >= self.max_seq:
            raise RuntimeError(
                f"the cache holds {self.max_seq} positions per sequence; "
                "rebuild it with a larger decode_steps"
            )
        meta = self.meta
        meta.is_prefill = False
        meta.cur_select_index = self.table[:, self._filled].contiguous()
        assert meta.b_seq_len is not None
        meta.b_seq_len = meta.b_seq_len + 1
        self._filled += 1
        meta.max_actual_seq_len = self._filled
        return meta


class ModuleTimer:
    """Per-module device time from forward hooks, one row per module.

    Hooks rather than ``torch.profiler``: the rows then carry the same names as the
    module tree :mod:`lite_llama.tools.profiling` prints, and there is no trace to
    post-process. CUDA events are recorded on the current stream, so the measurement
    does not serialise the layer the way a ``synchronize()`` per module would.

    Only modules shallower than ``max_depth`` are hooked. Deeper is not more
    informative: a MoE block would contribute one row per expert. Ops that own no
    module at all (the fused add-and-normalise, RoPE) roll into whichever parent called
    them, so read the rows as a breakdown, not a partition — a parent's time includes
    its children's.

    Usage:
        with ModuleTimer(layer, "cuda") as timer:
            layer(...)
        rows = timer.results()
    """

    def __init__(self, module: nn.Module, device: str | torch.device, max_depth: int = 2) -> None:
        self._cuda = torch.device(device).type == "cuda"
        self._module = module
        self._names = [
            name for name, _ in module.named_modules() if name and name.count(".") < max_depth
        ]
        self._pending: dict[str, object] = {}
        self._spans: dict[str, list[tuple[object, object]]] = {}
        self._handles: list[object] = []

    def __enter__(self) -> ModuleTimer:
        for name in self._names:
            sub = self._module.get_submodule(name)
            self._handles.append(sub.register_forward_pre_hook(self._open(name)))
            self._handles.append(sub.register_forward_hook(self._close(name)))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()  # type: ignore[attr-defined]
        self._handles.clear()

    def _mark(self) -> object:
        """A point in the stream (CUDA) or on the clock (CPU)."""
        if not self._cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _open(self, name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: object) -> None:
            self._pending[name] = self._mark()

        return hook

    def _close(self, name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: object, _out: object) -> None:
            start = self._pending.pop(name, None)
            if start is not None:
                self._spans.setdefault(name, []).append((start, self._mark()))

        return hook

    def results(self) -> tuple[OpTiming, ...]:
        """Totals per module, slowest first. Synchronises once on CUDA."""
        if self._cuda and self._spans:
            torch.cuda.synchronize()
        rows = [
            OpTiming(name, len(spans), sum(self._elapsed(*span) for span in spans))
            for name, spans in self._spans.items()
        ]
        return tuple(sorted(rows, key=lambda row: row.ms, reverse=True))

    def _elapsed(self, start: object, end: object) -> float:
        if self._cuda:
            return start.elapsed_time(end)  # type: ignore[attr-defined]
        return (end - start) * 1e3  # type: ignore[operator]


class LayerReference:
    """A second implementation of the same layer, for the harness to diff against.

    The seam exists so the comparison target is a choice rather than a hard-coded
    import: :class:`~lite_llama.tools.harness.reference.HFLayerReference` is the one
    that ships, and these three methods are all a vLLM-backed reference would need
    (ROADMAP F1: "the same harness can run vLLM's corresponding layer").

    An implementation must be deterministic, must take ``(cos, sin)`` from the caller —
    computing its own would fold a RoPE difference into what is meant to be a layer
    comparison — and its :meth:`decode` must continue the sequence :meth:`prefill`
    started.
    """

    #: Short label the report prints, e.g. ``"transformers Qwen3DecoderLayer"``.
    name: str = "reference"

    def state_dict(self) -> dict[str, torch.Tensor]:
        """This layer's weights, named as HuggingFace names them."""
        raise NotImplementedError

    def prefill(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Causal forward over a whole prompt, from an empty cache."""
        raise NotImplementedError

    def decode(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """One token, attending over everything :meth:`prefill` cached."""
        raise NotImplementedError


class SingleLayerHarness:
    """One decoder layer of a registered model, built and driven in isolation.

    Construction goes through the production path — ``ModelRegistry`` for the class, the
    meta-device skeleton for the tree, ``materialise_parameters`` for the storage — so
    the layer under test is assembled by exactly the code that assembles it in service:
    the same per-layer quantisation decision, the same MoE-or-dense choice, the same
    fused projections. Reimplementing the layer here would be a second implementation to
    keep in sync, and a harness that agrees with itself proves nothing.

    Args:
        config: Parsed model config; supplies geometry, dtype and quantisation.
        layer_index: Which layer of the stack to build; negative indexes from the end.
            The choice matters — MoE models put dense MLPs on some layers, and
            quantised checkpoints exclude others from conversion.
        device: Where the layer and its cache live. The Triton kernels need a GPU, so
            ``"cpu"`` can build and load a layer but not run one.

    Raises:
        IndexError: If ``layer_index`` is outside the stack.
    """

    def __init__(self, config: ModelConfig, layer_index: int, *, device: str = "cuda") -> None:
        if not -config.num_layers <= layer_index < config.num_layers:
            raise IndexError(
                f"layer_index {layer_index} is outside the {config.num_layers}-layer stack"
            )
        self.config = config
        self.layer_index = layer_index % config.num_layers
        self.device = device

        model_cls = ModelRegistry.resolve(config.model_type).load_class()
        # The skeleton costs nothing but the module tree: parameters allocate on meta,
        # and only the chosen layer is given real storage below. Everything else — the
        # embeddings, the other layers, the lm_head — is freed when this method returns.
        with init_empty_parameters():
            skeleton = model_cls(config)
        self.layer: nn.Module = skeleton.layers[self.layer_index]
        materialise_parameters(self.layer, device, dtype=config.dtype)
        self.layer.to(device).eval()
        # Buffers, not parameters, so the skeleton built the cos/sin caches for real.
        self.rotary: nn.Module = skeleton.rotary_emb.to(device)

        self._hf_prefix: str = model_cls.hf_prefix
        self._packed = model_cls.packed_modules_mapping
        self._weights = "uninitialised"
        logger.info(
            "built layer %d of %s on %s (%.3f GiB of parameters)",
            self.layer_index,
            config.model_type,
            device,
            self.param_bytes() / 2**30,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoints_dir: str | Path,
        layer_index: int,
        *,
        device: str = "cuda",
        max_seq_len: int = 2048,
    ) -> SingleLayerHarness:
        """Build from a checkpoint directory's ``config.json`` alone; reads no weights."""
        config = ModelConfig.from_pretrained(checkpoints_dir, max_seq_len)
        return cls(config, layer_index, device=device)

    # ---------------------------------------------------------------- weights #
    @property
    def weights(self) -> str:
        """How the current parameter values got there, for the report to state."""
        return self._weights

    def checkpoint_prefix(self) -> str:
        """Checkpoint key prefix of this layer, e.g. ``"model.layers.3."``."""
        return f"{self._hf_prefix}layers.{self.layer_index}."

    def translate(self, key: str) -> weight_keys.Target:
        """Map a full-model checkpoint key onto one of *this layer's* parameters.

        Translation runs first and the layer filter second, on the *parameter* name
        rather than on the checkpoint key. The order is load-bearing: the MoE expert
        rule keys off ``.mlp.experts`` with a layer path in front of it, so stripping
        the prefix before translating would leave a stacked-expert tensor unrecognised
        and aim it at ``mlp.experts.0.gate_proj.weight``, which no parameter answers to.

        Returns:
            ``(parameter name relative to the layer, shard id)``, or ``None`` for every
            key belonging to another layer or to no layer at all.
        """
        target = weight_keys.translate_text_key(key.removeprefix(self._hf_prefix), self._packed)
        if target is None:
            return None
        name, shard_id = target
        relative = weight_keys.strip_prefix(name, f"layers.{self.layer_index}.")
        return None if relative is None else (relative, shard_id)

    def load_weights(self, checkpoint: Iterable[tuple[str, torch.Tensor]], source: str) -> None:
        """Fill this layer from a full-model checkpoint stream, ignoring the rest.

        Args:
            checkpoint: ``(key, tensor)`` pairs keyed as the model's own checkpoint is;
                every key outside this layer is skipped.
            source: Provenance label the report prints.

        Raises:
            ValueError: If any of the layer's parameters ends up unfilled or half
                filled. Inherited from :func:`lite_llama.models.weights.load_weights`,
                and the reason a one-layer load is worth routing through it: a rename
                rule that stopped matching fails here instead of producing a layer that
                runs and is quietly wrong.
        """
        weight_keys.load_weights(self.layer, checkpoint, self.translate)
        self._weights = source

    def load_checkpoint(self, checkpoints_dir: str | Path) -> None:
        """Read this layer's tensors out of a HuggingFace checkpoint directory.

        Only keys under this layer are read, and the shards are memory-mapped, so a
        671 B checkpoint costs one layer's bytes rather than all of them.
        """
        prefix = self.checkpoint_prefix()
        self.load_weights(
            hf_weights_iterator(
                checkpoints_dir,
                self.device,
                dequantize_fp8=self.config.quant is None,
                dequant_dtype=self.config.dtype,
                key_filter=lambda key: key.startswith(prefix),
            ),
            source=f"checkpoint {Path(checkpoints_dir).name}",
        )

    def load_state_dict(self, state: dict[str, torch.Tensor], source: str) -> None:
        """Fill the layer from a *bare layer* state dict in HuggingFace's naming.

        The prefix this layer would carry in a full checkpoint is put back before
        translation, so a reference layer's own ``state_dict()`` loads through the very
        rules the real thing uses — which is what makes an accuracy comparison possible
        with no checkpoint on disk at all.
        """
        prefix = self.checkpoint_prefix()
        self.load_weights(((prefix + key, value) for key, value in state.items()), source)

    def randomise(self, seed: int = 0) -> None:
        """Fill the layer with random weights, for timing and dispatch reports.

        Norm weights start at one and everything else is normal with standard deviation
        :data:`RANDOM_STD`, so activations stay in the range the kernels see in service
        instead of saturating or vanishing.

        Raises:
            ValueError: On a quantised layer. Random 8-bit blocks and random scales have
                no defined relationship, and the result would be noise presented as
                measurement; point the harness at the checkpoint instead.
        """
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, param in self.layer.named_parameters():
                if isinstance(param, RawParameter):
                    raise ValueError(
                        f"{name} is a quantised parameter; random initialisation is only "
                        "defined for an unquantised layer"
                    )
                if name.endswith("norm_weight"):
                    param.fill_(1.0)
                    continue
                sample = torch.empty(param.shape, dtype=torch.float32)
                sample.normal_(0.0, RANDOM_STD, generator=generator)
                param.copy_(sample)
        self._weights = f"random (seed={seed})"

    def param_bytes(self) -> int:
        """Bytes this layer's parameters occupy on the device."""
        return sum(p.numel() * p.element_size() for p in self.layer.parameters())

    # ------------------------------------------------------------------- run #
    def new_cache(self, batch: int, seq_len: int, decode_steps: int) -> SingleLayerCache:
        """Allocate KV bookkeeping for one prefill plus ``decode_steps`` decode steps."""
        return SingleLayerCache(
            batch,
            seq_len,
            decode_steps,
            kv_row=self.config.kv_cache_row,
            dtype=self.config.kv_cache_torch_dtype,
            device=self.device,
        )

    def hidden_states(self, batch: int, seq_len: int, seed: int = 0) -> torch.Tensor:
        """Random activations shaped like the residual stream entering this layer."""
        generator = torch.Generator().manual_seed(seed)
        sample = torch.empty(batch, seq_len, self.config.hidden_size, dtype=torch.float32)
        sample.normal_(0.0, 1.0, generator=generator)
        return sample.to(device=self.device, dtype=self.config.dtype)

    def rope(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` for ``positions``, so a reference can be fed the same pair."""
        with torch.no_grad():
            return self.rotary(hidden_states, positions)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, meta: AttentionMetadata) -> torch.Tensor:
        """Run the layer once and return the residual-stream output.

        The layer returns ``(mlp_output, residual)`` because the residual add lives
        inside the *next* fused norm; the block's actual output is their sum, which is
        what a reference implementation returns and therefore what gets compared.

        ``hidden_states`` is copied first. With no incoming residual the fused norm hands
        the input tensor back *as* the residual, and the second norm of the block then
        accumulates into it — in a full stack that in-place add is the point, but here the
        caller keeps the tensor and feeds it to the reference afterwards, which would then
        be reading this layer's post-attention sum instead of the prompt.
        """
        hidden_states = hidden_states.clone()
        positions = _positions_for(meta, hidden_states.shape[1], hidden_states.device)
        out, residual = self.layer(hidden_states, meta, 0, self.rope(hidden_states, positions))
        return out + residual

    def run(
        self,
        *,
        batch: int = 1,
        seq_len: int = 128,
        decode_steps: int = 8,
        iters: int = 3,
        reference: LayerReference | None = None,
    ) -> LayerReport:
        """Measure this layer, and optionally diff it against ``reference``.

        Args:
            batch: Sequences per step.
            seq_len: Prompt length.
            decode_steps: Decode steps after the prefill; also how many positions past
                the prompt the cache reserves.
            iters: Measured repetitions. One warm-up round runs first and is discarded —
                the first call pays Triton autotuning and the dispatch decision, neither
                of which belongs in a steady-state number.
            reference: A second implementation of the same layer to diff against, or
                ``None`` to measure only. Its weights are mirrored into this layer
                first, so the comparison is of arithmetic and not of initialisation.

        Returns:
            The measured report. ``prefill_diff``/``decode_diff`` are set only when a
            reference was given.

        Raises:
            RuntimeError: If no reference was given and the layer still holds
                uninitialised memory, which would make every number below a reading of
                whatever the allocator handed out.
        """
        if reference is not None:
            self.load_state_dict(reference.state_dict(), source=f"mirrored from {reference.name}")
        elif self._weights == "uninitialised":
            raise RuntimeError(
                "the layer holds uninitialised memory; call load_checkpoint(), "
                "load_state_dict() or randomise() before run()"
            )

        _reset_peak_memory(self.device)
        self._measure_once(batch, seq_len, decode_steps)  # warm-up, discarded

        prefill_total = decode_total = 0.0
        for _ in range(iters):
            prefill_ms, decode_ms = self._measure_once(batch, seq_len, decode_steps)
            prefill_total += prefill_ms
            decode_total += decode_ms

        with ModuleTimer(self.layer, self.device) as timer:
            self._measure_once(batch, seq_len, decode_steps)

        prefill_diff = decode_diff = None
        if reference is not None:
            prefill_diff, decode_diff = self._compare(reference, batch, seq_len)

        return LayerReport(
            model_type=self.config.model_type,
            layer_index=self.layer_index,
            mlp_kind=type(self.layer.mlp).__name__,
            device=self.device,
            dtype=str(self.config.dtype).replace("torch.", ""),
            weights=self._weights,
            batch=batch,
            seq_len=seq_len,
            decode_steps=decode_steps,
            param_bytes=self.param_bytes(),
            peak_mem_gb=_peak_memory_gb(self.device),
            prefill_ms=prefill_total / iters,
            decode_ms=decode_total / (iters * max(decode_steps, 1)),
            ops=timer.results(),
            kernels=dispatched_kernels(),
            prefill_diff=prefill_diff,
            decode_diff=decode_diff,
        )

    def _measure_once(self, batch: int, seq_len: int, decode_steps: int) -> tuple[float, float]:
        """One prefill plus ``decode_steps`` decodes on a fresh cache; wall ms of each.

        A fresh cache per repetition is the point of measuring: reusing one would let
        the decode kernel's history grow round after round, so the last repetition would
        be timing a longer sequence than the first.
        """
        cache = self.new_cache(batch, seq_len, decode_steps)
        prompt = self.hidden_states(batch, seq_len)
        token = self.hidden_states(batch, 1, seed=1)

        with _wall_clock(self.device) as prefill:
            self.forward(prompt, cache.begin_prefill())
        with _wall_clock(self.device) as decode:
            for _ in range(decode_steps):
                self.forward(token, cache.step_decode())
        return prefill.ms, decode.ms

    def _compare(self, reference: LayerReference, batch: int, seq_len: int) -> tuple[Diff, Diff]:
        """Diff the prefill and one decode step against ``reference`` on shared inputs.

        The decode step is compared as well as the prefill because the two run different
        kernels over different cache layouts: a paged-decode bug is invisible in a
        prefill-only check, and it is the failure this harness exists to catch early.
        """
        cache = self.new_cache(batch, seq_len, decode_steps=1)
        prompt = self.hidden_states(batch, seq_len)
        token = self.hidden_states(batch, 1, seed=1)
        prompt_positions = _row_positions(0, seq_len, batch, prompt.device)
        token_positions = _row_positions(seq_len, seq_len + 1, batch, token.device)

        mine_prefill = self.forward(prompt, cache.begin_prefill())
        theirs_prefill = reference.prefill(prompt, self.rope(prompt, prompt_positions))
        mine_decode = self.forward(token, cache.step_decode())
        theirs_decode = reference.decode(token, self.rope(token, token_positions))
        return Diff.between(mine_prefill, theirs_prefill), Diff.between(mine_decode, theirs_decode)


def dispatched_kernels() -> tuple[str, ...]:
    """Names of the implementations dispatch has chosen so far this process.

    Read out of the dispatcher's decision cache rather than re-ranked here, so the
    report names what ran under the shapes that actually ran.
    """
    from ...kernels.dispatcher import REGISTRY

    return tuple(sorted({decision.spec.name for decision in REGISTRY.decisions()}))


def layer_keys(checkpoints_dir: str | Path, prefix: str) -> Iterator[str]:
    """Checkpoint keys under ``prefix``, without reading a single tensor.

    Answers "does this checkpoint even name the layer I asked for" in milliseconds,
    where a wrong answer otherwise surfaces as a coverage error after a full scan.
    Safetensors shards only; a ``.bin`` checkpoint has to be unpickled to be listed.
    """
    from safetensors import safe_open

    for path in hf_weight_files(checkpoints_dir):
        if path.suffix != ".safetensors":
            continue
        with safe_open(path, framework="pt", device="cpu") as shard:
            yield from (key for key in sorted(shard.keys()) if key.startswith(prefix))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _row_positions(start: int, end: int, batch: int, device: torch.device) -> torch.Tensor:
    """``[batch, end - start]`` absolute positions, one row broadcast over the batch."""
    return torch.arange(start, end, device=device).unsqueeze(0).expand(batch, end - start)


def _positions_for(meta: AttentionMetadata, count: int, device: torch.device) -> torch.Tensor:
    """Absolute positions of the ``count`` tokens this step feeds in.

    ``b_seq_len`` is the length *including* this step's tokens, so they sit at
    ``[b_seq_len - count, b_seq_len)``. Every sequence in a harness batch has the same
    length, which is what lets one row be broadcast over the batch.
    """
    assert meta.b_seq_len is not None
    end = int(meta.b_seq_len[0].item())
    return _row_positions(end - count, end, meta.b_seq_len.shape[0], device)


@dataclass
class _Elapsed:
    """Milliseconds the timed block took, filled in on exit."""

    ms: float = 0.0


class _wall_clock:  # lower-case: used as a context manager, so it reads as a verb
    """Time a block of device work, synchronising once at each end.

    Wall time rather than CUDA events because the block is a whole phase: the
    synchronise that makes it honest happens twice per phase, not twice per op.
    """

    def __init__(self, device: str | torch.device) -> None:
        self._cuda = torch.device(device).type == "cuda"
        self._elapsed = _Elapsed()
        self._start = 0.0

    def __enter__(self) -> _Elapsed:
        if self._cuda:
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self._elapsed

    def __exit__(self, *exc: object) -> None:
        if self._cuda:
            torch.cuda.synchronize()
        self._elapsed.ms = (time.perf_counter() - self._start) * 1e3


def _reset_peak_memory(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_gb(device: str | torch.device) -> float:
    """Peak allocation since the last reset; zero off CUDA, where it has no meaning."""
    if torch.device(device).type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**30
