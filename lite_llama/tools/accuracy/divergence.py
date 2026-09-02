"""Where a whole-model run first leaves its HuggingFace reference.

:class:`DivergenceChecker` runs the same token ids through a checkpoint's own
``transformers`` model and through lite_llama's, hooks every decoder layer's
output on both sides, and reports the first layer whose diff leaves the numeric
noise band — plus, for that layer, whether the divergence starts in the
attention block or the MLP. A layer that goes wrong poisons every layer after
it, so only the first divergence is diagnostic; the table still shows them all.

Usage:
    lite-llama acc.divergence --model-dir my_weight/Qwen3-0.6B
    # or, as a library:
    report = find_first_divergent_layer("my_weight/Qwen3-0.6B")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ...executor.loader import DefaultModelLoader
from ...models.config import ModelConfig
from ...models.registry import ModelRegistry
from ...utils.logger import get_logger
from ..harness import Diff, SingleLayerCache, SingleLayerHarness

logger = get_logger(__name__)

#: What ``acc.divergence`` feeds the model when the caller names no prompt:
#: long enough that attention has history to work over, plain enough that any
#: tokenizer maps it without surprises.
DEFAULT_PROMPT = (
    "Explain in plain language why the sky is blue. Cover how sunlight "
    "scatters off air molecules, why shorter wavelengths scatter more, and "
    "what that means for the colors we see at sunrise and sunset."
)

#: Diff-to-reference ratio past which a layer counts as diverged. bf16 against
#: the eager reference sits around 2e-2 per layer — the band the single-layer
#: parity tests pin — so the default keeps a 2.5x separation from that band
#: while a real defect (a wrong scale, a wrong slice, wrong weights) still
#: lands an order of magnitude above it.
DEFAULT_REL_THRESHOLD = 5e-2


class PrefillCache(SingleLayerCache):
    """The paged-KV bookkeeping one whole-model prefill needs.

    The single-layer cache's padded row layout — request ``i`` owns rows
    ``[i * max_seq, (i + 1) * max_seq)`` — with one KV buffer per layer, so
    every layer of the stack keeps its own history. There is no decode step:
    this tool diffs one prefill pass, and a decode-path divergence is the
    single-layer harness's territory (it re-runs both phases on the layer it is
    handed).
    """

    def __init__(
        self,
        num_layers: int,
        batch: int,
        seq_len: int,
        *,
        kv_row: tuple[int, int],
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        # decode_steps=0: the cache holds exactly the prompt, so an accidental
        # step_decode() raises instead of silently reading past the reservation.
        super().__init__(batch, seq_len, 0, kv_row=kv_row, dtype=dtype, device=device)
        # The parent allocated the one buffer a lone layer needs; give every
        # layer of the stack its own, identical in shape.
        self.meta.kv_buffer = [torch.zeros_like(self.meta.kv_buffer[0]) for _ in range(num_layers)]


@dataclass(frozen=True)
class LayerDiff:
    """One decoder layer's output against the reference, and whether it left the band.

    ``diff`` is ``None`` only when the two sides' shapes disagreed — a wiring
    bug the note names, and a divergence no threshold could excuse.
    """

    layer: int
    diff: Diff | None
    diverged: bool
    note: str | None = None


@dataclass(frozen=True)
class SubmoduleDiff:
    """Attention and MLP output diffs at the first divergent layer.

    Inputs agree up to that layer, so whichever block's output leaves the band
    first is where the divergence originates; if both agree, the arithmetic
    between them — the fused norms, the residual add — is what moved.
    """

    self_attn: Diff
    mlp: Diff

    def culprit(self, rel_threshold: float) -> str:
        """``"self_attn"``, ``"mlp"`` or ``"norm"`` — where the divergence starts."""
        if _diverged(self.self_attn, rel_threshold):
            return "self_attn"
        return "mlp" if _diverged(self.mlp, rel_threshold) else "norm"


@dataclass(frozen=True)
class LogitsDiff:
    """Final-layer logits at the last position: the stack's user-visible end."""

    max_abs: float
    top_lite: int
    top_hf: int

    @property
    def agree(self) -> bool:
        """Whether the two sides pick the same next token."""
        return self.top_lite == self.top_hf


@dataclass(frozen=True)
class DivergenceReport:
    """Everything one divergence run measured, renderable as text or JSON."""

    model_type: str
    num_layers: int
    seq_len: int
    rel_threshold: float
    layers: tuple[LayerDiff, ...]
    first_divergent: int | None
    culprit: str | None
    submodules: SubmoduleDiff | None
    logits: LogitsDiff | None

    @property
    def ok(self) -> bool:
        """True when every layer stayed inside the band."""
        return self.first_divergent is None

    def render(self) -> str:
        """Aligned text: the layer table, the attribution, the logits."""
        lines = [
            f"{self.model_type}: {self.num_layers} layers, seq_len={self.seq_len}, "
            f"rel threshold={self.rel_threshold:.0e}"
        ]
        for row in self.layers:
            if row.diff is None:
                lines.append(f"  layer {row.layer:>3}  {row.note}")
                continue
            marker = "  <-- first divergence" if row.layer == self.first_divergent else ""
            lines.append(
                f"  layer {row.layer:>3}  max_abs={row.diff.max_abs:.3e}"
                f" mean_abs={row.diff.mean_abs:.3e} rel={row.diff.rel:.3e}{marker}"
            )
        if self.ok:
            lines.append("every layer within the band; no divergence")
        else:
            culprit = f" (culprit: {self.culprit})" if self.culprit else ""
            lines.append(f"first divergent layer: {self.first_divergent}{culprit}")
            if self.submodules is not None:
                lines.append(
                    f"    self_attn  max_abs={self.submodules.self_attn.max_abs:.3e}"
                    f" rel={self.submodules.self_attn.rel:.3e}"
                )
                lines.append(
                    f"    mlp        max_abs={self.submodules.mlp.max_abs:.3e}"
                    f" rel={self.submodules.mlp.rel:.3e}"
                )
        if self.logits is not None:
            verdict = "" if self.logits.agree else "  (disagree)"
            lines.append(
                f"  logits (last position)  max_abs={self.logits.max_abs:.3e}"
                f"  argmax lite={self.logits.top_lite} hf={self.logits.top_hf}{verdict}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready view; :meth:`render` is for the terminal, this for tooling."""

        def as_diff(diff: Diff | None) -> dict[str, float] | None:
            return (
                None
                if diff is None
                else {"max_abs": diff.max_abs, "mean_abs": diff.mean_abs, "rel": diff.rel}
            )

        return {
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "seq_len": self.seq_len,
            "rel_threshold": self.rel_threshold,
            "ok": self.ok,
            "first_divergent": self.first_divergent,
            "culprit": self.culprit,
            "layers": [
                {
                    "layer": row.layer,
                    "diff": as_diff(row.diff),
                    "diverged": row.diverged,
                    "note": row.note,
                }
                for row in self.layers
            ],
            "submodules": (
                None
                if self.submodules is None
                else {
                    "self_attn": as_diff(self.submodules.self_attn),
                    "mlp": as_diff(self.submodules.mlp),
                }
            ),
            "logits": (
                None
                if self.logits is None
                else {
                    "max_abs": self.logits.max_abs,
                    "top_lite": self.logits.top_lite,
                    "top_hf": self.logits.top_hf,
                    "agree": self.logits.agree,
                }
            ),
        }


class DivergenceChecker:
    """Locates the first decoder layer where lite_llama leaves its HF reference.

    Both sides load the same checkpoint and run the same token ids through
    their own full pipeline — embeddings, rotary, every layer — with forward
    hooks capturing each decoder layer's residual-stream output. The lite side
    is built by the production loader (:class:`DefaultModelLoader`), so a
    divergence means the serving path's arithmetic is off, not a tool-side
    reimplementation's. Each side computes its own position embeddings too: a
    RoPE difference is a real divergence, and it should surface at layer 0.

    Prefill only. The decode path runs different kernels over a different
    cache layout, and layer-local decode comparison is what
    :class:`~lite_llama.tools.harness.SingleLayerHarness` exists for —
    :meth:`harness_for` hands the named layer straight to it.

    Args:
        config: Parsed model config.
        lite_model: The lite_llama model, loaded and in eval mode.
        hf_model: The ``transformers`` model over the same weights, with eager
            attention, in eval mode.
        device: Where both sides run.
        model_dir: Checkpoint directory the models came from, when there is
            one; it is what :meth:`harness_for` needs.

    Raises:
        ValueError: If the HF model does not expose a ``model.layers`` stack.
    """

    def __init__(
        self,
        config: ModelConfig,
        lite_model: nn.Module,
        hf_model: nn.Module,
        *,
        device: str = "cuda",
        model_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.lite = lite_model
        self.hf = hf_model
        self.device = device
        self._model_dir = Path(model_dir) if model_dir is not None else None
        try:
            self._hf_layers = list(hf_model.model.layers)  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise ValueError(
                f"{type(hf_model).__name__} has no model.layers decoder stack; the "
                "checker needs a causal LM built the transformers way"
            ) from exc

    @classmethod
    def from_checkpoint(
        cls, model_dir: str | Path, *, device: str = "cuda", max_seq_len: int = 2048
    ) -> DivergenceChecker:
        """Load both sides of the comparison from a checkpoint directory.

        The lite side goes through the loader the engine uses; the HF side is
        ``AutoModelForCausalLM`` with eager attention — the same reference
        implementation the single-layer harness diffs against.
        """
        from transformers import AutoModelForCausalLM

        model_dir = Path(model_dir)
        config = ModelConfig.from_pretrained(model_dir, max_seq_len)
        lite_model = DefaultModelLoader().load_model(
            config, ModelRegistry.resolve(config.model_type).load_class(), str(model_dir), device
        )
        hf_model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=config.dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
        return cls(config, lite_model, hf_model, device=device, model_dir=model_dir)

    # ------------------------------------------------------------------ run #
    @torch.no_grad()
    def run(
        self, input_ids: torch.Tensor, *, rel_threshold: float = DEFAULT_REL_THRESHOLD
    ) -> DivergenceReport:
        """Diff one prefill over ``input_ids`` and locate the first divergence.

        Args:
            input_ids: ``[1, seq_len]`` token ids on the right device. One
                prompt at a time: the report's logits row names one argmax pair.
            rel_threshold: Diff-to-reference ratio past which a layer counts as
                diverged.

        Returns:
            The report: per-layer diffs, the first divergent layer, the
            attention-vs-MLP attribution there, and the final logits.

        Raises:
            ValueError: On a batch other than one, or when the two sides stack
                a different number of layers (which means the comparison itself
                was misassembled).
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"diff one prompt at a time: the logits row assumes batch 1 "
                f"(got {input_ids.shape[0]})"
            )
        seq_len = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(1, seq_len)
        )

        hf_layers, hf_logits = self._forward_hf(input_ids)
        lite_layers, lite_logits = self._forward_lite(input_ids, position_ids)

        rows: list[LayerDiff] = []
        first: int | None = None
        for index, (mine, theirs) in enumerate(zip(lite_layers, hf_layers, strict=True)):
            try:
                diff = Diff.between(mine, theirs)
            except ValueError as exc:  # shapes disagreed: wiring, not arithmetic
                row = LayerDiff(index, None, diverged=True, note=exc.args[0])
            else:
                row = LayerDiff(index, diff, diverged=_diverged(diff, rel_threshold))
            rows.append(row)
            if row.diverged and first is None:
                first = index

        submodules = culprit = None
        if first is not None:
            submodules = self._localise(first, input_ids, position_ids)
            culprit = submodules.culprit(rel_threshold)
            logger.info(
                "layer %d diverges first (culprit: %s); see the report for the "
                "attention-vs-MLP numbers",
                first,
                culprit,
            )

        return DivergenceReport(
            model_type=self.config.model_type,
            num_layers=len(rows),
            seq_len=seq_len,
            rel_threshold=rel_threshold,
            layers=tuple(rows),
            first_divergent=first,
            culprit=culprit,
            submodules=submodules,
            logits=_logits_row(lite_logits, hf_logits),
        )

    def harness_for(self, layer_index: int) -> SingleLayerHarness:
        """A single-layer harness on ``layer_index``, loaded from the same checkpoint.

        The bridge from a divergence report to the deeper tool: once a layer is
        named, the harness re-runs it alone — per-module timing, kernel
        dispatch, and the decode phase this prefill-only diff does not exercise.

        Raises:
            ValueError: If the checker was built without a checkpoint directory.
        """
        if self._model_dir is None:
            raise ValueError(
                "harness_for needs the checkpoint directory; build via from_checkpoint()"
            )
        harness = SingleLayerHarness(self.config, layer_index, device=self.device)
        harness.load_checkpoint(self._model_dir)
        return harness

    # -------------------------------------------------------------- forwards #
    def _forward_hf(self, input_ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """One eager forward over the HF model, every layer's output captured."""
        captured: list[torch.Tensor] = []
        handles = [
            layer.register_forward_hook(_capture_module_output(captured))
            for layer in self._hf_layers
        ]
        try:
            logits = self.hf(input_ids, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
        return captured, logits[:, -1]

    def _forward_lite(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """One prefill over the lite model, each layer's true output captured."""
        cache = PrefillCache(
            self.config.num_layers,
            1,
            input_ids.shape[1],
            kv_row=self.config.kv_cache_row,
            dtype=self.config.kv_cache_torch_dtype,
            device=self.device,
        ).begin_prefill()
        captured: list[torch.Tensor] = []
        handles = [
            layer.register_forward_hook(_capture_layer_output(captured))
            for layer in self.lite.layers
        ]
        try:
            # The last position is the one whose logits matter; projecting only
            # it skips seq_len - 1 vocabulary GEMMs the report never reads.
            logits = self.lite(
                input_ids,
                position_ids,
                cache,
                logits_positions=torch.full(
                    (1,), input_ids.shape[1] - 1, dtype=torch.long, device=input_ids.device
                ),
            )
        finally:
            for handle in handles:
                handle.remove()
        return captured, logits

    def _localise(
        self,
        layer_index: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> SubmoduleDiff:
        """Attribute the divergence at ``layer_index`` to attention, MLP, or the norms.

        Re-runs both forwards — deterministic for the same inputs and weights —
        with hooks on that one layer's ``self_attn`` and ``mlp`` blocks. At the
        first divergent layer the inputs to both blocks agree, so the block
        whose output leaves the band first is where the error originates.
        """
        lite_layer = self.lite.layers[layer_index]
        hf_layer = self._hf_layers[layer_index]
        sinks: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("lite_attn", "lite_mlp", "hf_attn", "hf_mlp")
        }
        pairs = (
            (lite_layer.self_attn, sinks["lite_attn"]),
            (lite_layer.mlp, sinks["lite_mlp"]),
            (hf_layer.self_attn, sinks["hf_attn"]),
            (hf_layer.mlp, sinks["hf_mlp"]),
        )
        handles = [
            module.register_forward_hook(_capture_module_output(sink)) for module, sink in pairs
        ]
        try:
            self._forward_lite(input_ids, position_ids)
            self._forward_hf(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        return SubmoduleDiff(
            self_attn=_diff_or_unbounded(sinks["lite_attn"][0], sinks["hf_attn"][0]),
            mlp=_diff_or_unbounded(sinks["lite_mlp"][0], sinks["hf_mlp"][0]),
        )


def find_first_divergent_layer(
    model_dir: str | Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
    device: str = "cuda",
    max_seq_len: int = 2048,
) -> DivergenceReport:
    """Tokenize ``prompt`` and locate the first layer where the two engines part ways.

    The one-call entry point behind ``lite-llama acc.divergence``: loads both
    sides from ``model_dir``, feeds them the same tokens, and returns the
    layer-by-layer report.

    Raises:
        ValueError: If the prompt tokenizes to no tokens at all.
    """
    from transformers import AutoTokenizer

    checker = DivergenceChecker.from_checkpoint(model_dir, device=device, max_seq_len=max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.numel() == 0:
        raise ValueError("the prompt tokenized to no tokens")
    # A prompt longer than the deployment's window is compared over its first
    # max_seq_len tokens; the alternative (refusing) helps nobody debugging.
    input_ids = input_ids[:, :max_seq_len]
    return checker.run(input_ids, rel_threshold=rel_threshold)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _diverged(diff: Diff, rel_threshold: float) -> bool:
    """Whether a diff is past the noise band.

    Non-finite values count as divergence unconditionally: a broken layer tends
    to produce them, and every NaN comparison is False, so a plain
    ``rel > threshold`` would wave exactly those layers through.
    """
    return not math.isfinite(diff.max_abs) or diff.rel > rel_threshold


def _capture_layer_output(sink: list[torch.Tensor]):
    """Hook that records a lite decoder layer's residual-stream output.

    The layer returns ``(mlp_output, residual)`` because the residual add lives
    inside the *next* fused norm; the block's actual output — what the HF
    side's layer returns — is the sum.
    """

    def hook(_module: nn.Module, _args: object, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        sink.append(output[0] + output[1])

    return hook


def _capture_module_output(sink: list[torch.Tensor]):
    """Hook that records a module's output, unwrapping the legacy tuple.

    transformers 5.x decoder layers and attention blocks return the tensor;
    earlier releases wrapped it — with weights, or with the cache — in a tuple.
    """

    def hook(_module: nn.Module, _args: object, output: object) -> None:
        sink.append(output[0] if isinstance(output, tuple) else output)

    return hook


def _diff_or_unbounded(actual: torch.Tensor, reference: torch.Tensor) -> Diff:
    """A diff that reads a shape mismatch as unbounded divergence."""
    try:
        return Diff.between(actual, reference)
    except ValueError:
        return Diff(float("inf"), float("inf"), float("inf"))


def _logits_row(lite_logits: torch.Tensor, hf_logits: torch.Tensor) -> LogitsDiff:
    """Compare the last position's logits: the stack's user-visible end."""
    return LogitsDiff(
        max_abs=(lite_logits.float() - hf_logits.float()).abs().max().item(),
        top_lite=int(lite_logits.argmax(dim=-1).item()),
        top_hf=int(hf_logits.argmax(dim=-1).item()),
    )
