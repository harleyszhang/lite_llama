"""Optional sequence-parallel rewrite for all-reduce/RMSNorm seams.

This framework runs eager + CUDA graphs and does not compile through inductor,
so vLLM's ``SequenceParallelismPass`` — which matches the pattern on an FX graph
— has no graph to match on here. The equivalent is a *module-level* pass: it
walks the ``nn.Module`` tree, finds the seams where a row-parallel projection's
all-reduce feeds an RMSNorm, and marks them to run the reduce-scatter ->
local-norm -> all-gather decomposition
(:func:`~rapid_llm.kernels.ops.layernorm.skip_rmsnorm.sequence_parallel_allreduce_rmsnorm`).

The transformation is the one vLLM performs::

    Input -> AllReduce -> RMSNorm -> Output
    becomes
    Input -> ReduceScatter -> RMSNorm -> AllGather -> Output

The rewrite is opt-in because it adds a second all-gather for the residual and
does not always outperform the standard all-reduce. Runtime eligibility also
keeps it away from token counts that cannot be sharded evenly and from active
TBO/L3 communication-overlap policies.

Usage:
    SequenceParallelPass().apply(model)   # after the model is built, before capture
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    import torch.nn as nn

_log = get_logger(__name__)

#: Environment variable switching the sequence-parallel pass on.
SP_ENV = "RAPID_LLM_SEQUENCE_PARALLEL"


def sequence_parallel_enabled() -> bool:
    """Whether the sequence-parallel pass is active for this process.

    Disabled by default until a deployment opts in with a measured workload.
    """
    raw = os.environ.get(SP_ENV, "0").strip().lower()
    return raw not in ("", "0", "false", "off")


def is_sequence_parallel(module: nn.Module) -> bool:
    """Whether ``module`` was marked for the sequence-parallel decomposition.

    The decoder blocks read this at their all-reduce/norm seam instead of
    re-deriving the decision; a module the pass never visited is ``False``.
    """
    return bool(getattr(module, "_sequence_parallel", False))


def sequence_parallel_eligible(
    module: nn.Module, *, num_tokens: int, world_size: int, overlap_active: bool
) -> bool:
    """Whether this invocation can safely use the token-sharded rewrite."""
    return (
        is_sequence_parallel(module)
        and world_size > 1
        and num_tokens > 0
        and num_tokens % world_size == 0
        and not overlap_active
    )


class SequenceParallelPass:
    """Module-level graph pass marking every ``AllReduce->RMSNorm`` seam.

    Walks the model tree and marks each seam where a row-parallel projection's
    all-reduce feeds an RMSNorm to run the sequence-parallel decomposition
    instead. A marked seam then:

    * skips the row-parallel all-reduce (the partial sum stays unreduced), and
    * runs :func:`~rapid_llm.kernels.ops.layernorm.skip_rmsnorm.sequence_parallel_allreduce_rmsnorm`
      at the norm — reduce-scatter the partial, norm only this rank's token
      segment, all-gather the result.

    This is the eager-mode analogue of vLLM's ``SequenceParallelismPass``: the
    same pattern, recognised on the module graph rather than an FX graph.

    The pass only marks seams; each invocation still checks token divisibility
    and conflicting overlap policies. It is a no-op when TP is off or when
    :data:`SP_ENV` disables it.

    Args:
        enabled: Override :func:`sequence_parallel_enabled`; ``None`` reads the env.
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = sequence_parallel_enabled() if enabled is None else enabled
        #: Names of the modules this pass marked, for observability.
        self.matched: list[str] = []

    def apply(self, model: nn.Module) -> int:
        """Mark every ``AllReduce->RMSNorm`` seam in ``model``.

        Args:
            model: The built model (before CUDA-graph capture).

        Returns:
            The number of seams marked. ``0`` when the pass is disabled or TP
            is off.
        """
        self.matched.clear()
        if not self.enabled:
            return 0
        from .parallel_state import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() <= 1:
            # No peers to scatter across; the decomposition would degenerate to
            # the plain fused norm, so leave the seams on the all-reduce path.
            return 0

        for name, module in model.named_modules():
            if self._is_ar_rmsnorm_seam(module):
                module._sequence_parallel = True
                self.matched.append(name)

        _log.info("SequenceParallelPass marked %d AllReduce->RMSNorm seams", len(self.matched))
        return len(self.matched)

    @staticmethod
    def _is_ar_rmsnorm_seam(module: nn.Module) -> bool:
        """Whether ``module`` is a decoder block with an o_proj all-reduce -> norm seam.

        The seam is a block that owns the two-stage split (an attention stage
        whose row-parallel output projection is followed by a post-attention
        RMSNorm). Recognised structurally rather than by class, so every
        :class:`~rapid_llm.models.base.DecoderLayer` variant — dense, MoE, MLA —
        matches without the pass importing the model layer (which would cycle).
        """
        return callable(getattr(module, "forward_attn_stage", None)) and callable(
            getattr(module, "_post_attention_norm", None)
        )
