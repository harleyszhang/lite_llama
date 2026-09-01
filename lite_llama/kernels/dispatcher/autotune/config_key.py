"""Autotune key structure — the stable contract behind a stored config.

:class:`TuneKey` normalises (op, m/n/k buckets, dtype, GPU name) into one
hashable key, so a config measured for one shape or machine is never
silently reused for another.

Usage:
    key = TuneKey.build("fused_moe", 16, 4096, 4096, "fp16", gpu_name)
"""

from __future__ import annotations

from dataclasses import dataclass

#: M-bucket boundaries (upper-inclusive). The last boundary is open-ended.
_M_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512)


def bucket_m(m: int) -> int:
    """Quantise a row count into the nearest M-bucket (round up).

    Examples:
        >>> bucket_m(1)
        16
        >>> bucket_m(17)
        32
        >>> bucket_m(64)
        64
        >>> bucket_m(513)
        512
    """
    for b in _M_BUCKETS:
        if m <= b:
            return b
    return _M_BUCKETS[-1]


def make_shape_bucket(m: int, n: int, k: int) -> str:
    """Format a shape triple into the canonical bucket string."""
    return f"M{bucket_m(m)}_N{n}_K{k}"


def normalize_gpu_name(name: str) -> str:
    """Normalise a GPU device name for use as a key component.

    ``torch.cuda.get_device_name()`` returns strings like ``"NVIDIA A10"``
    which contain spaces; we replace them with underscores for safe use in
    filenames and JSON keys.
    """
    return name.strip().replace(" ", "_")


@dataclass(frozen=True)
class TuneKey:
    """Immutable identifier for one autotune entry.

    This is the **v0.5 stable contract** referenced by v0.6's ``perf_key``.
    Fields must not be renamed or reordered without a version bump in the
    JSON schema.

    Attributes:
        gpu: Normalised GPU name, e.g. ``"NVIDIA_A10"``.
        op: Kernel operation family, e.g. ``"fused_moe"``.
        shape_bucket: Shape string ``"M{m}_N{n}_K{k}"`` with bucketed M.
        dtype: Data type label, e.g. ``"fp16"``, ``"int8"``, ``"int4"``.
    """

    gpu: str
    op: str
    shape_bucket: str
    dtype: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "gpu": self.gpu,
            "op": self.op,
            "shape_bucket": self.shape_bucket,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> TuneKey:
        """Deserialise from a plain dict."""
        return cls(gpu=d["gpu"], op=d["op"], shape_bucket=d["shape_bucket"], dtype=d["dtype"])

    @classmethod
    def build(cls, op: str, m: int, n: int, k: int, dtype: str, gpu: str | None = None) -> TuneKey:
        """Convenience factory that applies bucketing and GPU detection.

        Args:
            op: Kernel family name.
            m: Number of activation rows (will be bucketed).
            n: Output columns.
            k: Reduction dimension.
            dtype: Dtype label string.
            gpu: GPU name override; when None, auto-detects from CUDA.
        """
        if gpu is None:
            import torch

            gpu = (
                normalize_gpu_name(torch.cuda.get_device_name(0))
                if torch.cuda.is_available()
                else "unknown"
            )
        return cls(gpu=gpu, op=op, shape_bucket=make_shape_bucket(m, n, k), dtype=dtype)
