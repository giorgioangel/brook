# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Brook: TEASAR skeletonization of 3D label volumes on NVIDIA GPUs."""
from __future__ import annotations

__version__ = "0.1.0"

DEFAULT_TEASAR_PARAMS = {
    "scale": 1.5,
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500,
    "soma_detection_threshold": 750,
    "soma_invalidation_const": 300,
    "soma_invalidation_scale": 2,
}


class DimensionError(Exception):
    """The input volume has unsupported dimensions."""


from ._helpers import (  # noqa: E402
    connect_points,
    cross_sectional_area,
    join_close_components,
    oversegment,
    postprocess,
    postprocess_many,
    synapses_to_targets,
)
from ._intake import skeletonize, warmup  # noqa: E402
from ._packed import PackedSkeletons  # noqa: E402
from .batch import BatchedSkeletons, skeletonize_batch  # noqa: E402


def __getattr__(name):
    # The worker-pool names import brook._pool (and multiprocessing) on first access.
    if name in {"GpuPool", "skeletonize_chunked", "iter_chunks", "chunk_shape_for"}:
        from . import _pool

        return getattr(_pool, name)
    raise AttributeError(f"module 'brook' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "skeletonize", "skeletonize_batch", "skeletonize_chunked", "connect_points", "warmup",
    "BatchedSkeletons", "PackedSkeletons", "GpuPool", "iter_chunks", "chunk_shape_for",
    "postprocess", "postprocess_many", "join_close_components", "synapses_to_targets",
    "cross_sectional_area", "oversegment", "DEFAULT_TEASAR_PARAMS", "DimensionError", "__version__",
]
