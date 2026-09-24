# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Single-component tracing and preparation through Brook CUDA."""
from __future__ import annotations

import inspect

import numpy as np

from . import _runtime as runtime

_DEFAULTS = runtime.engine().trace_defaults()


def _inputs(labels, dbf, graph, void_check):
    core_engine = runtime.engine()
    ctx = runtime.context(labels.device if isinstance(labels, core_engine.DeviceArray) else None)

    def volume(value):
        if isinstance(value, core_engine.DeviceArray):
            return value
        value = np.asarray(value)
        if value.dtype == np.float16:
            value = value.astype(np.float32)  # exact host input representation
        return ctx.upload(value)

    mask, field = volume(labels), volume(dbf)
    if graph is not None:
        graph = volume(graph)
    if void_check is not None:
        filled, count = void_check
        void_check = (None if filled is None else volume(filled), int(count))
    return core_engine, ctx, mask, field, graph, void_check


def prepare(labels, DBF, scale=_DEFAULTS["scale"], const=_DEFAULTS["const"], anisotropy=(1, 1, 1),
            soma_detection_threshold=_DEFAULTS["soma_detection_threshold"],
            soma_acceptance_threshold=_DEFAULTS["soma_acceptance_threshold"],
            pdrf_scale=_DEFAULTS["pdrf_scale"], pdrf_exponent=_DEFAULTS["pdrf_exponent"],
            soma_invalidation_scale=_DEFAULTS["soma_invalidation_scale"],
            soma_invalidation_const=_DEFAULTS["soma_invalidation_const"], fix_branching=True,
            manual_targets_before=None, manual_targets_after=None, root=None, max_paths=None,
            voxel_graph=None, pre=None, void_check=None):
    """Prepare compiled fields and component metadata; return None when no root exists."""
    if pre is not None:
        raise NotImplementedError("'pre' state dictionaries are not supported; use prepare()")
    core_engine, ctx, mask, field, graph, void_check = _inputs(labels, DBF, voxel_graph, void_check)
    params = dict(scale=scale, const=const, soma_detection_threshold=soma_detection_threshold,
                  soma_acceptance_threshold=soma_acceptance_threshold, pdrf_scale=pdrf_scale,
                  pdrf_exponent=pdrf_exponent, soma_invalidation_scale=soma_invalidation_scale,
                  soma_invalidation_const=soma_invalidation_const, max_paths=max_paths)
    result = core_engine._prepare_public(ctx, mask, field, params, anisotropy, fix_branching,
                                    [tuple(int(v) for v in p) for p in manual_targets_before or []],
                                    [tuple(int(v) for v in p) for p in manual_targets_after or []],
                                    None if root is None else tuple(int(v) for v in root), graph, void_check)
    if result is not None:
        result.update(max_paths=max_paths, voxel_graph=graph, fix_branching=fix_branching,
                      scale=scale, const=const, anisotropy=anisotropy,
                      soma_invalidation_scale=soma_invalidation_scale,
                      soma_invalidation_const=soma_invalidation_const)
    return result


def trace_defaults():
    return {name: parameter.default for name, parameter in inspect.signature(prepare).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


def trace(labels, DBF, **kwargs):
    from osteoid import Skeleton

    if kwargs.pop("pre", None) is not None:
        raise NotImplementedError("'pre' state dictionaries are not supported; use prepare()")
    anisotropy = kwargs.pop("anisotropy", (1, 1, 1))
    branching = kwargs.pop("fix_branching", True)
    before = kwargs.pop("manual_targets_before", None) or []
    after = kwargs.pop("manual_targets_after", None) or []
    root = kwargs.pop("root", None)
    graph = kwargs.pop("voxel_graph", None)
    prior = kwargs.pop("void_check", None)
    core_engine, ctx, mask, field, graph, prior = _inputs(labels, DBF, graph, prior)
    result, assembled, vertices, edges = core_engine._trace_public(ctx, mask, field, kwargs, anisotropy, branching,
                                            [tuple(int(v) for v in p) for p in before],
                                            [tuple(int(v) for v in p) for p in after],
                                            None if root is None else tuple(int(v) for v in root), graph, prior)
    if not assembled:
        return Skeleton()
    a = anisotropy
    transform = np.array([[a[0], 0, 0, 0], [0, a[1], 0, 0], [0, 0, a[2], 0]], np.float32)
    return Skeleton(result.vertices if vertices is None else vertices,
                    result.edges if edges is None else edges, radii=result.radii, transform=transform)


__all__ = ["trace", "prepare", "trace_defaults"]
