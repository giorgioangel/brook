# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""GPU postprocessing of `PackedSkeletons`."""
from . import _runtime as runtime
from ._packed import PackedSkeletons


def _context(p):
    return runtime.context(p.vertices.device if p.on_device else None)


def components(p):
    """A connected-component id for every packed vertex (int64, on the GPU)."""
    context = _context(p)
    return context.skeleton_components(p.to_core(context))


def _call(p, method, *args):
    context = _context(p)
    return PackedSkeletons.from_core(getattr(context, method)(p.to_core(context), *args))


def remove_dust(p, dust_threshold):
    """Remove connected components whose cable length is below `dust_threshold` (physical units),
    on the GPU."""
    if dust_threshold == 0 or int(p.v_off[-1]) == 0:
        return p
    return _call(p, "remove_dust", dust_threshold)


def remove_loops(p):
    """Break the cycles of each skeleton on the GPU."""
    return _call(p, "remove_loops")


def join_close_components(p):
    """Join the connected components of each skeleton into one tree on the GPU."""
    return _call(p, "join_components")


def remove_ticks(p, tick_threshold):
    """Remove terminal branches shorter than `tick_threshold` (physical units), on the GPU."""
    return _call(p, "remove_ticks", tick_threshold)


def postprocess(p, dust_threshold=1500, tick_threshold=3500):
    """GPU postprocessing of packed skeletons: dust, loops, joins and ticks, with Brook's tie
    rules. Returns new `PackedSkeletons`."""
    return _call(p, "postprocess", dust_threshold, tick_threshold)
