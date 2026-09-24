# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Compiled bounding boxes, component counts and dust metadata."""
from __future__ import annotations

from dataclasses import dataclass

from . import _runtime as runtime


@dataclass
class Components:
    ids: object
    bbmin: object
    bbmax: object
    counts: object
    n_labels: int

    def bbox_shape(self, segid):
        return tuple(int(self.bbmax[segid, a]) - int(self.bbmin[segid, a]) + 1 for a in range(3))

    def bbox_volume(self, segid):
        x, y, z = self.bbox_shape(segid)
        return x * y * z


def _device(cc_labels):
    core_engine = runtime.engine()
    if isinstance(cc_labels, core_engine.DeviceArray):
        return runtime.context(cc_labels.device), cc_labels
    context = runtime.context()
    return context, context.upload(cc_labels)


def analyze(cc_labels, dust_threshold=1000, n_labels=None):
    context, data = _device(cc_labels)
    return Components(*runtime.engine()._analyze_components(context, data, dust_threshold, n_labels))


def components_from(bbmin_h, bbmax_h, counts_h, dust_threshold, n_labels):
    ids = runtime.engine()._component_ids(counts_h, dust_threshold, n_labels)
    return Components(ids, bbmin_h, bbmax_h, counts_h, n_labels)


def boxes_and_counts(cc_labels, n_labels):
    context, data = _device(cc_labels)
    result = analyze(data, n_labels=n_labels)
    return tuple(context.upload_tensor(value) for value in (result.bbmin, result.bbmax, result.counts))


def bounding_boxes(cc_labels, n_labels=None):
    context, data = _device(cc_labels)
    result = analyze(data, n_labels=n_labels)
    return context.upload_tensor(result.bbmin), context.upload_tensor(result.bbmax)


def voxel_counts(cc_labels, n_labels=None):
    context, data = _device(cc_labels)
    return context.upload_tensor(analyze(data, n_labels=n_labels).counts)


def find_objects(cc_labels, n_labels=None):
    result = analyze(cc_labels, n_labels=n_labels)
    return [None if result.bbmax[label, 0] < 0 else
            tuple(slice(int(result.bbmin[label, axis]), int(result.bbmax[label, axis]) + 1) for axis in range(3))
            for label in range(1, result.n_labels + 1)]


__all__ = ["Components", "analyze", "components_from", "boxes_and_counts", "bounding_boxes", "voxel_counts", "find_objects"]
