# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Lower-level input handling: label formatting, upload and object masks."""
from . import DimensionError
from ._helpers import connect_points, synapses_to_targets
from ._intake import format_labels, skeletonize, trace_defaults, warmup
from ._runtime import context, engine


def in_place_safe(array):
    return array.flags.owndata


def upload_labels(labels):
    return context().upload(format_labels(labels))


def apply_object_mask(all_labels, object_ids, in_place=False):
    if object_ids is None:
        return all_labels
    return engine().mask_labels(all_labels, list(object_ids), in_place=bool(in_place) and in_place_safe(all_labels))


__all__ = ["skeletonize", "connect_points", "synapses_to_targets", "DimensionError", "warmup", "format_labels",
           "trace_defaults", "upload_labels", "apply_object_mask", "in_place_safe"]
