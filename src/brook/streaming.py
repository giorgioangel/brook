# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Streamed CCL and EDT entrypoints for host volumes."""
from __future__ import annotations

import numpy as np

from . import _runtime as runtime
from ._intake import _budget_bytes, format_labels


def default_budget_bytes():
    requested = _budget_bytes()
    return requested or max(runtime.memory_info()[0] // 4, 64 << 20)


def _budget(value):
    return default_budget_bytes() if not value else max(1, int(value))


def connected_components(labels_host, budget_bytes=None):
    labels = format_labels(labels_host)
    if labels.dtype.kind == "f":
        labels, _ = runtime.engine().normalize_labels(labels)
    components, mapping, _ = runtime.context().connected_components_streamed(labels, budget_bytes=_budget(budget_bytes))
    return components, mapping


def edt(labels_host, anisotropy=(1, 1, 1), black_border=False, budget_bytes=None):
    return runtime.context().edt_streamed(format_labels(labels_host), np.asarray(anisotropy, np.float32),
                                         bool(black_border), budget_bytes=_budget(budget_bytes))


__all__ = ["connected_components", "edt", "default_budget_bytes"]


def analyze(cc_host, dust_threshold=1000, n_labels=None, budget_bytes=None):
    from .preamble import Components

    return Components(*runtime.engine()._analyze_components_streamed(
        runtime.context(), format_labels(cc_host), dust_threshold, n_labels, _budget(budget_bytes)))


__all__.append("analyze")
