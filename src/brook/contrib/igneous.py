# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Igneous integration backed by Brook (experimental; it may change in later releases).

Kimimaro runs instead only when selected explicitly (`forge="kimimaro"`), through the module
Igneous itself imported.
"""
from ._igneous import engine, execute, install, stats, uninstall

__all__ = ["engine", "execute", "install", "stats", "uninstall"]


def __getattr__(name):
    # Expose the integration's other names (e.g. its diagnostics) without forwarding
    # unknown operations to Kimimaro.
    from . import _igneous

    return getattr(_igneous, name)
