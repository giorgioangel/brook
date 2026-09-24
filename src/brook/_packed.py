# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Packed skeletons: every skeleton of a result in flat arrays, on the host or the GPU."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _runtime as runtime


@dataclass
class PackedSkeletons:
    """Skeletons in flat arrays: skeleton k has label `labels[k]`, vertices
    `vertices[v_off[k]:v_off[k+1]]` (float32, physical units), the same range of `radii`, and edges
    `edges[e_off[k]:e_off[k+1]]` (uint32, local to the skeleton). The arrays are on the GPU when
    `on_device`."""

    labels: np.ndarray
    v_off: np.ndarray
    e_off: np.ndarray
    vertices: object
    edges: object
    radii: object
    anisotropy: tuple

    def __len__(self):
        return int(self.labels.size)

    @property
    def on_device(self):
        return not isinstance(self.vertices, np.ndarray)

    @classmethod
    def from_core(cls, packed):
        result = cls(packed.labels, packed.v_off, packed.e_off, packed.vertices,
                     packed.edges, packed.radii, tuple(packed.anisotropy))
        result._core = packed
        return result

    def to_core(self, context=None):
        """The compiled engine's packed object: device buffers are borrowed, host buffers uploaded."""
        if context is None:
            context = runtime.context(self.vertices.device if self.on_device else None)
        args = (self.labels, self.v_off, self.e_off, self.vertices, self.edges, self.radii, self.anisotropy)
        if self.on_device:
            # Rebind metadata instead of caching stale labels/offsets after a
            # caller edits their arrays. Device data itself stays zero-copy.
            return runtime.engine()._pack_device(context, *args)
        return context.pack(*args)

    def to_host(self):
        """These skeletons with their arrays copied to the host (itself when already there)."""
        if not self.on_device:
            return self
        return type(self)(self.labels, self.v_off, self.e_off, self.vertices.get(),
                          self.edges.get(), self.radii.get(), self.anisotropy)

    def to_device(self):
        """These skeletons with their arrays on the current GPU (itself when already there)."""
        return self if self.on_device else type(self).from_core(self.to_core())

    def to_skeletons(self):
        """`{label: osteoid.Skeleton}` in physical space."""
        from osteoid import Skeleton

        h = self.to_host()
        a = self.anisotropy
        transform = np.array([[a[0], 0, 0, 0], [0, a[1], 0, 0], [0, 0, a[2], 0]], np.float32)
        out = {}
        for k, label in enumerate(h.labels.tolist()):
            v0, v1 = int(h.v_off[k]), int(h.v_off[k+1])
            e0, e1 = int(h.e_off[k]), int(h.e_off[k+1])
            out[label] = Skeleton(h.vertices[v0:v1], h.edges[e0:e1], h.radii[v0:v1],
                                  segid=label, transform=transform.copy(), space="physical")
        return out

    @classmethod
    def from_skeletons(cls, skeletons, anisotropy):
        """Pack `{label: osteoid.Skeleton}` on the host."""
        labels = np.fromiter(skeletons, dtype=np.int64, count=len(skeletons))
        vertices, edges, radii = [], [], []
        voff, eoff = [0], [0]
        for skeleton in skeletons.values():
            vertices.append(skeleton.vertices)
            edges.append(skeleton.edges)
            radii.append(skeleton.radii)
            voff.append(voff[-1] + len(skeleton.vertices))
            eoff.append(eoff[-1] + len(skeleton.edges))
        def concatenate(parts, shape, dtype):
            return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(shape, dtype)
        return cls(labels, np.asarray(voff, np.int64), np.asarray(eoff, np.int64),
                   concatenate(vertices, (0, 3), np.float32), concatenate(edges, (0, 2), np.uint32),
                   concatenate(radii, (0,), np.float32), tuple(float(v) for v in anisotropy))


def merge_fragments(parts, origins, on_device=False):
    """Merge `PackedSkeletons` chunk results placed at `origins` (physical units) on the GPU;
    the result stays there when `on_device`."""
    parts = list(parts)
    origins = list(origins)
    if len(parts) != len(origins):
        raise ValueError("fragment/origin count mismatch")
    aniso = parts[0].anisotropy if parts else (1., 1., 1.)
    if not parts or not any(len(p) for p in parts):
        return PackedSkeletons.from_skeletons({}, aniso)
    context = runtime.context()
    core_engine = context.merge_fragments([part.to_core(context) for part in parts], origins)
    result = PackedSkeletons.from_core(core_engine)
    return result if on_device else result.to_host()
