# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Batched skeletonization of independent samples, from and to device memory."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _runtime as runtime
from ._intake import DEFAULT_TEASAR_PARAMS, _host_array, _numpy_readable, _on_device, format_labels
from ._packed import PackedSkeletons


@dataclass
class BatchedSkeletons:
    """Every skeleton of every sample, packed in sample order.

    vertices/edges/radii are one packed array set (device-resident unless `to_host()` was called);
    `labels[k]` is skeleton k's label in its own sample, `sample[k]` its sample index and
    `s_off[i]:s_off[i+1]` the skeleton range of sample i. Vertices are in each sample's own frame."""
    packed: PackedSkeletons
    labels: np.ndarray
    sample: np.ndarray
    s_off: np.ndarray
    _core: object = field(default=None, repr=False)

    def __len__(self):
        return int(self.s_off.size) - 1

    @property
    def vertices(self):
        return self.packed.vertices

    @property
    def edges(self):
        return self.packed.edges

    @property
    def radii(self):
        return self.packed.radii

    @property
    def v_off(self):
        return self.packed.v_off

    @property
    def e_off(self):
        return self.packed.e_off

    @property
    def on_device(self):
        return self.packed.on_device

    def __getitem__(self, i):
        """Sample i as a PackedSkeletons view sharing the batch's storage."""
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        lo, hi = int(self.s_off[i]), int(self.s_off[i + 1])
        if self._core is not None and self.on_device:
            return PackedSkeletons.from_core(self._core.slice(lo, hi))
        p = self.packed
        return PackedSkeletons(self.labels[lo:hi], p.v_off[lo:hi + 1] - p.v_off[lo], p.e_off[lo:hi + 1] - p.e_off[lo],
                               p.vertices[p.v_off[lo]:p.v_off[hi]], p.edges[p.e_off[lo]:p.e_off[hi]],
                               p.radii[p.v_off[lo]:p.v_off[hi]], p.anisotropy)

    def to_host(self):
        """This batch with its arrays copied to the host (itself when already there)."""
        if not self.on_device:
            return self
        return BatchedSkeletons(self.packed.to_host(), self.labels, self.sample, self.s_off)

    def to_skeletons(self):
        """list[dict[label, osteoid.Skeleton]], one dict per sample."""
        h = self.to_host()
        return [h[i].to_skeletons() for i in range(len(h))]

    def padded(self, max_skeletons=None, max_vertices=None):
        """Dense host arrays: vertices [B, S, V, 3], radii [B, S, V], labels [B, S], mask [B, S, V]."""
        h = self.to_host(); p = h.packed
        counts = np.diff(h.s_off); nv = np.diff(p.v_off)
        S = int(max_skeletons or (counts.max() if len(counts) else 0)); V = int(max_vertices or (nv.max() if len(nv) else 0))
        B = len(h)
        vertices = np.zeros((B, S, V, 3), np.float32); radii = np.zeros((B, S, V), np.float32)
        labels = np.full((B, S), -1, np.int64); mask = np.zeros((B, S, V), bool)
        for k in range(len(h.labels)):
            b, s = int(h.sample[k]), k - int(h.s_off[h.sample[k]])
            if s >= S:
                continue
            v0, v1 = int(p.v_off[k]), int(p.v_off[k + 1]); n = min(v1 - v0, V)
            vertices[b, s, :n] = p.vertices[v0:v0 + n]; radii[b, s, :n] = p.radii[v0:v0 + n]; mask[b, s, :n] = True; labels[b, s] = h.labels[k]
        return dict(vertices=vertices, radii=radii, labels=labels, mask=mask)

    @classmethod
    def _from_core(cls, core):
        return cls(PackedSkeletons.from_core(core.packed), core.labels, core.sample, core.s_off, core)


def _sample(sample, context, stream, borrow):
    """One device array: a device sample is borrowed or copied, a host sample uploaded."""
    if _on_device(sample):
        return context.upload_cuda_array(sample, stream, borrow)
    return context.upload(format_labels(np.asfortranarray(_host_array(sample))))


def _dense_host(array):
    """`_host_array`, and an ndarray for a host object that NumPy reads but that is not iterable, so
    that a dense batch is split along its first axis."""
    array = _host_array(array)
    if isinstance(array, np.ndarray) or _on_device(array) or hasattr(array, "__iter__") or not _numpy_readable(array):
        return array
    return np.asarray(array)


def _samples(samples, context, stream, borrow):
    """Device arrays for a list of samples or a dense (B, ...) array; host samples are uploaded."""
    if _on_device(samples):
        return _split_dense(context, samples, stream, borrow)
    samples = _dense_host(samples)
    if isinstance(samples, np.ndarray):
        return [context.upload(format_labels(np.asfortranarray(s))) for s in samples]
    return [_sample(s, context, stream, borrow) for s in samples]


def _graphs(voxel_graph, count, context, stream, borrow):
    """Per-sample voxel graphs aligned with the samples: [] without graphs, else one device array
    (or None) per sample, from a sequence or a dense (B, ...) array."""
    if voxel_graph is None:
        return []
    voxel_graph = _dense_host(voxel_graph)
    if _on_device(voxel_graph) or isinstance(voxel_graph, np.ndarray):
        graphs = _samples(voxel_graph, context, stream, borrow)
    else:
        graphs = [None if g is None else _sample(g, context, stream, borrow) for g in voxel_graph]
    if len(graphs) != count:
        raise ValueError(f"brook: {len(graphs)} voxel graphs for {count} samples")
    return graphs


def _split_dense(context, dense, stream, borrow):
    """A dense device array (B, ...): one device copy per sample, from its CUDA array interface."""
    cai = dense.__cuda_array_interface__
    shape, strides, typestr = tuple(cai["shape"]), cai.get("strides"), cai["typestr"]
    item = int(typestr[2:]) if typestr[2:].isdigit() else np.dtype(typestr).itemsize
    if strides is None:
        strides, acc = [], item
        for s in reversed(shape):
            strides.insert(0, acc)
            acc *= s
    base = int(cai["data"][0])

    class _View:                      # a sample of the dense array; keeps the array alive while borrowed
        def __init__(self, ptr, shp, strd):
            self.__cuda_array_interface__ = dict(shape=shp, strides=tuple(strd), typestr=typestr, data=(ptr, False), version=3)
            self.owner = dense
    return [context.upload_cuda_array(_View(base + i * strides[0], shape[1:], strides[1:]), stream, borrow) for i in range(shape[0])]


def _is_point(entry):
    """A single 3-D point, rather than one sample's sequence of points."""
    if isinstance(entry, np.ndarray):
        return entry.ndim == 1 and entry.shape[0] == 3
    if entry is None or isinstance(entry, (str, bytes)):
        return False
    try:
        values = list(entry)
    except TypeError:
        return False
    return len(values) == 3 and all(np.ndim(value) == 0 for value in values)


def _sample_targets(targets, count, name):
    """One list of 3-D points per sample, in each sample's own coordinates.

    None, an omitted argument and an empty sequence all mean "no sample has targets"; a sample
    without targets carries an empty sequence (or None) of its own."""
    if targets is None:
        return []
    entries = list(targets)
    if not entries:
        return []
    if any(_is_point(entry) for entry in entries):
        raise ValueError(f"brook: {name} is per sample in a batch: pass one sequence of points for "
                         f"each sample (an empty sequence for a sample without targets), not one flat list of points")
    if len(entries) != count:
        raise ValueError(f"brook: {name} has {len(entries)} entries for {count} samples")
    return [[] if entry is None else [tuple(int(c) for c in point) for point in entry] for entry in entries]


def skeletonize_batch(samples, teasar_params=DEFAULT_TEASAR_PARAMS, anisotropy=(1, 1, 1), dust_threshold=1000,
                      fix_branching=True, fix_borders=True, fill_holes=False, fix_avocados=False,
                      extra_targets_before=None, extra_targets_after=None, progress=False, output="packed_device",
                      stream=None, borrow=True, voxel_graph=None):
    """Skeletonize independent samples in one call; result[i] equals skeletonize(samples[i], ...).

    `samples`: a sequence of label volumes (host arrays, or device arrays with a CUDA array
    interface; shapes may differ) or a dense array with the samples along its first axis; an array
    that exposes only DLPack is host input when it is in host memory, as in `skeletonize`.
    `anisotropy` and target points follow each sample's axis order, as in `skeletonize`.
    Fortran-contiguous device samples are read in place (`borrow=True`), others copied; `stream`
    is the producer's stream handle that Brook waits for (default: the arrays' own
    CUDA-array-interface stream, else the legacy default stream).
    `extra_targets_before`/`extra_targets_after` are per sample: a sequence aligned with `samples`
    holding each sample's own points (an empty sequence or None for a sample without targets).
    `voxel_graph`: per-sample voxel connectivity graphs (cc3d convention, uint32) aligned with the
    samples, as a sequence (host or device arrays; None for a sample without one) or a dense array
    with the samples along its first axis; sample i is then skeletonized with
    `voxel_graph=voxel_graph[i]`. Returns BatchedSkeletons on the device (`output="packed_device"`),
    on the host (`"packed"`) or as a list of dicts of osteoid skeletons (`"skeletons"`).
    `progress` is accepted for symmetry with `skeletonize` and has no effect."""
    if output not in ("packed_device", "packed", "skeletons"):
        raise ValueError("brook: output must be 'packed_device', 'packed' or 'skeletons'")
    runtime.require_gpu()
    context = runtime.context()
    device_samples = _samples(samples, context, stream, borrow)
    before = _sample_targets(extra_targets_before, len(device_samples), "extra_targets_before")
    after = _sample_targets(extra_targets_after, len(device_samples), "extra_targets_after")
    device_graphs = _graphs(voxel_graph, len(device_samples), context, stream, borrow)
    core = context.skeletonize_batch_device(
        device_samples, dict(teasar_params), np.array(anisotropy, np.float32), dust_threshold, fix_branching,
        fix_borders, before, after, fill_holes, fix_avocados, voxel_graphs=device_graphs)
    result = BatchedSkeletons._from_core(core)
    if output == "packed_device":
        return result
    result = result.to_host()
    return result.to_skeletons() if output == "skeletons" else result


__all__ = ["skeletonize_batch", "BatchedSkeletons"]
