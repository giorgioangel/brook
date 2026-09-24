# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""`skeletonize` and `warmup`: input handling around the compiled engine."""
from __future__ import annotations

import os
import re
import warnings

import numpy as np

from . import DEFAULT_TEASAR_PARAMS, DimensionError
from . import _runtime as runtime
from ._packed import PackedSkeletons

_INCORE_BYTES_PER_VOXEL = 14
_INCORE_MEMORY_FRACTION = 0.5
# DLPack device types in host memory, which np.from_dlpack reads in place: kDLCPU, kDLCUDAHost
# (pinned) and kDLROCMHost. Managed memory (kDLCUDAManaged) is device input.
_DLPACK_HOST_DEVICES = (1, 3, 11)


def format_labels(labels, in_place=False):
    """Adapt host I/O and dimensions without rearranging contiguous input bytes."""
    if any(base.__module__.startswith("crackle") for base in type(labels).__mro__):
        labels = labels[:]
    labels = np.asarray(labels)
    if labels.dtype == bool:
        labels = labels.view(np.uint8)
    while labels.ndim > 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim > 3:
        raise DimensionError(f"{labels.ndim} dimensions is not supported (max 3).")
    while labels.ndim < 3:
        labels = labels[..., np.newaxis]
    return labels


def trace_defaults():
    return runtime.engine().trace_defaults()


def _use_streaming(streaming, labels):
    if streaming is not None:
        return bool(streaming)
    env = os.environ.get("BROOK_STREAMING", "").strip().lower()
    if env:
        return env not in ("0", "off", "false", "no")
    free, _ = runtime.memory_info()
    needed = int(labels.size) * (labels.dtype.itemsize + _INCORE_BYTES_PER_VOXEL)
    if needed > _INCORE_MEMORY_FRACTION * free:
        warnings.warn(f"brook: a {tuple(labels.shape)} volume does not fit in GPU memory; it is streamed, "
                      "with single-volume results and bounded component crops. "
                      "brook.skeletonize_chunked() processes overlapping chunks at in-core speed.", stacklevel=3)
        return True
    return False


def _parse_stream_budget_mb(value):
    """Bytes for a BROOK_STREAM_BUDGET_MB value, with the rule of the C API: None (unset) gives 0,
    the default (a quarter of free GPU memory, at least 64 MiB); any other value must be a positive
    integer number of MiB."""
    if value is None:
        return 0
    if re.fullmatch("[0-9]+", value) is None or not 0 < int(value) <= np.iinfo(np.uintp).max >> 20:
        raise ValueError(f"invalid BROOK_STREAM_BUDGET_MB={value!r}: expected a positive integer "
                         "number of MiB")
    return int(value) << 20


def _budget_bytes():
    return _parse_stream_budget_mb(os.environ.get("BROOK_STREAM_BUDGET_MB"))


def _empty(output, anisotropy):
    return {} if output == "skeletons" else PackedSkeletons.from_skeletons({}, anisotropy)


def _objects(core_engine, anisotropy):
    from osteoid import Skeleton

    a = anisotropy
    transform = np.array([[a[0], 0, 0, 0], [0, a[1], 0, 0], [0, 0, a[2], 0]], np.float32)
    return {int(s.id): Skeleton(s.vertices, s.edges, radii=s.radii, segid=int(s.id),
                                space="physical", transform=transform.copy()) for s in core_engine}


def _execute(method, labels, options, progress):
    if not progress:
        return method(labels, **options)
    from tqdm import tqdm

    with tqdm(total=1, desc="Skeletonizing (brook)", unit="volume") as bar:
        result = method(labels, **options)
        bar.update(1)
    return result


def skeletonize(all_labels, teasar_params=DEFAULT_TEASAR_PARAMS, anisotropy=(1, 1, 1),
                object_ids=None, dust_threshold=1000, progress=True, fix_branching=True,
                in_place=False, fix_borders=True, parallel=1, parallel_chunk_size=100,
                extra_targets_before=(), extra_targets_after=(), fill_holes=False,
                fix_avocados=False, voxel_graph=None, streaming=None, output="skeletons", stream=None, borrow=True):
    """Skeletonize a 3D label volume on an NVIDIA GPU; a drop-in for `kimimaro.skeletonize`.

    Takes Kimimaro's arguments and returns `{label: osteoid.Skeleton}` by default. `parallel` and
    `parallel_chunk_size` are accepted for compatibility and ignored. As in Kimimaro, keys left out
    of `teasar_params` take the defaults of `kimimaro.trace` (for example `pdrf_exponent=16`), not
    those of `brook.DEFAULT_TEASAR_PARAMS`. Brook adds:

    - `output`: `"skeletons"` (default), `"packed"` (host `PackedSkeletons`) or `"packed_device"`
      (packed arrays left on the GPU).
    - `streaming`: `None` streams when Brook's memory estimate exceeds half the free device memory
      (the `BROOK_STREAMING` environment variable overrides this); `True`/`False` select the mode.
      Streaming does not support `voxel_graph`, `fill_holes` or `fix_avocados`.
    - Device input: an array with a CUDA array interface is read on the GPU. A
      Fortran-contiguous array on the context's device is read in place (`borrow=True`); `stream`
      is the producer's stream handle, which Brook waits for before reading. `object_ids`,
      streaming and floating-point labels are host-input features. An array that exposes only
      DLPack is host input when it is in host memory and otherwise raises `TypeError` (convert it
      with `cupy.from_dlpack` or `torch.from_dlpack`). With host labels, `voxel_graph` is read
      through NumPy, so a device graph that NumPy cannot read (a CuPy array, a CUDA tensor) needs
      device labels.
    """
    if output not in ("skeletons", "packed", "packed_device"):
        raise ValueError("brook: output must be 'skeletons', 'packed' or 'packed_device'")
    runtime.require_gpu()
    context = runtime.context()
    params = dict(teasar_params)
    anisotropy = np.array(anisotropy, dtype=np.float32)
    if _on_device(all_labels):
        return _skeletonize_device(context, all_labels, params, anisotropy, object_ids, dust_threshold, fix_branching,
                                   fix_borders, extra_targets_before, extra_targets_after, fill_holes, fix_avocados,
                                   voxel_graph, streaming, output, progress, stream, borrow)
    labels = format_labels(_host_array(all_labels), in_place=in_place)
    original_ids = None if object_ids is None else list(object_ids)
    ids = original_ids
    mutate = bool(in_place) and labels.flags.owndata
    # Mask query conversion and in-place mutation precede the dust shortcut in
    # Kimimaro. A single-ID np.where never mutates caller storage.
    ids_normalized = False
    if ids is not None and len(ids) != 1:
        checked_ids = runtime.engine().normalize_object_ids(labels.dtype, ids)
        if mutate:
            labels = runtime.engine().mask_labels(labels, ids, in_place=True)
            ids = None
        else:
            ids = checked_ids
            ids_normalized = True
    if labels.size <= dust_threshold:
        return _empty(output, anisotropy)
    mode = _use_streaming(streaming, labels)
    if labels.dtype == np.float16 and (mode or voxel_graph is not None):
        # The compiled preamble has no float16 specialization here.
        raise TypeError("No matching signature found")
    if mode and (voxel_graph is not None or fill_holes or fix_avocados):
        if not runtime.engine().labels_nonzero(labels, original_ids):
            return _empty(output, anisotropy)
        raise NotImplementedError("brook streaming mode does not support voxel_graph, fill_holes or fix_avocados; "
                                  "pass streaming=False or process the volume in chunks.")
    if voxel_graph is not None and labels.dtype.kind == "f":
        if not runtime.engine().labels_nonzero(labels, original_ids):
            return _empty(output, anisotropy)
        # The graph preamble's mapping routine accepts integer labels.
        raise TypeError("No matching signature found")
    extra = {}
    if labels.dtype.kind == "f":
        labels, extra["black_border"], collapsed = runtime.engine().normalize_labels(labels, ids, _return_metadata=True)
        if collapsed:
            # Distance-field normalization divides by zero when
            # differing floating labels truncate to one nonzero integer label.
            raise ZeroDivisionError("float division by zero")
        ids = None
    elif ids is not None:
        if ids_normalized:
            pass
        elif isinstance(ids[0], int):
            bounds = np.iinfo(labels.dtype)
            value = ids[0]
            if value < bounds.min or value > bounds.max:
                ids = []
            else:
                ids = [value if value < (1 << 63) else value - (1 << 64)]
        else:
            labels = runtime.engine().mask_labels(labels, ids)
            ids = None
    options = dict(teasar_params=params, anisotropy=anisotropy, object_ids=ids,
                   dust_threshold=dust_threshold, fix_branching=fix_branching, fix_borders=fix_borders,
                   extra_targets_before=[tuple(int(c) for c in p) for p in extra_targets_before],
                   extra_targets_after=[tuple(int(c) for c in p) for p in extra_targets_after],
                   fill_holes=fill_holes, fix_avocados=fix_avocados,
                   voxel_graph=None if voxel_graph is None else _host_graph(voxel_graph), **extra)
    if mode:
        method = getattr(context, "skeletonize_streamed", None)
        if method is None:
            raise NotImplementedError("This compiled build does not include streamed skeleton tracing.")
        result = _objects(_execute(method, labels, dict(options, budget_bytes=_budget_bytes()), progress), anisotropy)
        if output == "skeletons":
            return result
        packed = PackedSkeletons.from_skeletons(result, anisotropy)
        return packed.to_device() if output == "packed_device" and len(packed) else packed
    if output == "skeletons":
        return _objects(_execute(context.skeletonize, labels, options, progress), anisotropy)
    packed = PackedSkeletons.from_core(_execute(context.skeletonize_packed, labels, options, progress))
    return packed if output == "packed_device" and len(packed) else packed.to_host()


def _on_device(labels):
    """Device input: an array other than a NumPy array that exposes the CUDA array interface.
    DLPack alone does not make an array device input (see `_host_array`)."""
    return not isinstance(labels, np.ndarray) and hasattr(labels, "__cuda_array_interface__")


def _host_array(array):
    """Host input as `format_labels` reads it. An array that exposes DLPack but neither the CUDA
    array interface nor a NumPy array protocol is imported with `np.from_dlpack` when it is in
    host memory, and raises on a device, whose input is read through the CUDA array interface.
    Anything else is returned unchanged for NumPy to read, as Kimimaro's intake does."""
    if not hasattr(array, "__dlpack_device__") or hasattr(array, "__cuda_array_interface__") or _numpy_readable(array):
        return array
    device = int(array.__dlpack_device__()[0])
    if device not in _DLPACK_HOST_DEVICES:
        raise TypeError(f"brook: this array exposes only DLPack (device type {device}); device input must expose "
                        "the CUDA array interface. Convert it first, e.g. with cupy.from_dlpack() or torch.from_dlpack().")
    return np.from_dlpack(array)


def _numpy_readable(array):
    """An object with one of NumPy's array protocols, through which Kimimaro's intake reads it."""
    return any(hasattr(array, name) for name in ("__array__", "__array_interface__", "__array_struct__"))


def _host_graph(voxel_graph):
    """The voxel graph of host labels, read as NumPy reads it (a JAX GPU array copies itself to the
    host); a graph with a CUDA array interface and no NumPy array protocol needs device labels."""
    if _on_device(voxel_graph) and not _numpy_readable(voxel_graph):
        raise TypeError("brook: a voxel_graph with a CUDA array interface needs device labels; pass the labels "
                        "on the device too, or the graph as a host array")
    return format_labels(_host_array(voxel_graph))


def _skeletonize_device(context, labels, params, anisotropy, object_ids, dust_threshold, fix_branching, fix_borders,
                        extra_targets_before, extra_targets_after, fill_holes, fix_avocados, voxel_graph, streaming, output, progress,
                        stream=None, borrow=True):
    """Skeletonize a device-resident label volume without a host round trip.

    A Fortran-contiguous array on the context's device is read in place (`borrow=True`; the input
    is never written and stays alive for the call); anything else is copied device-to-device into
    Brook's layout (a C-contiguous input is transposed on the GPU). `stream` is the producer's
    stream handle (default: the array's CUDA-array-interface stream, else the legacy default
    stream); Brook's stream waits for it before reading. A `voxel_graph` (uint32, the labels'
    shape) is read in place under the same rules when it is device-resident, else uploaded.
    Host-only conveniences are not offered on this path: object_ids (mask on the device first),
    streaming and floating-point labels."""
    if object_ids is not None:
        raise NotImplementedError("object_ids is not supported for device-resident input; mask the labels on the device")
    if streaming is not None and streaming:
        raise NotImplementedError("streaming is a host-input feature")
    data = context.upload_cuda_array(labels, stream, borrow)
    if data.dtype.kind == "f":
        raise TypeError("device-resident labels must be integer or boolean")
    if data.size <= dust_threshold:
        return _empty(output, anisotropy)
    graph = None
    if voxel_graph is not None:
        graph = (context.upload_cuda_array(voxel_graph, stream, borrow) if _on_device(voxel_graph)
                 else context.upload(format_labels(np.asfortranarray(_host_array(voxel_graph)))))
    options = dict(teasar_params=params, anisotropy=anisotropy, object_ids=None,
                   dust_threshold=dust_threshold, fix_branching=fix_branching, fix_borders=fix_borders,
                   extra_targets_before=[tuple(int(c) for c in p) for p in extra_targets_before],
                   extra_targets_after=[tuple(int(c) for c in p) for p in extra_targets_after],
                   fill_holes=fill_holes, fix_avocados=fix_avocados, voxel_graph=graph)
    if output == "skeletons":
        return _objects(_execute(context.skeletonize_device, data, options, progress), anisotropy)
    packed = PackedSkeletons.from_core(_execute(context.skeletonize_packed_device, data, options, progress))
    return packed if output == "packed_device" and len(packed) else packed.to_host()


def warmup(streaming=True):
    """Run small skeletonizations so that CUDA initialization and module loading happen before
    timed work; `streaming=True` also warms the streamed path."""
    n = 40
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    labels = np.zeros((n, n, n), dtype=np.uint32, order="F")
    labels[(x-14)**2 + (y-14)**2 + (z-14)**2 <= 100] = 1
    labels[12:17, 12:17, 14:38] = 1
    labels[28:32, 4:36, 28:32] = 2
    labels[4:36, 30:34, 4:8] = 3
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4,
                  soma_detection_threshold=6, soma_acceptance_threshold=8,
                  soma_invalidation_scale=1., soma_invalidation_const=0)
    for branching in (True, False):
        skeletonize(labels, teasar_params=params, dust_threshold=10, fix_branching=branching,
                    progress=False, streaming=False)
    if streaming:
        skeletonize(labels, teasar_params=params, dust_threshold=10, progress=False, streaming=True)
