# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Worker pools on one or more NVIDIA GPUs, and chunked skeletonization of large volumes."""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler

import numpy as np


def visible_devices():
    """The entries of CUDA_VISIBLE_DEVICES up to the first -1, or every device index when it is
    unset."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        devices = []
        for token in configured.split(","):
            token = token.strip()
            if token == "-1":
                break
            if token:
                devices.append(token)
        return devices
    from ._runtime import device_count

    return [str(i) for i in range(device_count())]


def _worker(device, tasks, results, cancelled, warm):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    def send(message):
        # A Connection serializes synchronously: pickle errors are caught by the
        # worker, rather than disappearing in a Queue feeder thread.
        results.send(message)

    try:
        from . import _intake as intake
        from ._runtime import memory_info, require_gpu

        # The startup check runs whether or not the worker warms up: an unsupported GPU, driver or
        # build fails the pool at start with Brook's message, not the first task.
        require_gpu()
        if warm:
            intake.warmup()
        send(("ready", device, int(memory_info()[0])))
    except BaseException:
        send(("fatal", device, traceback.format_exc()))
        results.close()
        return
    try:
        while True:
            task = tasks.get()
            if task is None:
                return
            generation, tid, key, name, shape, dtype, order, kwargs = task
            if generation <= cancelled.value:
                send(("cancelled", tid, key))
                continue
            try:
                if name is None:
                    from ._packed import merge_fragments

                    post = kwargs.pop("postprocess", None)
                    fused = merge_fragments(on_device=post is not None, **kwargs)
                    if post is not None:
                        from ._device_post import postprocess

                        fused = postprocess(fused, **post)
                    result = fused.to_host()
                else:
                    block = shared_memory.SharedMemory(name=name)
                    try:
                        labels = np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf, order=order)
                        result = intake.skeletonize(labels, **kwargs)
                        # IPC results cannot retain worker device pointers.
                        if hasattr(result, "to_host"):
                            result = result.to_host()
                        del labels
                    finally:
                        block.close()
                send(("done", tid, (key, result)))
            except BaseException:
                send(("error", tid, (key, traceback.format_exc())))
    finally:
        results.close()


class GpuPool:
    """One spawned worker per device entry; repeated entries are supported.

    Each entry becomes its worker's CUDA_VISIBLE_DEVICES, so it is an index among all of the
    machine's GPUs or a UUID, not an ordinal within the parent's visible devices; the default is
    the parent's visible devices. Each worker runs Brook's startup check when it starts, with or
    without warmup, and a failure raises RuntimeError from the constructor.

    A cancelled/failed iterator drains already-running jobs before releasing
    their shared buffers. The pool remains reusable after ordinary task errors.
    Only one map/merge operation may be active on a pool at a time. Workers are spawned, so
    create the pool under `if __name__ == "__main__":`.
    """

    def __init__(self, devices=None, warmup=True, in_flight=None):
        self.devices = [str(d) for d in (devices if devices is not None else visible_devices())]
        if not self.devices:
            raise RuntimeError("brook.GpuPool: no GPU visible")
        self.in_flight = 2 * len(self.devices) if not in_flight else int(in_flight)
        if self.in_flight < 1:
            raise ValueError("in_flight must be positive")
        self._closed = False
        self._operation = threading.Lock()
        self._blocks = {}
        self._next_tid = 0
        self._generation = 0
        self._procs = []
        self.free_bytes = []
        ctx = mp.get_context("spawn")
        self._tasks = ctx.Queue()
        self._readers = []
        self._cancelled = ctx.RawValue("q", 0)
        try:
            for device in self.devices:
                reader, writer = ctx.Pipe(duplex=False)
                self._readers.append(reader)
                process = ctx.Process(target=_worker,
                                      args=(device, self._tasks, writer, self._cancelled, warmup), daemon=True)
                process.start()
                self._procs.append(process)
                writer.close()
            while len(self.free_bytes) < len(self.devices):
                kind, device, payload = self._get()
                if kind == "fatal":
                    raise RuntimeError(f"brook.GpuPool: worker on GPU {device} failed to start:\n{payload}")
                if kind != "ready":
                    raise RuntimeError("brook.GpuPool: invalid worker startup response")
                self.free_bytes.append(int(payload))
        except BaseException:
            if "writer" in locals():
                writer.close()
            self.close(_terminate=True)
            raise

    def _get(self):
        while True:
            if self._closed:
                raise RuntimeError("brook.GpuPool: pool is closed")
            ready = wait(self._readers, timeout=1.0)
            for reader in ready:
                try:
                    return reader.recv()
                except (EOFError, OSError):
                    raise RuntimeError("brook.GpuPool: a worker result channel closed unexpectedly") from None
            dead = [p for p in self._procs if p.exitcode is not None]
            if dead:
                raise RuntimeError(
                    f"brook.GpuPool: a worker died (exit code {dead[0].exitcode}); spawned workers "
                    "require pool creation under `if __name__ == '__main__':` in scripts."
                )
            if self._closed:
                raise RuntimeError("brook.GpuPool: pool is closed")

    def _begin(self):
        if self._closed:
            raise RuntimeError("brook.GpuPool: pool is closed")
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("brook.GpuPool: map and merge operations cannot overlap")
        self._generation += 1
        return self._generation

    def _put(self, task):
        ForkingPickler.dumps(task)  # Validate before Queue's asynchronous feeder.
        self._tasks.put(task)

    def _free(self, tid):
        block = self._blocks.pop(tid, None)
        if block is not None:
            block.close()
            try:
                block.unlink()
            except FileNotFoundError:
                pass

    def _drain(self, pending, generation):
        self._cancelled.value = generation
        while pending and not self._closed:
            kind, tid, _ = self._get()
            if kind not in ("done", "error", "cancelled") or tid not in pending:
                raise RuntimeError("brook.GpuPool: unexpected result while draining tasks")
            pending.remove(tid)
            self._free(tid)

    def imap(self, tasks, ordered=True, **skeletonize_kwargs):
        """Skeletonize `(key, labels)` pairs on the pool's GPUs and yield `(key, result)`, in input
        order when `ordered`. Keywords go to `skeletonize`."""
        generation = self._begin()
        kwargs = dict(skeletonize_kwargs)
        kwargs.setdefault("progress", False)
        kwargs["in_place"] = True
        pending, completed = set(), {}
        next_out = self._next_tid
        iterator, exhausted = None, False
        try:
            iterator = iter(tasks)
            while not exhausted or pending or completed:
                while not exhausted and len(pending) + len(completed) < self.in_flight:
                    try:
                        key, labels = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    labels = np.asarray(labels)
                    if labels.dtype.hasobject:
                        raise TypeError("shared-memory label arrays cannot contain Python objects")
                    # Keep C/F contiguous layouts; for strided views choose their
                    # fastest endpoint axis and let CUDA handle the device reorder.
                    order = "C" if labels.ndim > 1 and abs(labels.strides[0]) > abs(labels.strides[-1]) else "F"
                    block = shared_memory.SharedMemory(create=True, size=max(labels.nbytes, 1))
                    tid = self._next_tid
                    self._next_tid += 1
                    self._blocks[tid] = block
                    try:
                        view = np.ndarray(labels.shape, dtype=labels.dtype, buffer=block.buf, order=order)
                        view[...] = labels
                        del view
                        self._put((generation, tid, key, block.name, labels.shape,
                                   labels.dtype.str, order, kwargs))
                    except BaseException:
                        self._free(tid)
                        raise
                    pending.add(tid)
                if not pending:
                    break
                kind, tid, payload = self._get()
                if tid not in pending or kind not in ("done", "error", "cancelled"):
                    raise RuntimeError("brook.GpuPool: unexpected task response")
                pending.remove(tid)
                self._free(tid)
                if kind == "error":
                    raise RuntimeError(f"brook.GpuPool: task {payload[0]!r} failed in a worker:\n{payload[1]}")
                if kind != "done":
                    raise RuntimeError("brook.GpuPool: task was unexpectedly cancelled")
                if kwargs.get("output") == "packed_device" and hasattr(payload[1], "to_device") and len(payload[1]):
                    # An explicitly requested device result is rebuilt on the
                    # parent's device after IPC.
                    payload = (payload[0], payload[1].to_device())
                if not ordered:
                    yield payload
                else:
                    completed[tid] = payload
                    while next_out in completed:
                        yield completed.pop(next_out)
                        next_out += 1
        except BaseException:
            try:
                self._drain(pending, generation)
            except BaseException:
                self.close(_terminate=True)
            raise
        finally:
            self._operation.release()

    def map(self, tasks, **kwargs):
        """`imap` in input order, as a list."""
        return list(self.imap(tasks, ordered=True, **kwargs))

    def merge(self, parts, origins, postprocess=None):
        """Merge packed chunk results placed at `origins` (physical units) in one worker, optionally
        postprocessing on the GPU (`postprocess`: a dict of `dust_threshold`, `tick_threshold`).
        Returns host `PackedSkeletons`."""
        generation = self._begin()
        tid = self._next_tid
        self._next_tid += 1
        queued = False
        try:
            parts = [part.to_host() if hasattr(part, "to_host") else part for part in parts]
            self._put((generation, tid, "merge", None, None, None, None,
                       dict(parts=parts, origins=origins, postprocess=postprocess)))
            queued = True
            kind, result_tid, payload = self._get()
            if result_tid != tid or kind not in ("done", "error"):
                raise RuntimeError("brook.GpuPool: unexpected merge response")
            queued = False
            if kind == "error":
                raise RuntimeError(f"brook.GpuPool: merging fragments failed:\n{payload[1]}")
            return payload[1]
        except BaseException:
            if queued:
                try:
                    self._drain({tid}, generation)
                except BaseException:
                    self.close(_terminate=True)
            raise
        finally:
            self._operation.release()

    def close(self, *, _terminate=False):
        """Stop the workers and free the shared buffers; the pool is also a context manager."""
        if self._closed:
            return
        self._closed = True
        self._cancelled.value = (1 << 63) - 1
        for process in self._procs:
            if process.is_alive():
                if _terminate:
                    process.terminate()
                else:
                    self._tasks.put(None)
        for process in self._procs:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        self._procs = []
        for tid in list(self._blocks):
            self._free(tid)
        self._tasks.close()
        self._tasks.cancel_join_thread()
        for reader in self._readers:
            reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, "_closed") and not self._closed:
                self.close(_terminate=True)
        except Exception:
            pass


def iter_chunks(shape, chunk_shape, overlap=1):
    """Slices that tile a 3D `shape` with chunks of `chunk_shape` that overlap by `overlap` voxels."""
    shape, chunk_shape = tuple(map(int, shape)), tuple(map(int, chunk_shape))
    if len(shape) != 3 or len(chunk_shape) != 3:
        raise ValueError("shape and chunk_shape must have three entries")
    starts = [range(0, max(s - overlap, 1), max(c - overlap, 1)) for s, c in zip(shape, chunk_shape)]
    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                yield tuple(slice(a, min(a + c, s)) for a, c, s in zip((x, y, z), chunk_shape, shape))


def chunk_shape_for(shape, itemsize, free_bytes, fix_branching=True):
    """The largest chunk (at most `shape`) whose in-core skeletonization fits in `free_bytes` of GPU
    memory, for labels of `itemsize` bytes."""
    from ._runtime import engine

    return tuple(engine().chunk_shape_for(shape, itemsize, free_bytes, fix_branching))


def skeletonize_chunked(all_labels, chunk_shape=None, devices=None, pool=None,
                        anisotropy=(1, 1, 1), postprocess=None, postprocess_engine="device",
                        output="skeletons", **skeletonize_kwargs):
    """Skeletonize a host volume in overlapping chunks on one or more GPUs and merge the pieces.

    `chunk_shape` defaults to the largest chunk that fits the GPU with the least free memory;
    `devices` as in `GpuPool`, or pass a running `pool`. `postprocess` (dict of `dust_threshold`,
    `tick_threshold`) runs on the GPU (`postprocess_engine="device"`) or with Kimimaro's rules on
    the CPU (`"kimimaro"`). Returns `{label: osteoid.Skeleton}`, or `PackedSkeletons` with
    `output="packed"`. `fix_borders` is always on; other keywords go to `skeletonize`. Workers are
    spawned, so call this under `if __name__ == "__main__":`.
    """
    from ._runtime import engine

    if postprocess_engine not in ("device", "kimimaro"):
        raise ValueError("brook: postprocess_engine must be 'device' or 'kimimaro'")
    skeletonize_kwargs["fix_borders"] = True
    own = pool is None
    pool = pool or GpuPool(devices)
    parts = {}
    try:
        if chunk_shape is None:
            chunk_shape = chunk_shape_for(all_labels.shape, np.dtype(all_labels.dtype).itemsize,
                                          min(pool.free_bytes), skeletonize_kwargs.get("fix_branching", True))
        slices = list(iter_chunks(all_labels.shape, chunk_shape))
        tasks = ((i, all_labels[sl]) for i, sl in enumerate(slices))
        for i, packed in pool.imap(tasks, anisotropy=tuple(anisotropy), output="packed", **skeletonize_kwargs):
            parts[i] = packed
        order = sorted(parts)
        starts = [[axis.start for axis in slices[i]] for i in order]
        origins = engine().physical_origins(starts, anisotropy)
        on_device = postprocess is not None and postprocess_engine == "device"
        merged = pool.merge([parts[i] for i in order], origins,
                            postprocess=dict(postprocess) if on_device else None)
    finally:
        if own:
            pool.close()
    if (postprocess is None or on_device) and output != "skeletons":
        return merged
    out = merged.to_skeletons()
    if postprocess is not None and not on_device:
        from ._helpers import postprocess_many

        out = postprocess_many(out, **dict(postprocess))
    if output != "skeletons":
        from ._packed import PackedSkeletons

        return PackedSkeletons.from_skeletons(out, anisotropy)
    return out
