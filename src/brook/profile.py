# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""
Opt-in profiling for Brook (zero overhead when disabled).

  BROOK_PROFILE=1      host wall-clock timers per stage / per path + NVTX ranges
  BROOK_PROFILE=sync   additionally synchronizes the device at the end of every
                       range whose block finished, so a range's time is its own GPU
                       time (exact attribution, at the cost of removing host/device
                       overlap); a CUDA error surfaces there and never replaces an
                       exception the block raised

When the optional nvtx extra is installed, manually instrumented ranges also
appear in Nsight Systems. Ranges can be grouped by component size. Call
prof.emit() to print a report and store it in brook.profile.last_report.
The compiled engine keeps its own execution statistics on the context
(`brook.device.context().stats`, for the last skeletonize, batch or streamed call);
this module times Python-side ranges only. Among the engine's keys:

  graphs               1 when CUDA graphs are enabled, 0 with BROOK_GRAPHS=0 (or off,
                       false, no)
  grid_*               the largest grid each cooperative kernel family used; grid_*
                       are diagnostic
  chunks_seq, chunks_pipe
                       lockstep chunks in which the main lane ran the sequential or
                       the pipelined body; chunks_k: those in which the draft lane ran
  iterations_seq, iterations_pipe, iterations_k
                       lockstep iterations of the two bodies, and draft-lane batches
                       run while the main lane had no labels; for one trace they add
                       up to iterations
  pipelined            1 when the pipelined body's buffers were allocated, whether
                       or not it ran; chunks_pipe and grid_pipe_* show that it ran

Scheduler counts depend on timing; the skeletons do not.

Usage:
    from brook.profile import prof
    with prof.range("edt"):
        ...
    with prof.component(bbox_voxels):          # sets the size class for nested ranges
        with prof.range("trace.paths.flood"):
            ...
    prof.count("paths", 1)
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from collections import defaultdict

MODE = os.environ.get("BROOK_PROFILE", "").strip().lower()
ENABLED = MODE not in ("", "0", "off", "false", "no")
SYNC = MODE == "sync"

# NVTX is optional instrumentation; profiling works without it.
try:
    import nvtx
except ImportError:
    _nvtx = None
else:
    class _Nvtx:
        RangePush = staticmethod(nvtx.push_range)
        RangePop = staticmethod(nvtx.pop_range)
    _nvtx = _Nvtx()

_NULL = contextlib.nullcontext()
last_report: str = ""

SIZE_CLASSES = (("S", 32 ** 3), ("M", 128 ** 3), ("L", 256 ** 3), ("XL", float("inf")))


def size_class(n_voxels: int) -> str:
    for name, limit in SIZE_CLASSES:
        if n_voxels <= limit:
            return name
    return "XL"  # pragma: no cover


class _Stat:
    __slots__ = ("total", "count", "max")

    def __init__(self):
        self.total = 0.0
        self.count = 0
        self.max = 0.0

    def add(self, dt: float):
        self.total += dt
        self.count += 1
        if dt > self.max:
            self.max = dt


class Profiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stats: dict[str, _Stat] = defaultdict(_Stat)
        self.counters: dict[str, int] = defaultdict(int)
        self.t0 = time.perf_counter()
        self._tls = threading.local()      # per-thread size class (components run concurrently)
        self._lock = threading.Lock()
        self.extra: list[str] = []

    # -- ranges --------------------------------------------------------------
    def range(self, name: str):
        if not ENABLED:
            return _NULL
        return self._range(name)

    @contextlib.contextmanager
    def _range(self, name: str):
        if _nvtx is not None:
            _nvtx.RangePush(name)
        t = time.perf_counter()
        try:
            yield
            # Only after a block that finished: a CUDA error surfaces here, and it never replaces
            # an exception the block raised.
            if SYNC:
                _sync()
        finally:
            dt = time.perf_counter() - t
            if _nvtx is not None:
                _nvtx.RangePop()
            cls = getattr(self._tls, "cls", None)
            with self._lock:
                self.stats[name].add(dt)
                if cls is not None:
                    self.stats[f"{name}@{cls}"].add(dt)

    def component(self, n_voxels: int):
        """Nested ranges are additionally bucketed by this component's size class."""
        if not ENABLED:
            return _NULL
        return self._component(n_voxels)

    @contextlib.contextmanager
    def _component(self, n_voxels: int):
        prev = getattr(self._tls, "cls", None)
        self._tls.cls = size_class(int(n_voxels))
        with self._lock:
            self.counters[f"components@{self._tls.cls}"] += 1
        try:
            yield
        finally:
            self._tls.cls = prev

    def count(self, name: str, k: int = 1):
        if ENABLED:
            cls = getattr(self._tls, "cls", None)
            with self._lock:
                self.counters[name] += k
                if cls is not None:
                    self.counters[f"{name}@{cls}"] += k

    def note(self, line: str):
        if ENABLED:
            self.extra.append(line)

    # -- report --------------------------------------------------------------
    def report(self, title: str = "brook profile") -> str:
        wall = time.perf_counter() - self.t0
        lines = [f"== {title} ({'sync' if SYNC else 'async'} timers) wall {wall:.2f}s =="]
        top = {k: v for k, v in self.stats.items() if "@" not in k}
        per_cls = {k: v for k, v in self.stats.items() if "@" in k}
        lines.append(f"{'range':<34}{'total s':>10}{'% wall':>8}{'count':>9}"
                     f"{'mean ms':>10}{'max ms':>10}")
        for name, st in sorted(top.items(), key=lambda kv: -kv[1].total):
            lines.append(f"{name:<34}{st.total:>10.2f}{100 * st.total / wall:>8.1f}"
                         f"{st.count:>9d}{1e3 * st.total / max(st.count, 1):>10.3f}"
                         f"{1e3 * st.max:>10.2f}")
        if per_cls:
            lines.append("")
            lines.append(f"{'range@size-class':<34}{'total s':>10}{'% wall':>8}{'count':>9}"
                         f"{'mean ms':>10}{'max ms':>10}")
            order = {c: i for i, (c, _) in enumerate(SIZE_CLASSES)}
            for name, st in sorted(per_cls.items(),
                                   key=lambda kv: (kv[0].split("@")[0],
                                                   order.get(kv[0].split("@")[1], 9))):
                lines.append(f"{name:<34}{st.total:>10.2f}{100 * st.total / wall:>8.1f}"
                             f"{st.count:>9d}{1e3 * st.total / max(st.count, 1):>10.3f}"
                             f"{1e3 * st.max:>10.2f}")
        if self.counters:
            lines.append("")
            lines.append("counters: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.counters.items())))
        lines.extend(self.extra)
        return "\n".join(lines)

    def emit(self, title: str = "brook profile"):
        """Print the report to stderr and keep it in `brook.profile.last_report`."""
        global last_report
        if not ENABLED:
            return
        last_report = self.report(title)
        print(last_report, file=sys.stderr, flush=True)


def _sync():
    try:
        from ._runtime import synchronize
    except ImportError:  # pragma: no cover - no compiled extension: nothing to synchronize
        return
    synchronize()  # a CUDA error surfaces here instead of being swallowed


prof = Profiler()
