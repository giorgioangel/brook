# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Device discovery and per-thread contexts, loaded lazily from the compiled extension."""
from __future__ import annotations

import os
import threading

_CONTEXTS = threading.local()


def engine():
    from . import _core

    return _core


def device_count():
    """Number of CUDA devices visible to this process."""
    return engine().device_count()


def current_device():
    """Ordinal of the calling thread's current CUDA device."""
    return engine().current_device()


def set_device(device):
    """Make `device` the calling thread's current CUDA device."""
    engine().set_device(int(device))


def memory_info(device=None):
    """`(free, total)` memory of a device (the current one by default), in bytes."""
    return engine().memory_info(-1 if device is None else int(device))


def device_properties(device=None):
    """Name, compute capability, memory and multiprocessor count of a device (the current one by
    default), as a dict."""
    return engine().device_properties(-1 if device is None else int(device))


def context(device=None):
    """The calling thread's Brook context on a device (the current one by default); there is one
    per process, Python thread and device."""
    pid = os.getpid()
    previous = getattr(_CONTEXTS, "pid", pid)
    if previous != pid and getattr(_CONTEXTS, "contexts", None):
        # Do not destroy or reuse CUDA handles inherited by fork. Worker pools
        # use spawn; retaining the inherited references avoids unsafe teardown.
        raise RuntimeError("Brook CUDA contexts cannot be reused after fork; start GPU workers with spawn.")
    _CONTEXTS.pid = pid
    if not hasattr(_CONTEXTS, "contexts"):
        _CONTEXTS.contexts = {}
    ordinal = current_device() if device is None else int(device)
    if ordinal not in _CONTEXTS.contexts:
        _CONTEXTS.contexts[ordinal] = engine().Context(ordinal)
    return _CONTEXTS.contexts[ordinal]


def build_info():
    """GPU code in this build (SASS and PTX architectures) and the CUDA toolkit, runtime and driver versions."""
    return engine().build_info()


def require_gpu(device=None):
    """Brook's startup check for a device (the current one by default): an NVIDIA GPU of compute
    capability 8.0 or newer, a driver new enough for this build, and GPU code the driver can load.
    Raises brook.device.CudaError (a RuntimeError) with the reason and the fix."""
    engine().check_device(-1 if device is None else int(device))


def synchronize():
    """Wait for the calling thread's context on the current device to finish its work."""
    context().synchronize()
