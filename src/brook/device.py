# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""NVIDIA GPU discovery, device selection and explicit execution contexts."""
from typing import TYPE_CHECKING

from ._runtime import (
    build_info,
    context,
    current_device,
    device_count,
    device_properties,
    engine,
    memory_info,
    require_gpu,
    set_device,
    synchronize,
)

if TYPE_CHECKING:
    from ._core import Context, CudaError, DeviceArray


def __getattr__(name):
    # Loaded from the compiled extension on first access; CudaError is brook._core.CudaError.
    if name in {"Context", "CudaError", "DeviceArray"}:
        return getattr(engine(), name)
    raise AttributeError(name)


__all__ = ["build_info", "context", "current_device", "device_count", "device_properties", "memory_info",
           "require_gpu", "set_device", "synchronize", "Context", "CudaError", "DeviceArray"]
