# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""The public Python API; runs without a GPU."""
import pytest

import brook
import brook.device


def test_all_names_resolve():
    for name in brook.__all__:
        assert getattr(brook, name) is not None, name
    namespace = {}
    exec("from brook import *", namespace)
    assert set(brook.__all__) <= set(namespace)


def test_dir_lists_all():
    assert set(brook.__all__) <= set(dir(brook))


def test_device_names_resolve():
    for name in brook.device.__all__:
        assert getattr(brook.device, name) is not None, name


def test_cuda_error_export():
    from brook import _core

    assert brook.device.CudaError is _core.CudaError
    assert issubclass(brook.device.CudaError, RuntimeError)
    assert "CudaError" in brook.device.__all__


@pytest.mark.parametrize("value", ["0", "-1", "abc", "", " 256", "256 ", "+256", "1.5", "0x10"])
def test_stream_budget_rejects(value):
    from brook._intake import _parse_stream_budget_mb

    with pytest.raises(ValueError, match="invalid BROOK_STREAM_BUDGET_MB="):
        _parse_stream_budget_mb(value)


def test_stream_budget_accepts(monkeypatch):
    from brook._intake import _budget_bytes, _parse_stream_budget_mb

    assert _parse_stream_budget_mb("256") == 256 << 20
    assert _parse_stream_budget_mb(None) == 0
    monkeypatch.delenv("BROOK_STREAM_BUDGET_MB", raising=False)
    assert _budget_bytes() == 0
    monkeypatch.setenv("BROOK_STREAM_BUDGET_MB", "64")
    assert _budget_bytes() == 64 << 20
    monkeypatch.setenv("BROOK_STREAM_BUDGET_MB", "")
    with pytest.raises(ValueError):
        _budget_bytes()
