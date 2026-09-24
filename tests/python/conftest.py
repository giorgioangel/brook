"""Tests run against the installed package. Imports of cupy, cupyx, kimimaro, edt, dijkstra3d and
scipy are blocked so that no result can come from them."""
import importlib.abc
import sys

import pytest

_FORBIDDEN = {"cupy", "cupyx", "kimimaro", "edt", "dijkstra3d", "scipy"}


class NoFallback(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _FORBIDDEN:
            raise ImportError(f"{fullname} is blocked in Brook's tests")
        return None


assert not (_FORBIDDEN & {name.split(".")[0] for name in sys.modules})
sys.meta_path.insert(0, NoFallback())


def pytest_configure(config):
    # The tests also run outside the repository root (the conda recipe, brook/_source).
    config.addinivalue_line("markers", "gpu: needs an NVIDIA GPU and the CUDA extension")


@pytest.fixture
def gpu():
    from brook.device import device_count, require_gpu

    # Skip only without a visible device; an unsupported driver, GPU or build fails the test with
    # Brook's startup-check message instead of skipping every GPU test with exit code 0.
    if not device_count():
        pytest.skip("no visible CUDA device")
    require_gpu()
