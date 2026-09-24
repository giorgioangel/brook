# Contributing

## Reporting a problem

Open an issue with a minimal example and the output of

```sh
python -c "import brook, brook.device as d; print(brook.__version__); print(d.build_info()); print(d.device_properties())"
```

together with the Linux distribution, the Python version and how Brook was installed. Report
security problems as described in [SECURITY.md](SECURITY.md), not in an issue.

## Platform

Brook supports Linux x86-64 with NVIDIA GPUs of compute capability 8.0 or newer. Building it
needs a CUDA toolkit 12.3 or newer. Windows is untested, and there are no macOS wheels.

## Build and test

```sh
python -m pip install . pytest ruff
python -m pytest tests/python
ruff check .
```

The GPU tests need an NVIDIA GPU and are skipped when none is visible. Without
`CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS`, the build targets the GPU of the build machine; on a
machine without one, pass for example `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=89` to pip.

The C/C++ SDK and its tests:

```sh
cmake -S . -B build/sdk -G Ninja -DCMAKE_BUILD_TYPE=Release -DBROOK_BUILD_PYTHON=OFF -DBROOK_BUILD_TESTS=ON
cmake --build build/sdk
ctest --test-dir build/sdk --output-on-failure
```

`ctest -L cpu` runs only the tests that need no GPU.

## Pull requests

CI runs only the CPU tests; run the GPU tests locally. For changes to CUDA code, also run
Compute Sanitizer as described in [docs/USAGE.md](docs/USAGE.md#testing). A pull request that changes skeleton
output says so.

## License

Contributions are licensed under GPL-3.0-only, the license of the project. No DCO sign-off is
required.
