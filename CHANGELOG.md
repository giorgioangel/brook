# Changelog

All notable changes to Brook are recorded here. The project follows
[semantic versioning](https://semver.org/).

## 0.1.0 - 2026-09-24

First release.

- `brook.skeletonize`: TEASAR skeletonization of 3D label volumes on NVIDIA GPUs, taking the
  arguments of `kimimaro.skeletonize` and returning `osteoid.Skeleton` objects. Output
  differences from Kimimaro 5.8.1 are listed in docs/USAGE.md.
- `brook.skeletonize_batch`: many volumes in one call; each result equals a separate
  `skeletonize` call.
- GPU input through the CUDA array interface (CuPy, PyTorch); packed results that stay on the
  GPU, exported through DLPack or the CUDA array interface.
- `brook.contrib.torch`: PyTorch tensors in and out (no gradient).
- Large volumes: streaming (each object's tracing workspace must fit on the GPU) and
  `skeletonize_chunked` on one or more GPUs.
- GPU postprocessing of packed skeletons (dust, loops, joins, ticks); Kimimaro-compatible
  `postprocess`, `join_close_components`, `connect_points`, `oversegment`,
  `cross_sectional_area` and `synapses_to_targets`.
- `brook.contrib.igneous` (experimental): runs Igneous `SkeletonTask`s on Brook.
- C and C++ SDK (CMake package, pkg-config) with the same engine.
- Requirements: Linux x86-64, an NVIDIA GPU of compute capability 8.0 or newer, and a driver
  that supports CUDA 12.3 or newer (13.0 for a CUDA 13 build). No macOS wheels; Windows is
  untested. A startup check names the problem and the fix for an unsupported GPU, driver or
  build (`brook.device.require_gpu()`, `brook.device.build_info()`).
