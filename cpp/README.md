# Brook C/C++ SDK

This directory contains the C++17/CUDA implementation and Python bindings.
The C/C++ library does not require Python.

## Build

CUDA Toolkit 12.3 or newer and CMake 3.24 or newer are required. For a C/C++ build:

```sh
cmake -S . -B build/sdk -DBROOK_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/sdk -j4
ctest --test-dir build/sdk --output-on-failure
cmake --install build/sdk --prefix /your/install/prefix --component SDK
```

Brook supports Linux x86-64 and needs an NVIDIA GPU of compute capability 8.0 or newer
(Ampere or later); it is tested on an NVIDIA RTX 4090 (`sm_89`) and an NVIDIA H100
(`sm_90`). There are no macOS builds, and Windows is untested.

Set `CMAKE_CUDA_ARCHITECTURES` or the `CUDAARCHS` environment variable for the target
GPUs, for example `89`, `90`, or the release list
`"80-real;86-real;89-real;90-real;100-real;120-real;80-virtual;120-virtual"`; without
either, the default is `native`, the build machine's GPU. Entries below 80 stop the
configure step. `-virtual` entries embed PTX, which the driver compiles for a GPU without
matching machine code; that needs a driver at least as new as the build's CUDA toolkit.
The configure output lists the build's code as `Brook CUDA code: SASS [...], PTX [...]`.
Building Python bindings additionally requires pybind11 and Python development headers;
point CMake to that environment with `Python_EXECUTABLE` and `pybind11_DIR`.

Creating a context (`brook_context_create`, `brook::api::Context`) runs the same startup
check as the Python package. For an unsupported GPU, a driver older than the build needs
(CUDA 12.3, or 13.0 for a CUDA 13 build), or a build without GPU code the driver can load,
`brook_context_create` returns `BROOK_CUDA_ERROR` and `brook_last_error()` holds a message
that names the problem and the fix; the C++ wrapper throws `std::runtime_error` with that
message.

Consumers can use `find_package(Brook CONFIG REQUIRED)` and link `Brook::brook`, or use
`pkg-config brook`. C programs include `brook/brook.h`; C++ programs can use the move-only
`brook::api::Context` and `brook::api::Skeletons` wrappers in `brook/brook.hpp`.
See `tests/c_smoke.c` and `tests/cpp_smoke.cpp` for independently linked examples.

The C ABI is provisional until the first stable ABI release. Calls are blocking. Inputs are
borrowed for the call; results own their storage and may outlive the execution context.
Calls through one context are serialized. Public calls restore the caller's CUDA device.

Python device volumes come from `Context.upload`; contiguous tensor data such as
vertices and edges use `Context.upload_tensor`. Device arrays expose `to_host()`/`get()`,
CUDA array interface, `__dlpack__` and `__dlpack_device__`. DLPack supports zero-copy
exports, requested GPU copies and CPU transfers. Exported owners can outlive both
the original Python array and its context. Producers complete before returning.

The C API's DLPack exports transfer one managed-tensor ownership reference to the
consumer, which must invoke its deleter once. Include `brook/dlpack.h` to inspect
the tensor fields. The C++ `Array::dlpack()` wrapper owns that reference until it
is released or moved to the consumer. DLPack's upstream header, license and pinned
source manifest are included in this SDK.

The skeletonization options include `fill_holes` and `fix_avocados` in Python,
and `BROOK_FILL_HOLES` / `BROOK_FIX_AVOCADOS` in the C/C++ flags. Avocado candidate
detection uses the supplied soma detection threshold. Its compiled host logic invokes
the GPU EDT when the detector records a merge. With a voxel graph, both options run on
the graph's components, and that EDT is the graph EDT, as in Kimimaro.

Point-to-point paths are available through `Context.connect_points` in the Python
extension, `brook_connect_points` in C, and `brook::api::Context::connect_points`
in C++. Inputs use nonzero foreground; vertices run from end to start, as in
Kimimaro's `point_to_point`, and are physical coordinates. As in
`kimimaro.connect_points`, the result does not depend on the input's memory layout.

Host-resident EDT is available through Python `Context.edt_streamed`, C
`brook_edt_streamed`, and C++ `Context::edt_streamed`. It runs full-length line
transforms in bounded slabs/tiles and returns a host F-order float32 field.
The caller can set a scheduling budget; input and output must not overlap.

Streamed CCL is exposed as `Context.connected_components_streamed` in Python/C++,
or `brook_connected_components_streamed` in C. Its host result owns F-order
labels, original-label mapping and minimum-voxel roots, with exactly the in-core
component numbering.

`Context.skeletonize_streamed` / `brook_skeletonize_streamed` combine host-resident
CCL/EDT with GPU crop tracing. The budget controls preamble slabs/tiles;
each component workspace still needs to fit on the device. Voxel graphs,
`fill_holes` and `fix_avocados` are not supported in this mode. Component
targets are sorted on the GPU.

Kimimaro-compatible oversegmentation is available through `Context.oversegment` in Python,
`brook_oversegment` in C, and `brook::api::oversegment` / `Context::oversegment`
in C++. It runs on the CPU; `fill_holes` uses the GPU.

## Cross-sectional area helper

`brook.cross_sectional_area` and `brook_cross_sectional_area` run on the CPU and need
no CUDA context. The C++ wrapper is
`brook::api::cross_sectional_area`; `tests/cross_section_smoke.cpp` is an independent
consumer.

This helper uses xs3d 1.13.0 (LGPL-3.0-or-later) through the separately replaceable
`libbrook_xs3d.so.1`. SDK/Python installs include its authors, GPL/LGPL texts,
complete corresponding source and standalone build recipe. See
[replacement and distribution instructions](third_party/xs3d/README.md).

## Distribution

The Python wheel installs the Python modules, the compiled binding, `libbrook`,
the independently replaceable xs3d bridge and (by default) the matching unmodified
CUDA runtime. Relative RUNPATH entries support relocation. The SDK offers
`Brook::brook` through CMake and `brook` through pkg-config, with no CUDA compiler
required for ordinary C/C++ consumers. Rebuilding the CUDA implementation itself
requires the CUDA toolkit and a C++ compiler.

Corresponding sources and notices are included in both distributions. See the
[package build instructions](../docs/USAGE.md#build-from-source), [differences from Kimimaro](../docs/USAGE.md#differences-from-kimimaro)
and [distribution notices](../NOTICE). Brook contributions are GPL-3.0-only;
incorporated components retain their respective license terms.
