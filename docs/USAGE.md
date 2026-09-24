# Brook user guide

The [README](../README.md) covers installation, a first call and the requirements. This guide
covers the TEASAR options, GPU input and output, batches, PyTorch, large volumes, Igneous,
source builds, the startup check, the C and C++ SDK, the stable API, environment variables,
the test suites, and where the output differs from Kimimaro.

- [TEASAR options](#teasar-options)
- [Differences from Kimimaro](#differences-from-kimimaro)
- [GPU input and output](#gpu-input-and-output)
- [Batches](#batches)
- [PyTorch](#pytorch)
- [Large volumes](#large-volumes)
- [Igneous](#igneous)
- [Build from source](#build-from-source)
- [Startup check](#startup-check)
- [C and C++](#c-and-c)
- [API stability](#api-stability)
- [Environment variables](#environment-variables)
- [Testing](#testing)

## TEASAR options

`brook.skeletonize` takes the arguments of `kimimaro.skeletonize`. A call with explicit TEASAR
parameters and postprocessing:

```python
import numpy as np
import brook

labels = np.load("labels.npy")  # 3D integer labels; 0 is background
skeletons = brook.skeletonize(
    labels,
    anisotropy=(16, 16, 40),
    teasar_params={
        "scale": 1.5, "const": 300,
        "pdrf_scale": 100000, "pdrf_exponent": 4,
        "soma_acceptance_threshold": 3500, "soma_detection_threshold": 750,
        "soma_invalidation_const": 300, "soma_invalidation_scale": 2,
    },
    dust_threshold=1000,
    fix_branching=True,
    fix_borders=True,
    progress=False,
)

for label, skeleton in skeletons.items():
    vertices, edges = skeleton.vertices, skeleton.edges
```

As in Kimimaro, keys left out of `teasar_params` take the defaults of `kimimaro.trace`
(for example `pdrf_exponent=16`), not those of `brook.DEFAULT_TEASAR_PARAMS`.
[examples/quickstart.py](../examples/quickstart.py)
runs without a data file.

Coordinates and radii are in the physical units given by `anisotropy`. C- and
Fortran-ordered arrays are both accepted. `parallel` is accepted for compatibility
and ignored; use `brook.GpuPool` for multiple GPUs. `brook.connect_points` takes
`kimimaro.connect_points`'s arguments; as in Kimimaro 5.8.1, `fill_holes` and
`in_place` are accepted and ignored. `brook.warmup()` initializes CUDA ahead of
latency-sensitive work.

## Differences from Kimimaro

Brook returned the same objects (label IDs) as Kimimaro 5.8.1 on every tested volume. The
geometry is close but not identical:

- **Equal-cost routes.** Where routes have the same cost in float32, Brook takes straight
  axial steps, while Kimimaro's heap order often alternates between slices; Brook's total
  cable length was 0.5% to 7.9% shorter on the twelve benchmark datasets. Brook also
  evaluates the PDRF term `dbf_max ** 1.01` in double precision.
- **Somas.** Kimimaro's trimming distance uses uint32 subtraction and keeps vertices inside
  the soma; Brook trims them and attaches each branch to the soma root, so a skeleton with a
  soma is one connected tree.
- **Hole filling.** Brook seeds the exterior flood from every background voxel on a
  component's bounding-box faces; fill_voids, which Kimimaro uses, can also fill some
  isolated voxels on those faces.
- **Avocados.** With `fix_avocados`, Kimimaro can drop an extra target or raise `IndexError`
  after its avocado stage renumbers the components; Brook assigns each target to the
  component that contains it.
- **Voxel graphs.** Kimimaro's invalidation can read other voxels' graph bits when its crop
  does not span the volume's full x and y extent; Brook reads each voxel's own bits.
- **Unreachable targets.** With branching correction, Brook stops tracing a component at the
  first target that no path reaches; Kimimaro retries it until `max_paths`.
- **connect_points.** With NumPy 2.4, Kimimaro 5.8.1 raises a casting error at its last step;
  Brook returns the scaled path, in physical coordinates.
- **postprocess and join_close_components.** These match Kimimaro when pykdtree is installed
  (Kimimaro's `accel` extra). With SciPy's KD-tree, Kimimaro can join a different pair of
  vertices when two pairs are the same distance apart.

## GPU input and output

`skeletonize` also accepts arrays that expose the CUDA array interface (CuPy,
PyTorch CUDA tensors). A Fortran-contiguous array is read in place; other layouts
are copied on the device. Pass the producer's stream as `stream=` (for example
`torch.cuda.current_stream().cuda_stream`) and Brook waits for it before reading.
An array that exposes only DLPack is read on the host when it is in host memory and
otherwise raises `TypeError` (convert it with `cupy.from_dlpack` or `torch.from_dlpack`).
`object_ids`, streaming and floating-point labels need host input. With host labels,
`voxel_graph` must be an array NumPy can read; pass a device graph with device labels.

```python
packed = brook.skeletonize(labels, output="packed_device", progress=False)

from brook.device_post import postprocess
packed = postprocess(packed, dust_threshold=1000, tick_threshold=3500)

vertices = packed.vertices.get()   # copy to NumPy when needed
skeletons = packed.to_skeletons()  # or build osteoid skeletons
```

`output="packed"` returns packed host arrays. Device result arrays support the CUDA
array interface and DLPack, so CuPy or PyTorch can use them without a copy. GPU
postprocessing covers dust and tick removal, loop removal and joining nearby
components.

## Batches

```python
result = brook.skeletonize_batch(samples, anisotropy=(4, 4, 4), dust_threshold=400)
skeletons = result[3].to_skeletons()  # sample 3, as {label: osteoid.Skeleton}
vertices = result.vertices            # all samples, packed on the GPU
```

`samples` is a list of volumes (host or device, shapes may differ) or a dense array
with the samples along its first axis. Each sample's result is the same as calling
`skeletonize` on it alone. Samples of one shape can share preprocessing, and the batch
traces its samples together, which is usually faster than separate calls but uses more
GPU memory. `extra_targets_before`, `extra_targets_after` and `voxel_graph` take one
entry per sample; anisotropy and points follow each sample's axis order, as in
`skeletonize`.

## PyTorch

`brook.contrib.torch` wraps both calls for CUDA tensors (PyTorch is not a
dependency). Results are torch tensors on the same device, and the stream defaults
to the current CUDA stream. Skeletonization has no gradient.

```python
from brook.contrib import torch as bt

skeletonize = bt.Skeletonize(anisotropy=(4, 4, 4), dust_threshold=400)
result = skeletonize(label_batch)  # 4-D CUDA tensor, samples along the first axis
dense = bt.padded(result)          # vertices, radii, labels, mask as dense tensors
```

## Large volumes

When a volume does not fit in GPU memory, Brook streams connected components and
distance transforms through the GPU in slabs; `streaming=True` or `False` selects
the mode explicitly. Each object's tracing workspace must still fit on the GPU, and
streaming does not support `voxel_graph`, `fill_holes` or `fix_avocados`.

Alternatively, process overlapping chunks, optionally across several GPUs:

```python
if __name__ == "__main__":
    skeletons = brook.skeletonize_chunked(
        labels,
        chunk_shape=(512, 512, 512),
        anisotropy=(16, 16, 40),
        postprocess={"dust_threshold": 1000, "tick_threshold": 3500},
    )
```

Chunking and streaming handle object boundaries differently. Scaling of
`skeletonize_chunked` across GPUs has not been measured.

## Igneous

With the `igneous` extra, existing Igneous skeleton tasks can run on Brook. The integration
is experimental and may change in later releases.

```python
from brook.contrib import igneous

with igneous.engine(postprocess="device"):
    task.execute()  # an existing Igneous SkeletonTask
```

## Build from source

You need Python 3.12 or newer, CMake 3.24+, a C++17 compiler and the NVIDIA CUDA
Toolkit 12.3 or newer.
`CMAKE_CUDA_ARCHITECTURES` or the `CUDAARCHS` environment variable select the target
GPUs (`89` is an NVIDIA RTX 4090, `90` an NVIDIA H100); without either, CMake targets the
build machine's GPU.

```sh
git clone https://github.com/giorgioangel/brook.git
cd brook
python -m pip install build
python -m build --wheel -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=89
python -m pip install dist/brook-*.whl
```

A source build installs the distribution `brook`, which is also the name of the
unrelated PyPI project: `pip install -U brook` would replace it with that project.

`CMAKE_CUDA_ARCHITECTURES` takes a list, such as
`"80-real;86-real;89-real;90-real;100-real;120-real;80-virtual;120-virtual"` (the
release wheels). A `-real` entry adds machine code for that architecture, a `-virtual`
entry adds PTX, and a plain number adds both. A GPU without matching machine code
compiles the PTX on its first use of Brook (the driver caches the result), which needs
a driver at least as new as the build's CUDA toolkit. Architectures below 80 are
rejected when CMake configures the build. Machine code for a GPU needs a toolkit that
supports it (CUDA 12.3 to 12.6 stop at compute capability 9.0; 10.0 and 12.0 need
CUDA 12.8 or newer). A build made with CUDA 13 needs a driver that supports CUDA 13.0
or newer (R580+).

To use a system CUDA runtime instead of the bundled one, build with
`-Ccmake.define.BROOK_BUNDLE_CUDA_RUNTIME=OFF` (Python) or
`-DBROOK_BUNDLE_CUDA_RUNTIME=OFF` (SDK). Package managers that supply the CUDA runtime,
such as conda-forge, build with `-DBROOK_BUNDLE_CUDA_RUNTIME=OFF`.

## Startup check

Before it first uses a GPU, Brook checks its compute capability, the driver version
and whether the build has GPU code the driver can load. On an unsupported setup it
raises `brook.device.CudaError` (a `RuntimeError`) that names the problem and the
fix, for example:

```text
This Brook build has no GPU code for CUDA device 0 (NVIDIA H100 80GB HBM3, compute capability 9.0): it contains SASS for sm_89 and no PTX. Build Brook for this GPU with CMAKE_CUDA_ARCHITECTURES=90 (pip: --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=90) or native. [cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device]
```

`brook.device.require_gpu()` runs the check explicitly, and `brook.device.build_info()`
lists the build's SASS and PTX architectures and the CUDA toolkit, runtime and driver
versions.

## C and C++

```sh
cmake -S . -B build/sdk -DBROOK_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/sdk -j4
cmake --install build/sdk --prefix /your/sdk/prefix --component SDK
```

Include `brook/brook.h` (C) or `brook/brook.hpp` (C++). CMake projects use
`find_package(Brook CONFIG REQUIRED)` and link `Brook::brook`; pkg-config uses
`brook`. See the [C/C++ SDK documentation](../cpp/README.md).

## API stability

The stable surface is `brook.__all__`, `brook.device`, `brook.device_post` and
`brook.contrib.torch`. `brook.contrib.igneous` is experimental, and other modules are lower
level; both may change within 0.x. CUDA errors, including a failed startup check, raise
`brook.device.CudaError` (a `RuntimeError`).

## Environment variables

| Variable | Effect |
| :--- | :--- |
| `BROOK_STREAMING` | `1` or `0` turns streaming on or off for calls with `streaming=None` (the default). |
| `BROOK_STREAM_BUDGET_MB` | GPU memory budget of streamed calls, a positive integer number of MiB; any other value raises `ValueError`. Unset: a quarter of free GPU memory, at least 64 MiB. |
| `BROOK_PROFILE` | `1` or `sync`: Python-side stage timers and NVTX ranges (`brook.profile`). |
| `BROOK_GRAPHS` | `0` runs the CUDA-graph loops from the host, for Compute Sanitizer's racecheck and synccheck; same skeletons, slower. |

Other `BROOK_*` variables are internal test and tuning switches; they are not part of
the API and can change.

## Testing

Run `python -m pytest tests/python` against an installed wheel and
`ctest --test-dir build/sdk --output-on-failure` for the C/C++ suite. GPU tests are
skipped when no NVIDIA GPU is visible, and fail when the GPU, driver or build does
not pass the startup check.

For changes to CUDA code, also run Compute Sanitizer, one process at a time: memcheck and
initcheck as they are, racecheck and synccheck with `BROOK_GRAPHS=0` (with CUDA graphs on,
these two tools abort in the lockstep tracer because of a limitation of their own).
