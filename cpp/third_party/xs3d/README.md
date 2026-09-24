# Replaceable xs3d numerical library

Brook uses xs3d 1.13.0 by William Silversmith, licensed LGPL-3.0-or-later.
The GPLv3 and LGPLv3 texts and AUTHORS are included in `cpp/licenses/xs3d`
in the source tree, `share/brook/licenses/xs3d` in the SDK, and
`brook/licenses/xs3d` in the Python installation.

Source: https://pypi.org/project/xs3d/1.13.0/
Upstream: https://github.com/seung-lab/cross-section

All five upstream C++ headers are retained. Brook (2026) changes the persistent
scratch declaration in `xs3d.hpp` to `thread_local`, and passes its visited
vector and color from the section traversal into the per-point helper to avoid
repeated thread-local lookup. Brook clears scratch at the end of each shape
operation. `bridge.cpp` and `bridge.h` provide the narrow C entry
points and are also LGPL-3.0-or-later. Numerical traversal and accumulation order
are unchanged. The normal path uses the upstream bool specialization over actual
C++ bool mask objects; an optional uint8 specialization is retained for comparisons. Brook compiles this code into **a separate shared library**,
`libbrook_xs3d.so.1`, rather than incorporating its numerical headers in libbrook.

The complete corresponding source and build recipe for this library are shipped
here. Rebuild it without CUDA or Python:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

You may replace `libbrook_xs3d.so.1` next to `libbrook.so` in an SDK installation
or in `brook/.libs` in a Python installation with a modified, interface-compatible
version. Alternatively place the replacement on `LD_LIBRARY_PATH`. The exported
C interface in `bridge.h` must be retained; the internal implementation may change.
The loader uses ordinary shared-library resolution, without a checksum or other
restriction preventing replacement. Reverse engineering for debugging changes
to this library is permitted under the LGPL.

Redistributors must retain the library attribution, GPL/LGPL notices and license
texts, make the corresponding library source available as required by the license,
and preserve a mechanism to use an interface-compatible modified library. The
bundled source and normal dynamic linking are the mechanism used here. Shipping
only the DSO without notices/source availability is not the complete distribution.
