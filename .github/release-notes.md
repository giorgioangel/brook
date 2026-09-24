Brook is TEASAR skeletonization of 3D label volumes on NVIDIA GPUs (CUDA), a drop-in replacement for `kimimaro.skeletonize` with Python, C and C++ interfaces.

**Platform:** Linux x86-64; NVIDIA GPU of compute capability 8.0 or newer (Ampere or later); NVIDIA driver that supports CUDA 12.3 or newer (R545+); CPython 3.12, 3.13 or 3.14. There are no macOS wheels, and Windows is untested. Other setups need a source build from the repository.

### Install

```sh
python -m pip install brook-cu12
```

Or install one of the wheels attached below: one per CPython (`cp312`, `cp313`, `cp314`), tagged `manylinux_2_28` (glibc 2.28 or newer), with the CUDA 12.8 runtime bundled. The import package is `brook`. `SHA256SUMS` lists the checksums of the wheels and the sdist (`sha256sum -c SHA256SUMS`), and each of them has a build provenance attestation (`gh attestation verify <file> --repo {{REPOSITORY}}`).

### Changes in {{VERSION}}

{{CHANGELOG}}

### Documentation and citation

[README](https://github.com/{{REPOSITORY}}/blob/{{TAG}}/README.md) · [Changelog](https://github.com/{{REPOSITORY}}/blob/{{TAG}}/CHANGELOG.md) · [Citation](https://github.com/{{REPOSITORY}}/blob/{{TAG}}/CITATION.cff) ("Cite this repository" on the repository page)
