#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
#
# Build Brook's release sdist and manylinux_2_28 x86-64 wheels. .github/workflows/release.yml runs
# this script, so the release files come from exactly these steps. It runs inside
# quay.io/pypa/manylinux_2_28_x86_64@sha256:531d7aa844bbb0c131d4ab011d3db741c4abc8d498cd5ccc86121046f62303b4
# and needs no GPU.
#
#   tools/build_manylinux.sh sdist                  # the renamed sdist, into $OUT
#   tools/build_manylinux.sh wheel <sdist.tar.gz>   # one repaired wheel per CPython, from that sdist
#   tools/build_manylinux.sh all                    # both
# Each file passes `twine check --strict`; the log shows `rpm -qa 'cuda-*'`, `auditwheel show` and
# the SASS and PTX architectures in libbrook.
#
# The wheels compile through ccache (a pinned, checksummed release), from one extracted sdist in one
# build directory, with a cache that lives only as long as this script. libbrook and brook_xs3d do
# not depend on the Python version: the first wheel compiles them, and every later wheel takes the
# same objects from the cache and compiles only the Python extension. The log shows ccache's
# statistics and the time of each wheel.
#
# Settings (environment variables, defaults in brackets):
#   DIST        distribution name [brook-cu12]; the import package is always brook
#   CUDA        CUDA toolkit, one of the versions in cuda_rpms below [12.8]
#   PYTHONS     CPython tags [cp312-cp312 cp313-cp313 cp314-cp314]
#   CUDA_ARCHS  CMAKE_CUDA_ARCHITECTURES [the release list below]
#   OUT         output directory [dist]
#   JOBS        parallel compile jobs [number of CPUs]
#   CI_REQUIREMENTS  hash-pinned build tools; by default exported from uv.lock's `ci` group with
#               `uv export --locked`, which needs uv on PATH
#
# Local use, from the repository root (uv exports the tool list on the host first):
#   uv export --locked --only-group ci --no-emit-project -o build/ci.txt
#   docker run --rm -v "$PWD:/io" -w /io -e CI_REQUIREMENTS=build/ci.txt -e CUDA_ARCHS=89-real \
#     quay.io/pypa/manylinux_2_28_x86_64@sha256:531d7aa844bbb0c131d4ab011d3db741c4abc8d498cd5ccc86121046f62303b4 \
#     tools/build_manylinux.sh all
set -euo pipefail

MODE=${1:?usage: build_manylinux.sh sdist | wheel <sdist.tar.gz> | all}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIST=${DIST:-brook-cu12}
CUDA=${CUDA:-12.8}
PYTHONS=${PYTHONS:-cp312-cp312 cp313-cp313 cp314-cp314}
# Compute capability 8.0 is Brook's minimum. Machine code for Ampere to Blackwell, plus compute_80
# and compute_120 PTX that the driver compiles for GPUs without matching machine code.
CUDA_ARCHS=${CUDA_ARCHS:-80-real;86-real;89-real;90-real;100-real;120-real;80-virtual;120-virtual}
OUT=$(mkdir -p "${OUT:-dist}" && cd "${OUT:-dist}" && pwd)
JOBS=${JOBS:-$(nproc)}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

case "$DIST" in
  brook-cu[0-9]*) ;;
  *) echo "DIST must be brook-cu<CUDA major>, not $DIST" >&2; exit 2 ;;
esac
MAJOR=${CUDA%%.*}
if [ "$DIST" != "brook-cu$MAJOR" ]; then
  echo "$DIST does not match CUDA $CUDA" >&2
  exit 2
fi

# Exact package versions from NVIDIA's rhel8 repository (the CUDA 12.8.1 and 13.0.2 releases).
cuda_rpms() {
  case "$CUDA" in
    12.8) echo cuda-nvcc-12-8-12.8.93-1 cuda-cudart-devel-12-8-12.8.90-1 cuda-cccl-12-8-12.8.90-1 \
               cuda-cuobjdump-12-8-12.8.90-1 ;;
    13.0) echo cuda-nvcc-13-0-13.0.88-1 cuda-cudart-devel-13-0-13.0.96-1 cuda-cccl-13-0-13.0.85-1 \
               cuda-cuobjdump-13-0-13.0.85-1 ;;
    *) echo "no pinned packages for CUDA $CUDA" >&2; return 2 ;;
  esac
}

# ccache's static x86-64 build from its GitHub release, checked against the release's sha256.
CCACHE_VERSION=4.13.6
CCACHE_SHA256=09e0547a0c3b250a76675c33130366f1399f3580842fb360c052520d56214ead
install_ccache() {
  local name=ccache-$CCACHE_VERSION-linux-x86_64-musl-static
  curl -fsSL --retry 3 -o "$WORK/$name.tar.gz" \
    "https://github.com/ccache/ccache/releases/download/v$CCACHE_VERSION/$name.tar.gz"
  echo "$CCACHE_SHA256  $WORK/$name.tar.gz" | sha256sum --check --strict -
  tar -xzf "$WORK/$name.tar.gz" -C "$WORK" "$name/ccache"
  CCACHE=$WORK/$name/ccache
  export CCACHE_DIR=$WORK/ccache
  "$CCACHE" --version | head -1
}

# A virtual environment with the build tools from uv.lock, installed hash-checked.
frontend() {
  local python=$1 venv=$2
  if [ ! -f "$WORK/ci.txt" ]; then
    if [ -n "${CI_REQUIREMENTS:-}" ]; then
      cp "$CI_REQUIREMENTS" "$WORK/ci.txt"
    else
      uv export --project "$REPO" --locked --only-group ci --no-emit-project --quiet -o "$WORK/ci.txt"
    fi
  fi
  "$python" -m venv "$venv"
  "$venv/bin/python" -m pip install --quiet --disable-pip-version-check --no-deps --only-binary :all: \
    --require-hashes -r "$WORK/ci.txt"
  "$venv/bin/python" -m pip freeze --disable-pip-version-check | tr '\n' ' '
  echo
}

build_sdist() {
  echo "== sdist: $DIST"
  mkdir -p "$WORK/tree"
  tar -C "$REPO" --exclude=./.git --exclude=./.venv --exclude=./build --exclude=./dist -cf - . | tar -C "$WORK/tree" -xf -
  # The distribution name cannot be dynamic: rename it, and the CUDA classifier, for this build only.
  sed -i -e "s/^name = \"brook\"\$/name = \"$DIST\"/" \
    -e "s/\"Environment :: GPU :: NVIDIA CUDA\"/\"Environment :: GPU :: NVIDIA CUDA :: $MAJOR\"/" \
    "$WORK/tree/pyproject.toml"
  grep -qx "name = \"$DIST\"" "$WORK/tree/pyproject.toml"
  grep -q "\"Environment :: GPU :: NVIDIA CUDA :: $MAJOR\"" "$WORK/tree/pyproject.toml"
  frontend "${PYTHON:-/opt/python/cp312-cp312/bin/python}" "$WORK/sdist-frontend"
  "$WORK/sdist-frontend/bin/python" -m build --sdist --no-isolation -o "$WORK/sdist-out" "$WORK/tree"
  SDIST=$(echo "$WORK"/sdist-out/*.tar.gz)
  "$WORK/sdist-frontend/bin/twine" check --strict "$SDIST"
  cp "$SDIST" "$OUT/"
  SDIST=$OUT/$(basename "$SDIST")
}

install_cuda() {
  local rpms toolkit maxgcc
  rpms=$(cuda_rpms)
  toolkit=/usr/local/cuda-$CUDA
  dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
  # shellcheck disable=SC2086
  dnf install -y -q $rpms
  rpm -qa 'cuda-*' | sort
  export PATH=$toolkit/bin:$PATH CUDACXX=$toolkit/bin/nvcc CUDAToolkit_ROOT=$toolkit
  maxgcc=$(grep -oE '__GNUC__ > [0-9]+' "$toolkit/targets/x86_64-linux/include/crt/host_config.h" | head -1 | awk '{print $3}')
  if [ "$(gcc -dumpversion | cut -d. -f1)" -gt "$maxgcc" ]; then
    echo "== nvcc $CUDA accepts gcc <= $maxgcc: installing gcc-toolset-$maxgcc"
    dnf install -y -q "gcc-toolset-$maxgcc-gcc-c++"
    export PATH=/opt/rh/gcc-toolset-$maxgcc/root/usr/bin:$PATH
  fi
  export CC CXX
  CC=$(command -v gcc)
  CXX=$(command -v g++)
  nvcc --version | tail -2
  echo "host compiler: $CXX $("$CXX" -dumpfullversion)"
}

build_wheels() {
  local sdist=$1 src py raw wheel lib start
  test -d /opt/python || { echo "wheel builds run in the manylinux_2_28 image" >&2; exit 2; }
  mkdir -p "$WORK/sdist"
  tar -xzf "$sdist" -C "$WORK/sdist"
  src=$(echo "$WORK"/sdist/*)
  grep -qx "name = \"$DIST\"" "$src/pyproject.toml"
  install_cuda
  install_ccache
  echo "== wheels: $DIST, CUDA $CUDA, architectures $CUDA_ARCHS, $JOBS jobs"
  for py in $PYTHONS; do
    echo "== $py"
    start=$SECONDS
    frontend "/opt/python/$py/bin/python" "$WORK/frontend-$py"
    raw=$WORK/raw/$py
    # The same source and build paths for every wheel, so that ccache sees the same compiler
    # command for every object outside the Python extension. Each wheel configures afresh.
    rm -rf "$WORK/build"
    "$CCACHE" --zero-stats > /dev/null
    CMAKE_BUILD_PARALLEL_LEVEL=$JOBS "$WORK/frontend-$py/bin/python" -m build --wheel --no-isolation -o "$raw" "$src" \
      "-Cbuild-dir=$WORK/build" \
      "-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS" "-Cbuild.tool-args=-j$JOBS" \
      "-Ccmake.define.CMAKE_CUDA_HOST_COMPILER=$CXX" \
      "-Ccmake.define.CMAKE_C_COMPILER_LAUNCHER=$CCACHE" "-Ccmake.define.CMAKE_CXX_COMPILER_LAUNCHER=$CCACHE" \
      "-Ccmake.define.CMAKE_CUDA_COMPILER_LAUNCHER=$CCACHE"
    "$CCACHE" --show-stats
    # libcuda.so.1 (the driver) is loaded by the CUDA runtime at run time; --exclude only guards
    # against grafting it. The bundled runtime is left as it is.
    auditwheel repair --plat manylinux_2_28_x86_64 --exclude libcuda.so.1 -w "$WORK/repaired-$py" "$raw"/*.whl
    wheel=$(echo "$WORK/repaired-$py"/*.whl)
    auditwheel show "$wheel"
    "$WORK/frontend-$py/bin/twine" check --strict "$sdist" "$wheel"
    cp "$wheel" "$OUT/"
    rm -rf "$WORK/unpacked" && mkdir "$WORK/unpacked"
    "$WORK/frontend-$py/bin/python" -m zipfile -e "$wheel" "$WORK/unpacked"
    lib=$WORK/unpacked/brook/.libs/libbrook.so.0
    echo "SASS: $(cuobjdump --list-elf "$lib" | grep -oE 'sm_[0-9]+a?' | sort -uV | tr '\n' ' ')"
    echo "PTX:  $(cuobjdump --list-ptx "$lib" | grep -oE 'sm_[0-9]+a?' | sort -uV | tr '\n' ' ')"
    echo "== $py: $((SECONDS - start)) s"
  done
}

case "$MODE" in
  sdist) build_sdist ;;
  wheel) build_wheels "$(cd "$(dirname "${2:?wheel needs the sdist}")" && pwd)/$(basename "$2")" ;;
  all)
    build_sdist
    build_wheels "$SDIST"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
ls -l "$OUT"
