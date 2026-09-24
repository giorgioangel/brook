// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "common.cuh"
#include <cooperative_groups.h>
namespace brook::cuda {

// Breadth-first flood of the background through the 6 face neighbours (BROOK_N* 0..5). The
// exactly-once test-and-set keeps every voxel on one worklist once. Ends when the front is
// empty (every voxel is visited at most once, so it always does).
__device__ __forceinline__ void fill_flood_body(
    const unsigned char* fg, int* vis, long long sx, long long sy, long long sz,
    int* wl_a, int* wl_b, int* counts)
{
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  int* cur = wl_a; int* nxt = wl_b;
  for (;;) {
    int fc = BROOK_VLOAD_I(&counts[0]);
    if (fc == 0) break;
    for (int i = tid; i < fc; i += nthreads) {
      long long u = (long long)BROOK_VLOAD_I(&cur[i]);
      long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
      for (int k = 0; k < 6; k++) {
        long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
        long long v = BROOK_IDX(nx, ny, nz, sx, sy);
        if (fg[v] != 0) continue;
        if (atomicExch(&vis[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
      }
    }
    __threadfence(); grid.sync();
    if (tid == 0) { counts[0] = BROOK_VLOAD_I(&counts[1]); counts[1] = 0; counts[2] += 1; }
    int* t = cur; cur = nxt; nxt = t;
    __threadfence(); grid.sync();
  }
}

}
