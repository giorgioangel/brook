// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "common.cuh"
namespace brook::cuda {

template<class Label>
__global__ void bbox_reduce(
    const Label* cc, long long sx, long long sy, long long sz,
    int* bbmin, int* bbmax)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long n = sx * sy * sz;
  if (v >= n) return;
  Label lab = cc[v];
  if (lab == 0) return;
  long long x, y, z;
  brook_unravel(v, sx, sy, x, y, z);
  long long b = (long long)lab * 3;
  atomicMin(&bbmin[b + 0], (int)x); atomicMax(&bbmax[b + 0], (int)x);
  atomicMin(&bbmin[b + 1], (int)y); atomicMax(&bbmax[b + 1], (int)y);
  atomicMin(&bbmin[b + 2], (int)z); atomicMax(&bbmax[b + 2], (int)z);
}

// Bounding boxes AND voxel counts in one pass, one thread per x-row: a row is walked as RUNS of
// equal labels and each run costs one count update and one bbox update (objects are many voxels
// long along x, so there are several times fewer atomics than voxels), and a bbox atomic is only
// issued when a cache-bypassing load says the run would move the box. Exact integers: the same
// boxes and counts as bbox_reduce / bincount, whatever the schedule.
template<class Label>
__global__ void analyze_rows(
    const Label* cc, long long sx, long long sy, long long sz,
    int* bbmin, int* bbmax, unsigned long long* counts)
{
  long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= sy * sz) return;
  int y = (int)(r % sy), z = (int)(r / sy);
  const Label* row = cc + r * sx;
  long long x0 = 0;
  while (x0 < sx) {
    Label lab = row[x0];
    long long x1 = x0 + 1;
    while (x1 < sx && row[x1] == lab) x1++;
    if (lab != 0) {
      long long b = (long long)lab * 3;
      atomicAdd(&counts[lab], (unsigned long long)(x1 - x0));
      if (__ldcg(&bbmin[b + 0]) > (int)x0) atomicMin(&bbmin[b + 0], (int)x0);
      if (__ldcg(&bbmax[b + 0]) < (int)(x1 - 1)) atomicMax(&bbmax[b + 0], (int)(x1 - 1));
      if (__ldcg(&bbmin[b + 1]) > y) atomicMin(&bbmin[b + 1], y);
      if (__ldcg(&bbmax[b + 1]) < y) atomicMax(&bbmax[b + 1], y);
      if (__ldcg(&bbmin[b + 2]) > z) atomicMin(&bbmin[b + 2], z);
      if (__ldcg(&bbmax[b + 2]) < z) atomicMax(&bbmax[b + 2], z);
    }
    x0 = x1;
  }
}

}
