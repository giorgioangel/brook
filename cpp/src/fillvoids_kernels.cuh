// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "common.cuh"
#include "fillvoids_flood.cuh"
#include <cooperative_groups.h>
namespace brook::cuda {
namespace cg=cooperative_groups;

// Seeds: background voxels on any of the six faces.
__global__ void fill_seed(
    const unsigned char* fg, int* vis, long long sx, long long sy, long long sz,
    int* wl, int* counts)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= sx * sy * sz || fg[v] != 0) return;
  long long x, y, z; brook_unravel(v, sx, sy, x, y, z);
  if (x == 0 || y == 0 || z == 0 || x == sx - 1 || y == sy - 1 || z == sz - 1) {
    vis[v] = 1;
    wl[atomicAdd(&counts[0], 1)] = (int)v;
  }
}

// filled = foreground or unreached background; counts[4] += newly filled voxels
__global__ void fill_finish(
    const unsigned char* fg, const int* vis, long long n, unsigned char* out, int* counts)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  int hole = (fg[v] == 0 && vis[v] == 0);
  out[v] = (fg[v] != 0 || hole) ? 1 : 0;
  if (hole) atomicAdd(&counts[4], 1);
}

__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) fill_flood(
    const unsigned char* fg, int* vis, long long sx, long long sy, long long sz,
    int* wl_a, int* wl_b, int* counts)
{ fill_flood_body(fg,vis,sx,sy,sz,wl_a,wl_b,counts); }

}
