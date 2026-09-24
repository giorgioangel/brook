// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#define BROOK_COOP_BLOCK 128
#define BROOK_COOP_MIN_BLOCKS 8
namespace brook::cuda {

// NVRTC has no system include path, so define the fixed-width integer types we
// use directly (these match CUDA's ABI: int=32-bit, long long=64-bit).

// Fortran-order flat index for a volume of shape (sx, sy, sz).
#define BROOK_IDX(x, y, z, sx, sy) \
    ( (long long)(x) + (long long)(sx) * ( (long long)(y) + (long long)(sy) * (long long)(z) ) )

// Decode a Fortran-order flat index back into (x, y, z).
__device__ __forceinline__ void brook_unravel(
    long long loc, long long sx, long long sy,
    long long &x, long long &y, long long &z
) {
    long long sxy = sx * sy;
    z = loc / sxy;
    long long rem = loc - z * sxy;
    y = rem / sx;
    x = rem - y * sx;
}

// 26-neighbour offsets (dx, dy, dz). Index order is fixed and shared by the CCL,
// SSSP and invalidation kernels. Layout: 0-5 face (6-hood), 6-17 edge (the three
// axis-pair diagonal groups xy/yz/xz), 18-25 corner. Each triple is distinct and
// the 26 together are exactly {-1,0,1}^3 \ {(0,0,0)}.
static __constant__ int BROOK_NX[26] = {
    -1, 1, 0, 0, 0, 0,                              // face
    -1,-1, 1, 1,   0, 0, 0, 0,  -1,-1, 1, 1,        // edges: xy, yz, xz
    -1,-1,-1,-1, 1, 1, 1, 1                         // corners
};
static __constant__ int BROOK_NY[26] = {
     0, 0,-1, 1, 0, 0,
    -1, 1,-1, 1,  -1,-1, 1, 1,   0, 0, 0, 0,
    -1,-1, 1, 1,-1,-1, 1, 1
};
static __constant__ int BROOK_NZ[26] = {
     0, 0, 0, 0,-1, 1,
     0, 0, 0, 0,  -1, 1,-1, 1,  -1, 1,-1, 1,
    -1, 1,-1, 1,-1, 1,-1, 1
};

// voxel_connectivity_graph bit per neighbour (cc3d convention), aligned to the
// BROOK_N* order above. neighbour k is permitted from voxel v iff
// (graph[v] >> BROOK_BIT[k]) & 1, the bit order of Kimimaro's dijkstra_invalidation.hpp.
static __constant__ int BROOK_BIT[26] = {
     1, 0, 3, 2, 5, 4,                 // face: -x +x -y +y -z +z
     9, 7, 8, 6, 17,13,16,12, 15,11,14,10,  // edges: xy, yz, xz
    25,21,23,19,24,20,22,18            // corners
};

// Index of the opposite (negated) offset, for pull-style gating: a pull of v
// from neighbour u = v + offset[k] is the edge u->v, whose direction is
// offset[BROOK_OPP[k]] in u's frame.
static __constant__ int BROOK_OPP[26] = {
     1, 0, 3, 2, 5, 4,
     9, 8, 7, 6, 13,12,11,10, 17,16,15,14,
    25,24,23,22,21,20,19,18
};

// dijkstra3d's neighbour order (compute_neighborhood): the faces and edges as above, then the
// corners (-,-,-)(+,-,-)(-,+,-)(-,-,+)(+,+,-)(+,-,+)(-,+,+)(+,+,+). Its railroad joins the FIRST
// rail it relaxes from the first popped voxel u that touches one, gated by bit BROOK_BIT[k] of
// graph[u] (brook_vg_ok_push); with a voxel graph its corner slots at x = 0 and x = sx - 1
// degenerate to yz-edge offsets under the corner's bit, which Brook does not reproduce.
// D3RANK: neighbour k -> its position in that order; D3INV: position -> k;
// KOF: (dx + 1) + 3 (dy + 1) + 9 (dz + 1) -> k (-1 at the centre).
static __constant__ int BROOK_D3RANK[26] = {
     0, 1, 2, 3, 4, 5,  6, 7, 8, 9,10,11,12,13,14,15,16,17,
    18,21,20,24,19,23,22,25
};
static __constant__ int BROOK_D3INV[26] = {
     0, 1, 2, 3, 4, 5,  6, 7, 8, 9,10,11,12,13,14,15,16,17,
    18,22,20,19,24,23,21,25
};
static __constant__ int BROOK_KOF[27] = {
    18,10,22,14, 4,16,20,12,24,  6, 2, 8, 0,-1, 1, 7, 3, 9,  19,11,23,15, 5,17,21,13,25
};

// Pull gate: is the edge u -> v (u = v + offset[k]) permitted by voxel graph vg?
// The edge direction u->v is -offset[k] = offset[OPP[k]] in u's frame.
__device__ __forceinline__ bool brook_vg_ok(
    const unsigned int* vg, int has_vg, long long u, int k) {
    if (!has_vg) return true;
    return ((vg[u] >> BROOK_BIT[BROOK_OPP[k]]) & 1u) != 0u;
}

// Push gate: is the edge u -> v (v = u + offset[k]) permitted? The direction
// u->v is offset[k] in u's (the frontier voxel's) frame.
__device__ __forceinline__ bool brook_vg_ok_push(
    const unsigned int* vg, int has_vg, long long u, int k) {
    if (!has_vg) return true;
    return ((vg[u] >> BROOK_BIT[k]) & 1u) != 0u;
}

// Cache-bypassing loads for global state that OTHER blocks mutate within the same
// kernel. Inside one long-running cooperative kernel a plain load can be served from
// the block's local line cache filled before another block's atomic update, and stay
// stale for the kernel's lifetime: a frontier voxel then propagates an outdated
// distance and, since it was already flagged, is never re-enqueued. The host loop
// never sees this because every round is a separate launch. Use these for dist/key/flag/header reads in cooperative
// kernels; per-kernel-constant inputs (field, DBF, vg) may stay plain loads.
#define BROOK_VLOAD_F(p)   __ldcg((const float*)(p))
#define BROOK_VLOAD_I(p)   __ldcg((const int*)(p))
#define BROOK_VLOAD_U64(p) __ldcg((const unsigned long long*)(p))

// One vertex-weighted step: fl(d + f), but NEVER d itself. Where the cost of entering a voxel is
// below float32 resolution at the current distance (steep PDRF exponents in thick regions, large
// distances), d + f == d: whole regions would share one distance, and every consumer that
// derives predecessors from the field ("the smallest neighbour") loses its orientation there --
// the backtrace walks in circles (on the hemibrain volume with pdrf_exponent = 8, into a
// 9.5-million-vertex "path"). Advancing by one ulp instead keeps distances strictly increasing
// along every step, so the smallest neighbour is always strictly nearer and walks terminate.
// The operator is monotone in d, so the flood's fixed point stays unique (independent of thread
// timing); the error is <= one ulp per absorbed step. Zero-cost voxels (rails, f == 0) are
// exempt: they absorb and are never expanded, and their distances order the rail choice.
__device__ __forceinline__ float brook_vw_step(float d, float f) {
    float c = d + f;
    if (c == d && f > 0.0f && d < __int_as_float(0x7f800000)) c = __int_as_float(__float_as_int(d) + 1);
    return c;
}

// atomicMin on the int reinterpretation of a non-negative float: the IEEE bit
// patterns of floats >= 0 (including +inf = 0x7f800000) are monotone as signed
// ints, so one native 32-bit atomicMin does the job of a CAS retry loop and
// returns the previous value as such a loop would. The pre-check load skips
// the atomic when no improvement is possible (the common case). All distance
// fields are >= 0.
__device__ __forceinline__ float brook_atomic_min_f(float* addr, float val) {
    float cur = *((volatile float*)addr);
    if (cur <= val) return cur;
    int old = atomicMin((int*)addr, __float_as_int(val));
    return __int_as_float(old);
}
}
