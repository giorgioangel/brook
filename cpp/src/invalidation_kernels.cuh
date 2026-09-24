// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "common.cuh"
#include <cooperative_groups.h>
namespace brook::cuda {
namespace cg=cooperative_groups;

#define BROOK_INV_MAXKEY 0xFFFFFFFFFFFFFFFFULL

// Offer voxel v the source `su` (straight-line distance d, already inside su's ball) from a
// claimed neighbour whose bottleneck distance has the bits muBits.
__device__ __forceinline__ void inval_offer(
    long long v, float d, int muBits, int su, int hiBits, unsigned long long* key, int* inq,
    int* nxt, int* ncount, int* far, int* fcount, unsigned long long* pb, int lane_max,
    int* touched, int* tcount)
{
  int mb = __float_as_int(d);
  if (mb < muBits) mb = muBits;                                  // M = max(M[u], d)
  if (mb < hiBits) {                                             // inside the band: claim now
    unsigned long long candkey = ((unsigned long long)(unsigned int)mb << 32)
                                 | (unsigned long long)(unsigned int)su;
    unsigned long long old = atomicMin(key + v, candkey);
    if (old > candkey) {
      if (old == BROOK_INV_MAXKEY) touched[atomicAdd(tcount, 1)] = (int)v;   // first write
      if (atomicExch(&inq[v], 1) != 1) nxt[atomicAdd(ncount, 1)] = (int)v;
    }
  } else if (BROOK_VLOAD_U64(&key[v]) == BROOK_INV_MAXKEY) {     // beyond it: wait for its band
    if (atomicCAS((unsigned int*)&inq[v], 0u, 2u) == 0u) {
      int pos = atomicAdd(fcount, 1);
      far[pos] = (int)v;
      if (pos < lane_max) pb[pos] = BROOK_INV_MAXKEY;           // its slot for a lane-parallel pull
    }
  }
}

// Expand frontier voxel u from the (source, M) it held at the round boundary.
__device__ __forceinline__ void inval_expand(
    int u32, int su32, int muBits, int hiBits, const unsigned char* field, const float* DBF,
    unsigned long long* key, const unsigned int* vg, int has_vg,
    long long sx, long long sy, long long sz,
    float wx, float wy, float wz, float scale, float constant,
    int* nxt, int* ncount, int* far, int* fcount, unsigned long long* pb, int lane_max,
    int* inq, int* touched, int* tcount)
{
  long long u = (long long)u32;            // int32 flat index
  long long su = (long long)su32;
  long long sux, suy, suz; brook_unravel(su, sx, sy, sux, suy, suz);
  float maxd = scale * DBF[su] + constant;
  long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
  for (int k = 0; k < 26; k++) {
    long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    long long v = BROOK_IDX(nx, ny, nz, sx, sy);
    if (field[v] == 0) continue;
    if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;   // push: gate vg[u] dir k
    float dvx = wx * (float)(nx - sux), dvy = wy * (float)(ny - suy), dvz = wz * (float)(nz - suz);
    float d = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);     // straight-line v -> u's source
    if (d >= maxd) continue;                                 // gate (strict <)
    inval_offer(v, d, muBits, su32, hiBits, key, inq, nxt, ncount, far, fcount, pb, lane_max, touched, tcount);
  }
}

// Band boundary: waiting voxel c takes the best offer of its claimed neighbours (all settled).
// Returns the source (or -1: claimed meanwhile, -2: no offer in reach) and the bits of M.
__device__ __forceinline__ int inval_pull(
    long long c, const float* DBF, const unsigned long long* key,
    const unsigned int* vg, int has_vg, long long sx, long long sy, long long sz,
    float wx, float wy, float wz, float scale, float constant, int* mbits)
{
  if (BROOK_VLOAD_U64(&key[c]) != BROOK_INV_MAXKEY) return -1;
  unsigned long long best = BROOK_INV_MAXKEY;
  long long x, y, z; brook_unravel(c, sx, sy, x, y, z);
  for (int k = 0; k < 26; k++) {
    long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    long long u = BROOK_IDX(nx, ny, nz, sx, sy);
    unsigned long long ku = BROOK_VLOAD_U64(&key[u]);
    if (ku == BROOK_INV_MAXKEY) continue;
    if (!brook_vg_ok(vg, has_vg, u, k)) continue;            // pull: the edge u -> c
    long long su = (long long)(ku & 0xffffffffULL);
    long long sux, suy, suz; brook_unravel(su, sx, sy, sux, suy, suz);
    float dvx = wx * (float)(x - sux), dvy = wy * (float)(y - suy), dvz = wz * (float)(z - suz);
    float d = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
    if (d >= scale * DBF[su] + constant) continue;
    int mb = __float_as_int(d), mu = (int)(ku >> 32);
    if (mb < mu) mb = mu;
    unsigned long long cand = ((unsigned long long)(unsigned int)mb << 32) | (unsigned long long)(unsigned int)su;
    if (cand < best) best = cand;
  }
  if (best == BROOK_INV_MAXKEY) return -2;
  *mbits = (int)(best >> 32);
  return (int)(best & 0xffffffffULL);
}

// Band index of a bottleneck distance on the fixed grid [k*delta, (k+1)*delta), never below kmin.
__device__ __forceinline__ int inval_band(int mbits, float delta, int kmin) {
  float m = __int_as_float(mbits);
  int k = (int)(m / delta);
  while ((float)(k + 1) * delta <= m) k++;
  while (k > 0 && (float)k * delta > m) k--;
  return k < kmin ? kmin : k;
}

__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) inval_coop(
    unsigned char* field, const float* DBF, unsigned long long* key, int* snap, int* snapm,
    const unsigned int* vg, int has_vg,
    long long sx, long long sy, long long sz,
    float wx, float wy, float wz, float scale, float constant, float delta,
    const long long* path, int npath,
    int* wl_a, int* wl_b, int* far_a, int* far_b, int* inq, int* touched, int* hdr, int max_rounds,
    int lane_max, float* smax, long long* sxyz, unsigned long long* pb_a, unsigned long long* pb_b)
{
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  // ---- seed: path vertices that are still foreground become sources (M = 0) ----
  for (int i = tid; i < npath; i += nthreads) {
    long long p = path[i];
    if (field[p] == 0) continue;                                // invalidated earlier
    unsigned long long old = atomicMin(key + p, (unsigned long long)(unsigned int)p);
    if (old == BROOK_INV_MAXKEY) {                               // dedupes repeated vertices
      inq[p] = 1;
      wl_a[atomicAdd(&hdr[0], 1)] = (int)p;
      touched[atomicAdd(&hdr[2], 1)] = (int)p;
    }
  }
  if (tid == 0) hdr[7] = 0x7f800000;
  __threadfence(); grid.sync();

  int* cur = wl_a; int* nxt = wl_b; int* far = far_a; int* keep = far_b;
  unsigned long long* pbf = pb_a; unsigned long long* pbk = pb_b;
  int band = 0, round = 0, open = 1, presnap = 0;
  while (open) {
    int hiBits = __float_as_int((float)(band + 1) * delta);
    for (; round < max_rounds; round++) {                        // rounds inside the band
      int fc = BROOK_VLOAD_I(&hdr[0]);
      if (fc == 0) break;
      // phase 1: snapshot (source, M) at the round boundary, re-arm inq. Skipped for the first
      // round after a lane-parallel band boundary: the promotion wrote the snapshot itself.
      int lanes = (fc <= lane_max);
      if (!presnap) {
      for (int i = tid; i < fc; i += nthreads) {
        long long u = (long long)BROOK_VLOAD_I(&cur[i]);
        unsigned long long ku = BROOK_VLOAD_U64(&key[u]);
        int su = (int)(ku & 0xffffffffULL);
        snap[i] = su; snapm[i] = (int)(ku >> 32);
        inq[u] = 0;
        if (lanes) {               // thin front: one lane per neighbour (lane_max)
          long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
          long long qx, qy, qz; brook_unravel((long long)su, sx, sy, qx, qy, qz);
          sxyz[2 * i] = x | (y << 21) | (z << 42);
          sxyz[2 * i + 1] = qx | (qy << 21) | (qz << 42);
          smax[i] = scale * DBF[su] + constant;
        }
      }
      __threadfence(); grid.sync();
      }
      presnap = 0;
      // phase 2: expand from the snapshot
      if (lanes) {
        long long items = (long long)fc << 5;
        for (long long j = tid; j < items; j += nthreads) {     // inval_expand's body, neighbour k
          int k = (int)(j & 31);
          if (k >= 26) continue;
          int i = (int)(j >> 5);
          long long p = (long long)__ldcg(&sxyz[2 * i]), q = (long long)__ldcg(&sxyz[2 * i + 1]);
          long long nx = (p & 0x1FFFFF) + BROOK_NX[k], ny = ((p >> 21) & 0x1FFFFF) + BROOK_NY[k],
                    nz = (p >> 42) + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long v = BROOK_IDX(nx, ny, nz, sx, sy);
          if (field[v] == 0) continue;
          long long u = (long long)BROOK_VLOAD_I(&cur[i]);
          if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
          long long sux = q & 0x1FFFFF, suy = (q >> 21) & 0x1FFFFF, suz = q >> 42;
          float dvx = wx * (float)(nx - sux), dvy = wy * (float)(ny - suy), dvz = wz * (float)(nz - suz);
          float d = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
          if (d >= BROOK_VLOAD_F(&smax[i])) continue;
          inval_offer(v, d, BROOK_VLOAD_I(&snapm[i]), BROOK_VLOAD_I(&snap[i]), hiBits, key, inq,
                      nxt, &hdr[1], far, &hdr[5], pbf, lane_max, touched, &hdr[2]);
        }
      } else
      for (int i = tid; i < fc; i += nthreads)
        inval_expand(BROOK_VLOAD_I(&cur[i]), BROOK_VLOAD_I(&snap[i]), BROOK_VLOAD_I(&snapm[i]), hiBits,
                     field, DBF, key, vg, has_vg, sx, sy, sz, wx, wy, wz, scale, constant,
                     nxt, &hdr[1], far, &hdr[5], pbf, lane_max, inq, touched, &hdr[2]);
      __threadfence(); grid.sync();
      if (tid == 0) { hdr[0] = BROOK_VLOAD_I(&hdr[1]); hdr[1] = 0; }
      int* t = cur; cur = nxt; nxt = t;
      __threadfence(); grid.sync();
    }
    // ---- band boundary: the waiting voxels pull from their (settled) claimed neighbours.
    // Windows: pull | promote | publish; hdr[7] was armed in the previous publish window.
    // A thin waiting list pulls with one lane per (voxel, neighbour) into its slot pbf[i] (a
    // serial 26-neighbour loop is ~10 barriers long) and its promotion writes the snapshot of
    // the first round, which then starts at phase 2.
    int nfar = BROOK_VLOAD_I(&hdr[5]);
    if (nfar == 0 || round >= max_rounds) { open = 0; continue; }
    int plan = (nfar <= lane_max);
    if (plan) {
      long long items = (long long)nfar << 5;
      for (long long j = tid; j < items; j += nthreads) {        // inval_pull's body, neighbour k
        int k = (int)(j & 31);
        if (k >= 26) continue;
        int i = (int)(j >> 5);
        long long c = (long long)BROOK_VLOAD_I(&far[i]);
        if (BROOK_VLOAD_U64(&key[c]) != BROOK_INV_MAXKEY) continue;
        long long x, y, z; brook_unravel(c, sx, sy, x, y, z);
        long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
        long long u = BROOK_IDX(nx, ny, nz, sx, sy);
        unsigned long long ku = BROOK_VLOAD_U64(&key[u]);
        if (ku == BROOK_INV_MAXKEY) continue;
        if (!brook_vg_ok(vg, has_vg, u, k)) continue;
        long long su = (long long)(ku & 0xffffffffULL);
        long long sux, suy, suz; brook_unravel(su, sx, sy, sux, suy, suz);
        float dvx = wx * (float)(x - sux), dvy = wy * (float)(y - suy), dvz = wz * (float)(z - suz);
        float d = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
        if (d >= scale * DBF[su] + constant) continue;
        int mb = __float_as_int(d), mu = (int)(ku >> 32);
        if (mb < mu) mb = mu;
        atomicMin(pbf + i, ((unsigned long long)(unsigned int)mb << 32) | (unsigned long long)(unsigned int)su);
        atomicMin(&hdr[7], mb);
      }
    } else
    for (int i = tid; i < nfar; i += nthreads) {
      int mb = 0;
      int s = inval_pull((long long)BROOK_VLOAD_I(&far[i]), DBF, key, vg, has_vg, sx, sy, sz,
                         wx, wy, wz, scale, constant, &mb);
      snap[i] = s; snapm[i] = mb;
      if (s >= 0) atomicMin(&hdr[7], mb);
    }
    __threadfence(); grid.sync();
    int lowest = BROOK_VLOAD_I(&hdr[7]);
    if (lowest != 0x7f800000) band = inval_band(lowest, delta, band + 1);
    hiBits = __float_as_int((float)(band + 1) * delta);
    for (int i = tid; i < nfar; i += nthreads) {                 // promote / keep waiting / drop
      int c = BROOK_VLOAD_I(&far[i]);
      int s, mb;
      if (plan) {
        if (BROOK_VLOAD_U64(&key[c]) != BROOK_INV_MAXKEY) continue;          // claimed inside the last band
        unsigned long long b = BROOK_VLOAD_U64(&pbf[i]);
        if (b == BROOK_INV_MAXKEY) { atomicExch(&inq[c], 0); continue; }     // no offer in reach (may return)
        s = (int)(b & 0xffffffffULL); mb = (int)(b >> 32);
      } else {
        s = BROOK_VLOAD_I(&snap[i]);
        if (s == -1) continue;
        if (s == -2) { atomicExch(&inq[c], 0); continue; }
        mb = BROOK_VLOAD_I(&snapm[i]);
      }
      if (mb < hiBits) {
        key[c] = ((unsigned long long)(unsigned int)mb << 32) | (unsigned long long)(unsigned int)s;
        touched[atomicAdd(&hdr[2], 1)] = c;
        int pos = atomicAdd(&hdr[1], 1);
        nxt[pos] = c;
        if (plan) {                                              // = phase 1 of its first round
          atomicExch(&inq[c], 0);
          snap[pos] = s; snapm[pos] = mb;
          long long x, y, z; brook_unravel((long long)c, sx, sy, x, y, z);
          long long qx, qy, qz; brook_unravel((long long)s, sx, sy, qx, qy, qz);
          sxyz[2 * pos] = x | (y << 21) | (z << 42);
          sxyz[2 * pos + 1] = qx | (qy << 21) | (qz << 42);
          smax[pos] = scale * DBF[s] + constant;
        } else atomicExch(&inq[c], 1);
      } else {
        int pos = atomicAdd(&hdr[6], 1);
        keep[pos] = c;
        if (pos < lane_max) pbk[pos] = BROOK_INV_MAXKEY;
      }
    }
    __threadfence(); grid.sync();
    if (tid == 0) {
      hdr[0] = BROOK_VLOAD_I(&hdr[1]); hdr[1] = 0;
      hdr[5] = BROOK_VLOAD_I(&hdr[6]); hdr[6] = 0;
      hdr[7] = 0x7f800000;
    }
    { int* t = cur; cur = nxt; nxt = t; t = far; far = keep; keep = t; }
    { unsigned long long* t = pbf; pbf = pbk; pbk = t; }
    presnap = plan;
    round++;
    __threadfence(); grid.sync();
  }

  // ---- finish: invalidated set == touched; clear, restore, publish, re-arm ----
  int n = BROOK_VLOAD_I(&hdr[2]);
  for (int i = tid; i < n; i += nthreads) {
    int v = BROOK_VLOAD_I(&touched[i]);
    field[v] = 0; key[v] = BROOK_INV_MAXKEY; inq[v] = 0;
  }
  __threadfence(); grid.sync();
  // hdr[3] = count, or -1 if max_rounds was exhausted (the host raises, as the host loop does)
  if (tid == 0) {
    hdr[3] = (BROOK_VLOAD_I(&hdr[0]) == 0 && BROOK_VLOAD_I(&hdr[5]) == 0) ? n : -1;
    hdr[0] = 0; hdr[1] = 0; hdr[2] = 0; hdr[5] = 0; hdr[6] = 0;
  }
}

}
