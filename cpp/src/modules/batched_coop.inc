// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti

#define BROOK_INF __int_as_float(0x7f800000)

// Multi-source, label-gated frontier flood. mode 0: geometric (edge weight W[k]);
// mode 1: vertex-weighted (cost of entering v = vwfield[v]). Follows the ordering rules of
// every brook flood kernel (atomic flag clear, distance read data-dependent on it, fenced
// barriers, cache-bypassing loads of kernel-mutated memory).
__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) flood_lab(
    float* dist, const LABEL_T* lab, const float* vwfield, const float* W, int mode,
    long long sx, long long sy, long long sz,
    int* wl_a, int* wl_b, int* counts, int* inq, int fcount0, int max_rounds,
    int lane_max, float* sdu, long long* sxyz, unsigned long long* stats,
    const unsigned int* nmask, int has_mask)
{
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  if (tid == 0) { counts[0] = fcount0; counts[1] = 0; stats[0] = 0; stats[1] = 0; }
  __threadfence(); grid.sync();
  // Neighbour token (nbr_mask): bit k of nmask[u] = "neighbour k of u is inside the volume and
  // has u's label". One load replaces 26 bounds checks and 26 label loads, and the neighbour is
  // u + off[k]: the same relaxations in the same order, so the same fixed point.
  long long off[26];
  for (int k = 0; k < 26; k++) off[k] = BROOK_NX[k] + sx * (BROOK_NY[k] + sy * BROOK_NZ[k]);

  int* cur = wl_a; int* nxt = wl_b;
  int round = 0;
  for (; round < max_rounds; round++) {
    int fcount = BROOK_VLOAD_I(&counts[0]);
    if (fcount == 0) break;
    if (tid == 0) { stats[0] += (unsigned long long)fcount; if (fcount <= lane_max) stats[1] += 1ULL; }
    if (fcount <= lane_max) {
      // thin front: one lane per neighbour (lane_max)
      for (int i = tid; i < fcount; i += nthreads) {
        long long u = (long long)BROOK_VLOAD_I(&cur[i]);
        int was = atomicExch(&inq[u], 0);
        sdu[i] = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);
        if (has_mask) { sxyz[i] = (long long)nmask[u]; continue; }
        long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
        sxyz[i] = x | (y << 21) | (z << 42);
      }
      __threadfence(); grid.sync();
      long long items = (long long)fcount << 5;
      for (long long j = tid; j < items; j += nthreads) {
        int k = (int)(j & 31);
        if (k >= 26) continue;
        int i = (int)(j >> 5);
        long long p = (long long)__ldcg(&sxyz[i]);
        if (has_mask) {
          if (!((p >> k) & 1LL)) continue;
          long long v = (long long)BROOK_VLOAD_I(&cur[i]) + off[k];
          float du = BROOK_VLOAD_F(&sdu[i]);
          float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand) {
            if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
          }
          continue;
        }
        long long nx = (p & 0x1FFFFF) + BROOK_NX[k], ny = ((p >> 21) & 0x1FFFFF) + BROOK_NY[k],
                  nz = (p >> 42) + BROOK_NZ[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
        long long u = (long long)BROOK_VLOAD_I(&cur[i]);
        long long v = BROOK_IDX(nx, ny, nz, sx, sy);
        if (lab[v] != lab[u]) continue;
        float du = BROOK_VLOAD_F(&sdu[i]);
        float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
        float old = brook_atomic_min_f(&dist[v], cand);
        if (old > cand) {
          if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
        }
      }
    } else
    for (int i = tid; i < fcount; i += nthreads) {
      long long u = (long long)BROOK_VLOAD_I(&cur[i]);
      int was = atomicExch(&inq[u], 0);
      float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);   // ordered after the clear
      if (has_mask) {
        unsigned int m = nmask[u];
        for (int k = 0; k < 26; k++) {
          if (!((m >> k) & 1u)) continue;
          long long v = u + off[k];
          float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand) {
            if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
          }
        }
        continue;
      }
      LABEL_T lu = lab[u];
      long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
      for (int k = 0; k < 26; k++) {
        long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
        long long v = BROOK_IDX(nx, ny, nz, sx, sy);
        if (lab[v] != lu) continue;                       // never leave the component
        float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
        float old = brook_atomic_min_f(&dist[v], cand);
        if (old > cand) {
          if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
        }
      }
    }
    __threadfence(); grid.sync();
    if (tid == 0) { counts[0] = BROOK_VLOAD_I(&counts[1]); counts[1] = 0; }
    int* tmp = cur; cur = nxt; nxt = tmp;
    __threadfence(); grid.sync();
  }
  if (tid == 0) { counts[2] = round; counts[3] = (BROOK_VLOAD_I(&counts[0]) != 0) ? 1 : 0; }
}

// Banded (near-far) version of flood_lab for fields with a large dynamic range (the PDRF spans 5
// decades): plain label-correcting rounds can expand a voxel repeatedly, because a
// voxel is first reached by an expensive direct hop and later improved through the cheap core.
// Here a relaxation only enqueues v when cand <= T; farther voxels just keep their tentative
// distance. When the band is exhausted, T grows GEOMETRICALLY (T <- growth * T, or min(far) + T0
// if that band is empty: few phases whatever the range) and one dense pass over the volume
// enqueues every voxel with T_old < dist <= T_new -- exactly the deferred ones, none of which
// was expanded yet -- so no far lists or flags are needed. Same monotone atomic-min fixed point as flood_lab.
// tbuf: [1] T [2] growth [3] T0; counts: [0],[1] frontier [2] rounds [3] exhausted [5] phases;
// minfar: min(far) bits (own array: polled and atomicMin'd by all); pub: thread 0's publications
__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) flood_lab_nf(
    float* dist, const LABEL_T* lab, const float* vwfield, const float* W, int mode,
    long long sx, long long sy, long long sz,
    int* wl_a, int* wl_b, int* counts, int* inq, int fcount0, int max_rounds, int max_phases,
    float* tbuf, int* minfar, int* pub,
    int lane_max, float* sdu, long long* sxyz, unsigned long long* stats,
    const unsigned int* nmask, int has_mask)
{
  long long off[26];                                             // Neighbour token, see flood_lab
  for (int k = 0; k < 26; k++) off[k] = BROOK_NX[k] + sx * (BROOK_NY[k] + sy * BROOK_NZ[k]);
  cg::grid_group grid = cg::this_grid();
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long nthreads = (long long)gridDim.x * blockDim.x;
  long long n = sx * sy * sz;
  if (tid == 0) { counts[0] = fcount0; counts[1] = 0; counts[2] = 0; counts[3] = 0; counts[5] = 0;
                  minfar[0] = 0x7f800000; pub[0] = 0; pub[1] = 0; stats[0] = 0; stats[1] = 0; }
  __threadfence(); grid.sync();

  int* cur = wl_a; int* nxt = wl_b;
  int rounds = 0;
  float growth = tbuf[2], t0 = tbuf[3];
  for (int phase = 0; phase < max_phases; phase++) {
    float T = BROOK_VLOAD_F(&tbuf[1]);
    for (; rounds < max_rounds; rounds++) {
      int fcount = BROOK_VLOAD_I(&counts[0]);
      if (fcount == 0) break;
      if (tid == 0) { stats[0] += (unsigned long long)fcount; if (fcount <= lane_max) stats[1] += 1ULL; }
      if (fcount <= lane_max) {                                  // thin front: lanes (see _gpu)
        for (long long i = tid; i < fcount; i += nthreads) {
          long long u = (long long)BROOK_VLOAD_I(&cur[i]);
          int was = atomicExch(&inq[u], 0);
          sdu[i] = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);
          if (has_mask) { sxyz[i] = (long long)nmask[u]; continue; }
          long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
          sxyz[i] = x | (y << 21) | (z << 42);
        }
        __threadfence(); grid.sync();
        long long items = (long long)fcount << 5;
        for (long long j = tid; j < items; j += nthreads) {
          int k = (int)(j & 31);
          if (k >= 26) continue;
          int i = (int)(j >> 5);
          long long p = (long long)__ldcg(&sxyz[i]);
          if (has_mask) {
            if (!((p >> k) & 1LL)) continue;
            long long v = (long long)BROOK_VLOAD_I(&cur[i]) + off[k];
            float du = BROOK_VLOAD_F(&sdu[i]);
            float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
            float old = brook_atomic_min_f(&dist[v], cand);
            if (old > cand && cand <= T) {
              if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
            }
            continue;
          }
          long long nx = (p & 0x1FFFFF) + BROOK_NX[k], ny = ((p >> 21) & 0x1FFFFF) + BROOK_NY[k],
                    nz = (p >> 42) + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long u = (long long)BROOK_VLOAD_I(&cur[i]);
          long long v = BROOK_IDX(nx, ny, nz, sx, sy);
          if (lab[v] != lab[u]) continue;
          float du = BROOK_VLOAD_F(&sdu[i]);
          float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand && cand <= T) {
            if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
          }
        }
      } else
      for (long long i = tid; i < fcount; i += nthreads) {
        long long u = (long long)BROOK_VLOAD_I(&cur[i]);
        int was = atomicExch(&inq[u], 0);
        float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);   // ordered after the clear
        if (has_mask) {
          unsigned int m = nmask[u];
          for (int k = 0; k < 26; k++) {
            if (!((m >> k) & 1u)) continue;
            long long v = u + off[k];
            float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
            float old = brook_atomic_min_f(&dist[v], cand);
            if (old > cand && cand <= T) {
              if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
            }
          }
          continue;
        }
        LABEL_T lu = lab[u];
        long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
        for (int k = 0; k < 26; k++) {
          long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long v = BROOK_IDX(nx, ny, nz, sx, sy);
          if (lab[v] != lu) continue;
          float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand && cand <= T) {                         // beyond the band: just keep dist
            if (atomicExch(&inq[v], 1) == 0) nxt[atomicAdd(&counts[1], 1)] = (int)v;
          }
        }
      }
      __threadfence(); grid.sync();
      if (tid == 0) { counts[0] = BROOK_VLOAD_I(&counts[1]); counts[1] = 0; }
      int* tmp = cur; cur = nxt; nxt = tmp;
      __threadfence(); grid.sync();
    }
    // ---- phase end: ONE dense pass promotes the next band (T, growth*T] and tracks the min of
    // what stays deferred; only if that band was empty a second pass jumps to min(far) + T0.
    // Every thread derives the same thresholds (same float ops on the same published values).
    // The promotion appends through counts[1], NOT counts[0]: every thread has just read
    // counts[0] (== 0) to leave the round loop, in this same barrier window. Rule: a word
    // that decides control flow is never written, not even atomically, in the window where it
    // is read. Thread 0 publishes the results (pub[]) after a barrier.
    float lo = T, hi = T * growth;
    int more = 0;
    for (int attempt = 0; attempt < 2; attempt++) {
      for (long long v = tid; v < n; v += nthreads) {
        float d = BROOK_VLOAD_F(&dist[v]);
        if (!(d > lo) || !(d < BROOK_INF)) continue;
        if (d <= hi) {
          if (atomicExch(&inq[v], 1) == 0) cur[atomicAdd(&counts[1], 1)] = (int)v;
        } else {
          int bits = __float_as_int(d);
          if (__ldcg(minfar) > bits) atomicMin(minfar, bits);
        }
      }
      __threadfence(); grid.sync();
      if (tid == 0) {
        int promoted = BROOK_VLOAD_I(&counts[1]);
        pub[0] = promoted; pub[1] = BROOK_VLOAD_I(minfar);
        counts[0] = promoted; counts[1] = 0; counts[5] += 1;
        minfar[0] = 0x7f800000;
      }
      __threadfence(); grid.sync();
      int promoted = BROOK_VLOAD_I(&pub[0]), mf = BROOK_VLOAD_I(&pub[1]);
      if (promoted != 0) { more = 1; break; }
      if (mf == 0x7f800000) break;                               // nothing deferred: converged
      lo = hi; hi = __int_as_float(mf) + t0;                     // empty band: jump to the next voxel
    }
    if (!more) break;
    if (tid == 0) tbuf[1] = hi;
    __threadfence(); grid.sync();
  }
  if (tid == 0) { counts[2] = rounds; counts[3] = (BROOK_VLOAD_I(&counts[0]) != 0) ? 1 : 0; }
}
