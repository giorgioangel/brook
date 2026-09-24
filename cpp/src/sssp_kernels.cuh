// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "common.cuh"
#include <cooperative_groups.h>
namespace brook::cuda {
namespace cg=cooperative_groups;

#define BROOK_INF __int_as_float(0x7f800000)

// --- free-space seeding for soma DAF (matches dijkstra3d edf_free_space) ---
// Foreground voxels within safe_radius of source get the closed-form minimum
// 26-connected free-space path length; the geodesic relaxation then continues.
__global__ void edf_freespace_seed(
    float* dist, const unsigned char* field, long long source,
    long long sx, long long sy, long long sz,
    float wx, float wy, float wz, float safe_radius)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long n = sx * sy * sz;
  if (v >= n || field[v] == 0) return;
  long long x, y, z; brook_unravel(v, sx, sy, x, y, z);
  long long sxx, syy, szz; brook_unravel(source, sx, sy, sxx, syy, szz);
  float rdx = wx * (float)(x - sxx), rdy = wy * (float)(y - syy), rdz = wz * (float)(z - szz);
  if (rdx * rdx + rdy * rdy + rdz * rdz > safe_radius * safe_radius) return;
  float dx = fabsf((float)(x - sxx)), dy = fabsf((float)(y - syy)), dz = fabsf((float)(z - szz));
  float dxyz = fminf(fminf(dx, dy), dz);
  float dxy = fminf(dx, dy), dyz = fminf(dy, dz), dxz = fminf(dx, dz);
  float val = dxyz * sqrtf(wx * wx + wy * wy + wz * wz)
            + wx * (dx - dxyz) + wy * (dy - dxyz) + wz * (dz - dxyz)
            + (dxy - dxyz) * (sqrtf(wx * wx + wy * wy) - wx - wy)
            + (dxz - dxyz) * (sqrtf(wx * wx + wz * wz) - wx - wz)
            + (dyz - dxyz) * (sqrtf(wy * wy + wz * wz) - wy - wz);
  dist[v] = val;
}

// ---- frontier (worklist) Bellman-Ford (brook_atomic_min_f is in common.cuh) ----
// One frontier expansion round (non-cooperative fallback). mode 0 = geometric DAF (edge weight W[k],
// foreground-gated); mode 1 = vertex-weighted (cost = vwfield[v]).
__global__ void sssp_frontier(
    float* dist, const unsigned char* fg, const float* vwfield, const float* W,
    const unsigned int* vg, int has_vg, int mode, int rail_terminal, float cap,
    long long sx, long long sy, long long sz,
    const int* frontier, int fcount,
    int* nxt, int* ncount, int* inq)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= fcount) return;
  long long u = (long long)frontier[i];   // int32 flat index (component <= 2^31)
  // Clear the flag ATOMICALLY and read dist[u] afterwards with a cache-bypassing load: a
  // neighbour improving dist[u] tests this flag with an atomic, and a plain store is not
  // ordered against it -- the neighbour could still see the flag set (skip the re-enqueue)
  // after this thread had already read the old distance: a lost improvement, i.e. a flood
  // that terminates unconverged.
  int was = atomicExch(&inq[u], 0);
  if (mode == 1 && rail_terminal && vwfield[u] == 0.0f) return;  // rails absorb
  // data-dependent on the atomic's RESULT: an atomic whose result is unused does not hold
  // the thread, so the load could be served before the clear took effect.
  float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);   // was is 0/1: offset 0
  long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
  for (int k = 0; k < 26; k++) {
    long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    long long v = BROOK_IDX(nx, ny, nz, sx, sy);
    if (mode == 0 && fg[v] == 0) continue;          // DAF: foreground only
    if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;   // push: gate vg[u] dir k
    float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
    float old = brook_atomic_min_f(&dist[v], cand);
    // cap gates propagation only (atomicMin still records dist): the early-exit
    // railroad raises cap until the nearest rail settles. cand <= cap keeps the
    // frontier inside the distance horizon; cap = +inf reproduces full BF.
    if (old > cand && cand <= cap) {
      if (atomicExch(&inq[v], 1) == 0) {            // append once per round
        int pos = atomicAdd(ncount, 1);
        nxt[pos] = (int)v;
      }
    }
  }
}

// Deterministic predecessor assignment for the vertex-weighted tree:
// parent[v] = argmin_u dist[u] over in-bounds neighbours; on ties the FIRST neighbour in the
// BROOK_N* table order (faces -x +x -y +y -z +z, then edges, then corners). Stored as flat
// index, or -1 (source / unreachable). Run after dist converges.
// Exact float32 ties are common (kimimaro's DAF tie-break term is below the ulp of the
// accumulated distance): faces-first picks the shortest physical step, where dijkstra3d keeps
// whichever tied predecessor its heap settled first. Same distance field, equally optimal
// trees; faces-first is a deterministic tie-break.
__global__ void vw_assign_parents(
    const float* dist, const float* field, const unsigned int* vg, int has_vg,
    long long* parent,
    long long sx, long long sy, long long sz, long long source, int rail_terminal)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long n = sx * sy * sz;
  if (v >= n) return;
  if (v == source || dist[v] >= BROOK_INF) { parent[v] = -1; return; }
  long long x, y, z; brook_unravel(v, sx, sy, x, y, z);
  float mindu = BROOK_INF; long long argu = -1;
  for (int k = 0; k < 26; k++) {
    long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    long long u = BROOK_IDX(nx, ny, nz, sx, sy);
    if (rail_terminal && field[u] == 0.0f) continue;
    if (!brook_vg_ok(vg, has_vg, u, k)) continue;
    float du = dist[u];
    if (du < mindu) { mindu = du; argu = u; }   // strict < => first k in table order on ties
  }
  parent[v] = argu;
}

// Single-thread backtrace: walk parent[] from target to source, write the path
// (source -> target order) into out (flat indices); returns length in out_len.
template<class Index>
__global__ void backtrace(
    const long long* parent, long long target, long long source,
    Index* out, int* out_len, int maxlen)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  // first count / collect target..source
  long long cur = target;
  int cnt = 0;
  while (cnt < maxlen) {
    out[cnt++] = cur;
    if (cur == source) break;
    long long p = parent[cur];
    if (p < 0) break;     // broken chain
    cur = p;
  }
  // reverse in place -> source..target
  for (int i = 0; i < cnt / 2; i++) {
    Index t = out[i]; out[i] = out[cnt - 1 - i]; out[cnt - 1 - i] = t;
  }
  *out_len = cnt;
}

// Backtrace straight from the distance field (no O(n) parent pass): from `entry`, step to
// the predecessor = argmin neighbour dist (skip rails when rail_terminal, voxel-graph gated),
// until `target` (the flood source, dist 0); out = target..entry.
// Warp-parallel: launched as ONE block of 32 threads (one warp). Lane k
// probes neighbour k of the current voxel; a synchronous min-reduction picks the smallest
// distance and then the lowest k among ties -- exactly the scalar loop's ascending-k strict-<
// rule -- so the path is identical. Coordinates are tracked incrementally (no 64-bit divisions
// per step).
// The join (rail_terminal): the entry fixes the cost D and the last road voxel u (its first
// step), but every rail u may push to has distance D too (entering a rail costs 0). The path
// joins the one dijkstra3d's railroad joins: u's first rail neighbour in dijkstra3d order under
// the push gate (BROOK_D3RANK). The walk goes on from u; it never takes an argmin at that rail.
__global__ void backtrace_by_dist_warp(
    const float* dist, const float* field, const unsigned int* vg, int has_vg,
    long long entry, long long target, int rail_terminal,
    long long sx, long long sy, long long sz,
    int* out, int* out_len, int maxlen)
{
  int k = threadIdx.x;                                  // lane = neighbour index (0..25 used)
  long long cur = entry;
  long long x, y, z; brook_unravel(cur, sx, sy, x, y, z);
  int cnt = 0;
  while (cnt < maxlen) {                                // uniform across lanes (cur, cnt are)
    if (k == 0) out[cnt] = (int)cur;
    cnt++;
    if (cur == target) break;
    float du = BROOK_INF;
    if (k < 26) {
      long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
      if (nx >= 0 && ny >= 0 && nz >= 0 && nx < sx && ny < sy && nz < sz) {
        long long u = BROOK_IDX(nx, ny, nz, sx, sy);
        if (!(rail_terminal && field[u] == 0.0f) && brook_vg_ok(vg, has_vg, u, k)) du = dist[u];
      }
    }
    // distances are >= 0 (or +inf for inactive lanes), so their bit patterns order like the
    // floats; the reduction takes (uniform mask, int value)
    int db = __float_as_int(du);
    int mb = __reduce_min_sync(0xFFFFFFFFu, db);
    if (mb >= 0x7f800000) break;                        // no finite neighbour: broken chain
    int kb = __reduce_min_sync(0xFFFFFFFFu, (db == mb) ? k : 64);   // lowest k among the minima
    x += BROOK_NX[kb]; y += BROOK_NY[kb]; z += BROOK_NZ[kb];
    cur = BROOK_IDX(x, y, z, sx, sy);
    if (rail_terminal && cnt == 1) {                    // cur = u: re-pick the join (see above)
      int key = 64;
      if (k < 26) {
        long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
        if (nx >= 0 && ny >= 0 && nz >= 0 && nx < sx && ny < sy && nz < sz &&
            field[BROOK_IDX(nx, ny, nz, sx, sy)] == 0.0f && brook_vg_ok_push(vg, has_vg, cur, k))
          key = BROOK_D3RANK[k];
      }
      int r = __reduce_min_sync(0xFFFFFFFFu, key);      // < 26: the entry itself qualifies
      if (r < 26 && k == 0) {
        int ks = BROOK_D3INV[r];
        out[0] = (int)BROOK_IDX(x + BROOK_NX[ks], y + BROOK_NY[ks], z + BROOK_NZ[ks], sx, sy);
      }
    }
  }
  if (k == 0) {
    for (int i = 0; i < cnt / 2; i++) {
      int t = out[i]; out[i] = out[cnt - 1 - i]; out[cnt - 1 - i] = t;
    }
    *out_len = cnt;
  }
}


#define BROOK_INF __int_as_float(0x7f800000)

__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) sssp_coop(
    float* dist, const unsigned char* fg, const float* vwfield, const float* W,
    const unsigned int* vg, int has_vg, int mode, int rail_terminal, float cap,
    long long sx, long long sy, long long sz,
    int* wl_a, int* wl_b, int* counts, int* inq, int fcount0, int max_rounds)
{
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  if (tid == 0) { counts[0] = fcount0; counts[1] = 0; }
  __threadfence(); grid.sync();

  int* cur = wl_a; int* nxt = wl_b;
  int round = 0;
  for (; round < max_rounds; round++) {
    int fcount = BROOK_VLOAD_I(&counts[0]);
    if (fcount == 0) break;
    for (int i = tid; i < fcount; i += nthreads) {
      long long u = (long long)BROOK_VLOAD_I(&cur[i]);
      // atomic clear, THEN read dist (see sssp_frontier). The read is made data-dependent
      // on the atomic's RESULT: an atomic whose result is unused does not block the thread,
      // so the load could be served before the clear took effect (same lost wakeup).
      int was = atomicExch(&inq[u], 0);
      if (mode == 1 && rail_terminal && vwfield[u] == 0.0f) continue;  // rails absorb
      float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);   // was is 0/1: offset 0
      long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
      for (int k = 0; k < 26; k++) {
        long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
        long long v = BROOK_IDX(nx, ny, nz, sx, sy);
        if (mode == 0 && fg[v] == 0) continue;
        if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
        float cand = (mode == 0) ? (du + W[k]) : brook_vw_step(du, vwfield[v]);
        float old = brook_atomic_min_f(&dist[v], cand);
        if (old > cand && cand <= cap) {     // cap gates propagation (early-exit railroad)
          if (atomicExch(&inq[v], 1) == 0) {
            int pos = atomicAdd(&counts[1], 1);
            nxt[pos] = (int)v;
          }
        }
      }
    }
    __threadfence(); grid.sync();                        // all relaxations + appends complete
    if (tid == 0) { counts[0] = BROOK_VLOAD_I(&counts[1]); counts[1] = 0; }
    int* tmp = cur; cur = nxt; nxt = tmp;
    __threadfence(); grid.sync();                        // new counts + swapped buffers visible
  }
  // report: rounds run, and whether the round budget ran out with work left (the host
  // raises instead of returning an unconverged field, as the host loop does)
  if (tid == 0) { counts[2] = round; counts[3] = (BROOK_VLOAD_I(&counts[0]) != 0) ? 1 : 0; }
}


#define BROOK_INF __int_as_float(0x7f800000)

// Front-scaled bands. min(far) is tracked as a packed key (distance bits << 32 | the far
// voxel's own field bits), so the phase end knows the deferred voxel nearest to the source AND
// what it costs to step there. The next band is then
//     T = d_min + clamp(front_factor * f_min, min_rel * d_min, delta)
// instead of d_min + delta. delta is scaled to the TARGET's penalty (a thin neurite: ~1e5 per
// step); when such a flood reaches a thick region -- a soma body, PDRF ~ 1 -- one delta-wide
// band swallows the whole region and it is relaxed label-correcting style, with many repeated
// relaxations. min_rel bounds the number of phases
// geometrically whatever the field does. Any band policy yields the same fixed point;
// front_factor <= 0 selects the fixed delta.
#define BROOK_MINFAR_NONE 0x7f8000007f800000ULL
__device__ __forceinline__ void brook_minfar(unsigned long long* minfar, float d, float f) {
  unsigned long long key = ((unsigned long long)(unsigned int)__float_as_int(d) << 32)
                           | (unsigned long long)(unsigned int)__float_as_int(f);
  if (__ldcg(minfar) > key) atomicMin(minfar, key);
}
__device__ __forceinline__ float brook_next_T(unsigned long long mf, float delta, float front_factor,
                                              float min_rel) {
  float d = __int_as_float((int)(mf >> 32));
  if (front_factor <= 0.0f || mf == BROOK_MINFAR_NONE) return d + delta;
  float f = __int_as_float((int)(mf & 0xFFFFFFFFULL));
  float w = front_factor * f, lo = min_rel * d;
  if (!(w > lo)) w = lo;
  if (!(w < delta)) w = delta;
  return d + w;
}


// Expand one frontier voxel u of the near band: relax its 26 neighbours, route
// improved voxels to the next near list (cand <= Tb) or the far list, capture
// rails, track min(far) and the touched list. Shared by the grid-wide rounds and
// the single-block thin-frontier rounds.
__device__ __forceinline__ void dstep_expand(
    int u32, float Tb, float* dist, const float* field, const unsigned int* vg, int has_vg,
    long long sx, long long sy, long long sz,
    int* inq, int* infar, int* nnxt, int* fcur, int* counts, unsigned long long* minfar_bits,
    unsigned long long* rail_best, int* touched, int do_stats, unsigned long long* stats)
{
  long long u = (long long)u32;
  int was = atomicExch(&inq[u], 0);                // atomic clear, THEN read dist (see sssp_frontier)
  if (field[u] == 0.0f) return;                    // rail absorbs (terminal)
  float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);   // read ordered after the clear
  if (du > Tb) return;                             // beyond the rail bound: cannot matter
  long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
  for (int k = 0; k < 26; k++) {
    long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    long long v = BROOK_IDX(nx, ny, nz, sx, sy);
    if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
    float cand = brook_vw_step(du, field[v]);
    float old = brook_atomic_min_f(&dist[v], cand);
    if (old > cand) {
      if (old >= BROOK_INF) touched[atomicAdd(&counts[5], 1)] = (int)v;   // first write
      if (do_stats) { atomicAdd(stats, 1ULL); if (old >= BROOK_INF) atomicAdd(stats + 1, 1ULL); }
      if (field[v] == 0.0f) {                      // capture nearest rail: fixes D and u (the join
                                                   // itself is re-picked by the backtrace)
        unsigned long long key =
            ((unsigned long long)__float_as_int(cand) << 32) | (unsigned int)v;
        atomicMin(rail_best, key);
      }
      if (cand <= Tb) {
        if (atomicExch(&inq[v], 1) == 0) nnxt[atomicAdd(&counts[1], 1)] = (int)v;
      } else {
        brook_minfar(minfar_bits, cand, field[v]);      // incremental min(far)
        if (atomicExch(&infar[v], 1) == 0) fcur[atomicAdd(&counts[2], 1)] = (int)v;
      }
    }
  }
}

__device__ __forceinline__ float dstep_bound(float T, const unsigned long long* rail_best) {
  // effective band bound = min(T, best rail distance so far); see header comment
  unsigned long long rb = __ldcg(rail_best);
  if (rb != 0xFFFFFFFFFFFFFFFFULL) T = fminf(T, __int_as_float((int)(rb >> 32)));
  return T;
}

__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) sssp_dstep(
    float* dist, const float* field, const unsigned int* vg, int has_vg,
    long long sx, long long sy, long long sz, float delta,
    int* near_a, int* near_b, int* far_a, int* far_b,
    int* inq, int* infar, int* counts, float* Tbuf, unsigned long long* minfar_bits,
    unsigned long long* rail_best, int* touched, long long source,
    int max_phases, int max_inner, int do_stats, unsigned long long* stats,
    int lane_max, float* sdu, long long* sxyz, float front_factor, float min_rel)
{
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  // counts: [0]=near_cur [1]=near_nxt [2]=far_cur [3]=far_nxt [4]=done [5]=touched
  // The workspace arrives clean (dist=inf, inq=infar=0 everywhere) and every voxel
  // this flood writes is appended to `touched` exactly once (the first improvement
  // from inf is seen by exactly one atomic), so dstep_reset can restore the clean
  // state in O(settled) afterwards instead of O(bbox) fills per path.
  if (tid == 0) {
    counts[0] = 1; counts[1] = 0; counts[2] = 0; counts[3] = 0; counts[4] = 0;
    counts[5] = 1;
    // delta = +inf: no cap (front-scaled bands only); the first band is scaled to the source
    Tbuf[0] = (delta < BROOK_INF) ? delta : fmaxf(front_factor * field[source], 1e-30f);
    rail_best[0] = 0xFFFFFFFFFFFFFFFFULL;
    minfar_bits[0] = BROOK_MINFAR_NONE;   // min(far) is tracked incrementally (see phase end)
    dist[source] = 0.0f; inq[source] = 1; near_a[0] = (int)source; touched[0] = (int)source;
  }
  __threadfence(); grid.sync();
  int* ncur = near_a; int* nnxt = near_b; int* fcur = far_a; int* fnxt = far_b;

  for (int phase = 0; phase < max_phases; phase++) {
    float T = BROOK_VLOAD_F(&Tbuf[0]);
    if (do_stats && tid == 0) atomicAdd(stats + 2, 1ULL);          // phases
    // ---- settle the near band [.., T] to convergence ----
    for (int inner = 0; inner < max_inner; inner++) {
      int nc = BROOK_VLOAD_I(&counts[0]);
      if (nc == 0) break;
      if (do_stats && tid == 0) atomicAdd(stats + 3, 1ULL);        // inner rounds
      float Tb = dstep_bound(T, rail_best);
      if (nc <= lane_max) {
        // thin front: one lane per neighbour (lane_max). Step A keeps the ordered part
        // (atomic flag clear, THEN the distance read) and parks distance + coordinates; step B
        // relaxes neighbour k of voxel i in thread 32*i + k with dstep_expand's body.
        for (int i = tid; i < nc; i += nthreads) {
          long long u = (long long)BROOK_VLOAD_I(&ncur[i]);
          int was = atomicExch(&inq[u], 0);
          float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);
          if (field[u] == 0.0f || du > Tb) { sdu[i] = -1.0f; continue; }
          long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
          sdu[i] = du; sxyz[i] = x | (y << 21) | (z << 42);
        }
        __threadfence(); grid.sync();
        long long items = (long long)nc << 5;
        for (long long j = tid; j < items; j += nthreads) {
          int k = (int)(j & 31);
          if (k >= 26) continue;
          int i = (int)(j >> 5);
          float du = BROOK_VLOAD_F(&sdu[i]);
          if (du < 0.0f) continue;
          long long p = (long long)__ldcg(&sxyz[i]);
          long long nx = (p & 0x1FFFFF) + BROOK_NX[k], ny = ((p >> 21) & 0x1FFFFF) + BROOK_NY[k],
                    nz = (p >> 42) + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long u = (long long)BROOK_VLOAD_I(&ncur[i]);
          long long v = BROOK_IDX(nx, ny, nz, sx, sy);
          if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
          float cand = brook_vw_step(du, field[v]);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand) {
            if (old >= BROOK_INF) touched[atomicAdd(&counts[5], 1)] = (int)v;
            if (do_stats) { atomicAdd(stats, 1ULL); if (old >= BROOK_INF) atomicAdd(stats + 1, 1ULL); }
            if (field[v] == 0.0f) {                  // nearest rail (as dstep_expand)
              unsigned long long key =
                  ((unsigned long long)__float_as_int(cand) << 32) | (unsigned int)v;
              atomicMin(rail_best, key);
            }
            if (cand <= Tb) {
              if (atomicExch(&inq[v], 1) == 0) nnxt[atomicAdd(&counts[1], 1)] = (int)v;
            } else {
              brook_minfar(minfar_bits, cand, field[v]);
              if (atomicExch(&infar[v], 1) == 0) fcur[atomicAdd(&counts[2], 1)] = (int)v;
            }
          }
        }
      } else
      for (int i = tid; i < nc; i += nthreads)
        dstep_expand(BROOK_VLOAD_I(&ncur[i]), Tb, dist, field, vg, has_vg, sx, sy, sz, inq, infar,
                     nnxt, fcur, counts, minfar_bits, rail_best, touched, do_stats, stats);
      __threadfence(); grid.sync();
      if (tid == 0) { counts[0] = BROOK_VLOAD_I(&counts[1]); counts[1] = 0; }
      int* t = ncur; ncur = nnxt; nnxt = t;
      __threadfence(); grid.sync();
    }
    // ---- phase end: three grid barriers. min(far) has been tracked
    // incrementally (atomicMin on every far-side improvement, and on the kept entries
    // below), so no reduction pass; tid 0 does the rail check and advances T in one go.
    if (tid == 0) {
      counts[4] = 0;
      unsigned long long rb = BROOK_VLOAD_U64(rail_best);
      if (rb != 0xFFFFFFFFFFFFFFFFULL &&
          __int_as_float((int)(rb >> 32)) <= T) counts[4] = 1;   // nearest rail is final
      Tbuf[0] = brook_next_T(BROOK_VLOAD_U64(minfar_bits), delta, front_factor, min_rel);
      minfar_bits[0] = BROOK_MINFAR_NONE;                   // re-armed for the next phase
    }
    __threadfence(); grid.sync();                                                                     // (1)
    if (BROOK_VLOAD_I(&counts[4])) break;
    float Tnew = BROOK_VLOAD_F(&Tbuf[0]);
    // ---- promote far voxels with dist<=min(Tnew, rail bound) into the next near band ----
    float Tp = dstep_bound(Tnew, rail_best);
    int fc = BROOK_VLOAD_I(&counts[2]);
    for (int i = tid; i < fc; i += nthreads) {
      long long w = (long long)BROOK_VLOAD_I(&fcur[i]);
      if (BROOK_VLOAD_I(&infar[w]) == 0) continue;                       // already pulled in via relaxation
      float dw = BROOK_VLOAD_F(&dist[w]);
      if (dw <= Tp) {
        infar[w] = 0;
        if (atomicExch(&inq[w], 1) == 0) ncur[atomicAdd(&counts[0], 1)] = (int)w;
      } else {
        brook_minfar(minfar_bits, dw, field[w]);              // kept: feeds the next min(far)
        fnxt[atomicAdd(&counts[3], 1)] = (int)w;
      }
    }
    __threadfence(); grid.sync();                                                                     // (2)
    if (tid == 0) { counts[2] = BROOK_VLOAD_I(&counts[3]); counts[3] = 0; }
    int* t = fcur; fcur = fnxt; fnxt = t;
    __threadfence(); grid.sync();                                                                     // (3)
    if (BROOK_VLOAD_I(&counts[0]) == 0 && BROOK_VLOAD_I(&counts[2]) == 0) break;         // nothing reachable left
  }
}

// Restore the workspace's clean state over the voxels the last flood touched.
__global__ void dstep_reset(
    float* dist, int* inq, int* infar, const int* touched, const int* counts)
{
  int n = counts[5];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int v = touched[i];
    dist[v] = BROOK_INF; inq[v] = 0; infar[v] = 0;
  }
}

}
