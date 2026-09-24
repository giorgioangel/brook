// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti

#define BROOK_INF __int_as_float(0x7f800000)
#define BROOK_MAXKEY 0xFFFFFFFFFFFFFFFFULL
#define BROOK_DRAFT_WIN 4096
// Flags: one int32 word per voxel. A label's invalidation and its next railroad flood run at the
// same time on the same voxels (pipelined speculation), so each owns bits and touches them with
// bit-atomics only -- never a whole-word exchange or a plain store (atomicOr / atomicAnd return
// the old word). The K batch's marks use the whole word: they
// exist only on the crit lane's voxels while all of those voxels' bits are clear. Bits >= 8 are
// never set, so `was >> 8` keeps the data dependence of ordering rule 1 without being provably 0.
#define BROOK_F_DQ 1      // railroad: in the near queue
#define BROOK_F_DF 2      // railroad: on the far list
#define BROOK_F_IQ 4      // invalidation: in the frontier
#define BROOK_F_IW 8      // invalidation: waiting for its band
#define BROOK_F_IS 16     // invalidation: claimed, and its key improved after it may have offered (stale offers out there)
#define BROOK_F_ID 32     // invalidation: waiting, and its recorded offer may be stale: needs the exact pull

// Arenas. The lockstep state is one flat index space made of arenas laid end to end: arena 0 is
// the whole volume, every further arena is the private bbox crop of a soma candidate (its hole
// filling changes which voxels belong to it, so it cannot live in the shared volume). A voxel's
// neighbours are computed in its arena's own coordinates and never leave it.
// aoff[a] = first global index of arena a, adim[3a..3a+2] = its (sx, sy, sz).
// Draft windows (body "k") are arenas too, all cubes of one power-of-two size laid after the dense
// arenas: `na` carries the dense count in its low 16 bits and log2(window voxels) above them, so
// that a window's arena is found by a shift, not by a scan over hundreds of arenas.
__device__ __forceinline__ int lock_arena(long long u, const long long* aoff, int na) {
  int nd = na & 0xFFFF, sh = na >> 16;
  if (sh != 0 && u >= aoff[nd]) return nd + (int)((u - aoff[nd]) >> sh);
  int a = 0;
  while (a + 1 < nd && u >= aoff[a + 1]) a++;
  return a;
}

// Front-scaled bands: see dstep (FRONT-SCALED BANDS) in sssp_kernels.cuh. Same packed key and policy,
// per label.
#define BROOK_MINFAR_NONE 0x7f8000007f800000ULL
__device__ __forceinline__ void brook_minfar(unsigned long long* minfar, float d, float f) {
  unsigned long long key = ((unsigned long long)(unsigned int)__float_as_int(d) << 32)
                           | (unsigned long long)(unsigned int)__float_as_int(f);
  if (__ldcg(minfar) > key) atomicMin(minfar, key);
}
__device__ __forceinline__ float brook_next_T(unsigned long long mf, float delta, float front_factor,
                                              float min_rel) {
  float d = __int_as_float((int)(mf >> 32));
  if (front_factor <= 0.0f) return d + delta;
  float f = __int_as_float((int)(mf & 0xFFFFFFFFULL));
  float w = front_factor * f, lo = min_rel * d;
  if (!(w > lo)) w = lo;
  if (!(w < delta)) w = delta;
  return d + w;
}

__device__ __forceinline__ float lock_bound(float T, const unsigned long long* rail_best) {
  unsigned long long rb = __ldcg(rail_best);             // stale (larger) reads cost work only
  if (rb != BROOK_MAXKEY) T = fminf(T, __int_as_float((int)(rb >> 32)));
  return T;
}

// Near-far delta-stepping railroad flood of every active label at once. Same per-label
// algorithm as the single-label dstep (rail-bounded band, incremental min(far), rail capture with the
// packed lowest-flat tie-break, which fixes the cost and the last road voxel u: backtrace_dist_lab
// re-picks the join among u's rails); T / delta / rail_best / minfar / done are per-label tables and
// the phase end is global. Ordering rules of every brook flood kernel apply (atomic flag
// clear, distance read data-dependent on it, fenced barriers, cache-bypassing loads).
// A voxel graph (vg, has_vg) gates every relaxation u -> v as sssp_dstep does (push: bit
// BROOK_BIT[k] of vg[u]); a window copy of the graph gates a window exactly like its object.
// counts: [0] near_cur [1] near_nxt [2] far_cur [3] far_nxt [5] touched [6] phases [7] rounds
// done: 0 flooding, 1 rail settled, 2 nothing reachable (caller ejects the label)
__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) dstep_lab(
    float* dist, const float* field, const LABEL_T* lab, const unsigned int* vg, int has_vg,
    const long long* aoff, const long long* adim, int na,
    const int* act, const int* tgt, const int* ih,
    float* T, float* delta, unsigned long long* rail_best, unsigned long long* minfar, int* done,
    int* near_a, int* near_b, int* far_a, int* far_b, int* inq, int* infar,
    int* counts, int* touched, int max_phases, int max_inner,
    int lane_max, float* sdu, float* stb, long long* sxyz, int* sar,
    float front_factor, float min_rel)
{
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  int nact = ih[0];
  if (tid == 0) for (int j = 0; j < 8; j++) counts[j] = 0;
  __threadfence(); grid.sync();

  // ---- seed: one source per label ----
  for (int i = tid; i < nact; i += nthreads) {
    int L = act[i]; int t = tgt[i];
    rail_best[L] = BROOK_MAXKEY; minfar[L] = BROOK_MINFAR_NONE; done[L] = 1;
    if (t < 0) continue;
    float ft = field[t];
    dist[t] = 0.0f;
    touched[atomicAdd(&counts[5], 1)] = t;
    if (ft == 0.0f) { rail_best[L] = (unsigned long long)(unsigned int)t; continue; }  // already a rail
    delta[L] = BROOK_INF;                              // no cap on the band (see sssp, band policy)
    T[L] = fmaxf(front_factor * ft, 1e-30f);           // first band: scaled to the source's own cost
    done[L] = 0;
    atomicOr(&inq[t], BROOK_F_DQ);
    near_a[atomicAdd(&counts[0], 1)] = t;
  }
  __threadfence(); grid.sync();

  int* ncur = near_a; int* nnxt = near_b; int* fcur = far_a; int* fnxt = far_b;
  for (int phase = 0; phase < max_phases; phase++) {
    // ---- settle every label's near band to convergence ----
    for (int inner = 0; inner < max_inner; inner++) {
      int nc = BROOK_VLOAD_I(&counts[0]);
      if (nc == 0) break;
      if (nc <= (nthreads >> 5)) {
        // One warp per voxel: the thin front fits one voxel per warp, so each voxel is expanded
        // entirely inside its warp -- the single-thread expansion below with the 26 neighbours on
        // lanes. Lane 0 does what must stay ordered (atomic flag clear, THEN the distance read) and
        // shuffles distance, bound and position to the other lanes; lane k relaxes neighbour k.
        // Against the two-step lanes below this drops the side arrays' store/load round trip, the
        // reloads of the front entry, label and arena dimensions, and one grid barrier (with its
        // fence) per round. Same relaxations, same fixed point; nothing couples two voxels'
        // expansions within a round, as in the single-thread path.
        const int lane = threadIdx.x & 31;
        for (int i = tid >> 5; i < nc; i += nthreads >> 5) {
          long long u = 0, a0 = 0, sx = 0, sy = 0, sz = 0, x = 0, y = 0, z = 0;
          float du = -1.0f, Tb = 0.0f;
          unsigned long long lu = 0;
          if (lane == 0) {
            u = (long long)BROOK_VLOAD_I(&ncur[i]);
            int was = atomicAnd(&inq[u], ~BROOK_F_DQ);
            lu = (unsigned long long)lab[u];                 // read-only: issued beside the atomic
            int L = (int)lu;
            float fu = field[u];
            int a = lock_arena(u, aoff, na);
            a0 = aoff[a]; sx = adim[3 * a]; sy = adim[3 * a + 1]; sz = adim[3 * a + 2];
            float d = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);
            float T0 = lock_bound(BROOK_VLOAD_F(&T[L]), &rail_best[L]);
            if (fu != 0.0f && d <= T0) { du = d; Tb = T0; brook_unravel(u - a0, sx, sy, x, y, z); }
          }
          du = __shfl_sync(0xffffffffu, du, 0);
          if (du < 0.0f) continue;                           // warp-uniform: rail, or beyond the bound
          u = __shfl_sync(0xffffffffu, u, 0);
          Tb = __shfl_sync(0xffffffffu, Tb, 0);
          lu = __shfl_sync(0xffffffffu, lu, 0);
          const int L = (int)lu;
          a0 = __shfl_sync(0xffffffffu, a0, 0);
          sx = __shfl_sync(0xffffffffu, sx, 0); sy = __shfl_sync(0xffffffffu, sy, 0); sz = __shfl_sync(0xffffffffu, sz, 0);
          x = __shfl_sync(0xffffffffu, x, 0); y = __shfl_sync(0xffffffffu, y, 0); z = __shfl_sync(0xffffffffu, z, 0);
          if (lane >= 26) continue;
          long long nx = x + BROOK_NX[lane], ny = y + BROOK_NY[lane], nz = z + BROOK_NZ[lane];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long v = a0 + BROOK_IDX(nx, ny, nz, sx, sy);
          LABEL_T lv = lab[v];                               // read-only: issued together
          float fv = field[v];
          if ((unsigned long long)lv != lu) continue;
          if (!brook_vg_ok_push(vg, has_vg, u, lane)) continue;
          float cand = brook_vw_step(du, fv);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand) {
            if (old >= BROOK_INF) touched[atomicAdd(&counts[5], 1)] = (int)v;
            if (fv == 0.0f) {                              // the nearest rail (fixes D and u)
              unsigned long long key =
                  ((unsigned long long)__float_as_int(cand) << 32) | (unsigned int)v;
              atomicMin(&rail_best[L], key);
            }
            if (cand <= Tb) {
              if ((atomicOr(&inq[v], BROOK_F_DQ) & BROOK_F_DQ) == 0) nnxt[atomicAdd(&counts[1], 1)] = (int)v;
            } else {
              brook_minfar(&minfar[L], cand, fv);
              if ((atomicOr(&infar[v], BROOK_F_DF) & BROOK_F_DF) == 0) fcur[atomicAdd(&counts[2], 1)] = (int)v;
            }
          }
        }
      } else if (nc <= lane_max) {
        // Thin front: a round is bound by the latency of expanding ONE voxel (26 neighbours in
        // sequence, each a chain of dependent random loads and atomics: ~6 us, against ~0.7 us
        // for a barrier). So the neighbours get a lane each. Step A (one thread per voxel)
        // does what must stay ordered -- clear the flag, THEN read the distance -- and leaves
        // the voxel's distance / bound / coordinates in side arrays; step B relaxes neighbour
        // k of voxel i in thread 32*i + k, i.e. one warp per voxel. Same relaxations, same
        // fixed point; an improvement of dist[u] after the snapshot finds inq[u] == 0 and
        // re-enqueues u, exactly as in the single-thread expansion.
        for (int i = tid; i < nc; i += nthreads) {
          long long u = (long long)BROOK_VLOAD_I(&ncur[i]);
          int was = atomicAnd(&inq[u], ~BROOK_F_DQ);
          float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);
          int L = (int)lab[u];
          float Tb = lock_bound(BROOK_VLOAD_F(&T[L]), &rail_best[L]);
          if (field[u] == 0.0f || du > Tb) { sdu[i] = -1.0f; continue; }
          int a = lock_arena(u, aoff, na);
          long long x, y, z; brook_unravel(u - aoff[a], adim[3 * a], adim[3 * a + 1], x, y, z);
          sdu[i] = du; stb[i] = Tb; sxyz[i] = x | (y << 21) | (z << 42); sar[i] = a;
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
          int a = BROOK_VLOAD_I(&sar[i]);
          long long sx = adim[3 * a], sy = adim[3 * a + 1], sz = adim[3 * a + 2];
          long long nx = (p & 0x1FFFFF) + BROOK_NX[k], ny = ((p >> 21) & 0x1FFFFF) + BROOK_NY[k],
                    nz = (p >> 42) + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long u = (long long)BROOK_VLOAD_I(&ncur[i]);
          long long v = aoff[a] + BROOK_IDX(nx, ny, nz, sx, sy);
          LABEL_T lu = lab[u]; int L = (int)lu;
          if (lab[v] != lu) continue;
          if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
          float Tb = BROOK_VLOAD_F(&stb[i]);
          float cand = brook_vw_step(du, field[v]);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand) {
            if (old >= BROOK_INF) touched[atomicAdd(&counts[5], 1)] = (int)v;
            if (field[v] == 0.0f) {                        // the nearest rail (fixes D and u)
              unsigned long long key =
                  ((unsigned long long)__float_as_int(cand) << 32) | (unsigned int)v;
              atomicMin(&rail_best[L], key);
            }
            if (cand <= Tb) {
              if ((atomicOr(&inq[v], BROOK_F_DQ) & BROOK_F_DQ) == 0) nnxt[atomicAdd(&counts[1], 1)] = (int)v;
            } else {
              brook_minfar(&minfar[L], cand, field[v]);
              if ((atomicOr(&infar[v], BROOK_F_DF) & BROOK_F_DF) == 0) fcur[atomicAdd(&counts[2], 1)] = (int)v;
            }
          }
        }
      } else
      for (int i = tid; i < nc; i += nthreads) {
        long long u = (long long)BROOK_VLOAD_I(&ncur[i]);
        int was = atomicAnd(&inq[u], ~BROOK_F_DQ);         // atomic clear, THEN read dist
        if (field[u] == 0.0f) continue;                    // rail absorbs (terminal)
        float du = BROOK_VLOAD_F(&dist[u + (long long)(was >> 8)]);
        LABEL_T lu = lab[u]; int L = (int)lu;
        float Tb = lock_bound(BROOK_VLOAD_F(&T[L]), &rail_best[L]);
        if (du > Tb) continue;                             // beyond the rail bound: cannot matter
        int a = lock_arena(u, aoff, na);
        long long a0 = aoff[a], sx = adim[3 * a], sy = adim[3 * a + 1], sz = adim[3 * a + 2];
        long long x, y, z; brook_unravel(u - a0, sx, sy, x, y, z);
        for (int k = 0; k < 26; k++) {
          long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long v = a0 + BROOK_IDX(nx, ny, nz, sx, sy);
          if (lab[v] != lu) continue;                      // never leave the component
          if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
          float cand = brook_vw_step(du, field[v]);
          float old = brook_atomic_min_f(&dist[v], cand);
          if (old > cand) {
            if (old >= BROOK_INF) touched[atomicAdd(&counts[5], 1)] = (int)v;   // first write
            if (field[v] == 0.0f) {                        // capture the nearest rail (fixes D and u)
              unsigned long long key =
                  ((unsigned long long)__float_as_int(cand) << 32) | (unsigned int)v;
              atomicMin(&rail_best[L], key);
            }
            if (cand <= Tb) {
              if ((atomicOr(&inq[v], BROOK_F_DQ) & BROOK_F_DQ) == 0) nnxt[atomicAdd(&counts[1], 1)] = (int)v;
            } else {
              brook_minfar(&minfar[L], cand, field[v]);
              if ((atomicOr(&infar[v], BROOK_F_DF) & BROOK_F_DF) == 0) fcur[atomicAdd(&counts[2], 1)] = (int)v;
            }
          }
        }
      }
      __threadfence(); grid.sync();
      if (tid == 0) { counts[0] = BROOK_VLOAD_I(&counts[1]); counts[1] = 0; counts[7] += 1; }
      int* t = ncur; ncur = nnxt; nnxt = t;
      __threadfence(); grid.sync();
    }
    // ---- phase end, per label: nearest rail final? else advance the band ----
    for (int i = tid; i < nact; i += nthreads) {
      int L = act[i];
      if (BROOK_VLOAD_I(&done[L]) != 0) continue;
      float Tl = BROOK_VLOAD_F(&T[L]);
      unsigned long long rb = BROOK_VLOAD_U64(&rail_best[L]);
      if (rb != BROOK_MAXKEY && __int_as_float((int)(rb >> 32)) <= Tl) { done[L] = 1; continue; }
      unsigned long long mf = BROOK_VLOAD_U64(&minfar[L]);
      if (mf == BROOK_MINFAR_NONE) { done[L] = 2; continue; }   // nothing reachable left, no rail
      T[L] = brook_next_T(mf, BROOK_VLOAD_F(&delta[L]), front_factor, min_rel);
      minfar[L] = BROOK_MINFAR_NONE;                        // re-armed for the next phase
    }
    __threadfence(); grid.sync();
    // ---- promote far voxels inside their label's new band; drop those of finished labels ----
    int fc = BROOK_VLOAD_I(&counts[2]);
    for (int i = tid; i < fc; i += nthreads) {
      long long w = (long long)BROOK_VLOAD_I(&fcur[i]);
      if ((BROOK_VLOAD_I(&infar[w]) & BROOK_F_DF) == 0) continue;
      int L = (int)lab[w];
      if (BROOK_VLOAD_I(&done[L]) != 0) continue;          // flags are restored by reset_touched
      float dw = BROOK_VLOAD_F(&dist[w]);
      float Tp = lock_bound(BROOK_VLOAD_F(&T[L]), &rail_best[L]);
      if (dw <= Tp) {
        atomicAnd(&infar[w], ~BROOK_F_DF);
        if ((atomicOr(&inq[w], BROOK_F_DQ) & BROOK_F_DQ) == 0) ncur[atomicAdd(&counts[0], 1)] = (int)w;
      } else {
        brook_minfar(&minfar[L], dw, field[w]);
        fnxt[atomicAdd(&counts[3], 1)] = (int)w;
      }
    }
    __threadfence(); grid.sync();
    if (tid == 0) { counts[2] = BROOK_VLOAD_I(&counts[3]); counts[3] = 0; counts[6] += 1; }
    int* t = fcur; fcur = fnxt; fnxt = t;
    __threadfence(); grid.sync();
    if (BROOK_VLOAD_I(&counts[0]) == 0 && BROOK_VLOAD_I(&counts[2]) == 0) break;
  }
}

// Is voxel v inside a straight-line ball of label L's pending path (the path accepted by the
// previous iteration, whose real invalidation is running right now)? Only a prediction.
__device__ __forceinline__ int lock_pending_dead(
    long long v, int L, const int* plast_off, const int* plast_len, const int* store, const float* DBF,
    const long long* aoff, const long long* adim, int na,
    float wx, float wy, float wz, float scale, float constant)
{
  int n = plast_len[L];
  if (n <= 0) return 0;
  int a = lock_arena(v, aoff, na);
  long long a0 = aoff[a], sx = adim[3 * a], sy = adim[3 * a + 1];
  long long x, y, z; brook_unravel(v - a0, sx, sy, x, y, z);
  const int* path = store + plast_off[L];
  for (int j = 0; j < n; j++) {
    long long p = (long long)path[j];
    long long px, py, pz; brook_unravel(p - a0, sx, sy, px, py, pz);
    float dx = wx * (float)(x - px), dy = wy * (float)(y - py), dz = wz * (float)(z - pz);
    float r = scale * DBF[p] + constant;
    if (dx * dx + dy * dy + dz * dz < r * r) return 1;
  }
  return 0;
}

// Target of every active label: the label's manual targets first (in pop order, by its own path
// counter), then the first still-alive voxel of its descending-DAF order at or after its cursor
// (alive only goes 1 -> 0, so the cursor only moves forward). -1: the label is finished
// (nothing alive, or its path budget is used up).
// An invalidation ball can kill a run of 10^5..10^7 consecutive entries, and a single thread
// walking it (dependent random loads) takes longer than the floods. So: a short serial probe
// per label settles the common case, then the unresolved labels are searched in parallel --
// every round all threads test a window of 2^shift entries per unresolved label and an
// atomicMin keeps the first alive position; windows grow 4x per round.
// spec = 1: the PREDICTION (see lock_offsets): voxels inside the pending path's balls count as
// dead, and the label's cursor is left alone (wcur[i] is this search's own position).
// spec = 2: a deliberately poor prediction (no ball test; tests of the rejection path).
// hdr: [0],[1] unresolved counts
__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) target_lab(
    const unsigned char* alive, const int* order, const long long* seg_end, long long* cursor,
    const int* act, const int* ih, const int* pcount,
    const int* mb_start, const int* mb_count, const int* mb_flat, const int* maxp,
    int* tgt, long long* wcur, int* un_a, int* un_b, unsigned long long* best, int* hdr, int max_rounds,
    int spec, const int* plast_off, const int* plast_len, const int* store, const float* DBF,
    const long long* aoff, const long long* adim, int na,
    float wx, float wy, float wz, float scale, float constant)
{
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  int nact = ih[0];
  if (tid == 0) { hdr[0] = 0; hdr[1] = 0; }
  __threadfence(); grid.sync();

  for (int i = tid; i < nact; i += nthreads) {
    int L = act[i];
    int iter = pcount[L];
    if (iter >= maxp[L]) { tgt[i] = -1; continue; }
    if (iter < mb_count[L]) { tgt[i] = mb_flat[mb_start[L] + iter]; continue; }
    long long c = cursor[L], e = seg_end[L];
    int pending = (spec == 1) && plast_len[L] > 0;
    if (!pending) {                                        // (the ball test is too long for a serial probe)
      long long stop = (c + 32 < e) ? c + 32 : e;
      while (c < stop && alive[order[c]] == 0) c++;        // serial probe
      if (!spec) cursor[L] = c;
      if (c < stop) { tgt[i] = order[c]; continue; }
      if (c >= e) { tgt[i] = -1; continue; }
    }
    wcur[i] = c;
    best[i] = BROOK_MAXKEY;
    un_a[atomicAdd(&hdr[0], 1)] = i;
  }
  __threadfence(); grid.sync();

  int* cur = un_a; int* nxt = un_b;
  int shift = 12;
  for (int round = 0; round < max_rounds; round++) {
    int nu = BROOK_VLOAD_I(&hdr[0]);
    if (nu == 0) break;
    long long W = 1LL << shift;
    long long items = (long long)nu << shift;
    for (long long j = tid; j < items; j += nthreads) {
      int i = BROOK_VLOAD_I(&cur[j >> shift]);
      int L = act[i];
      long long pos = (long long)BROOK_VLOAD_U64(&wcur[i]) + (j & (W - 1));
      if (pos >= seg_end[L]) continue;
      long long v = (long long)order[pos];
      if (alive[v] == 0) continue;
      if (__ldcg(&best[i]) <= (unsigned long long)pos) continue;      // cheap pre-check
      if (spec == 1 && lock_pending_dead(v, L, plast_off, plast_len, store, DBF, aoff, adim, na,
                                    wx, wy, wz, scale, constant)) continue;
      atomicMin(&best[i], (unsigned long long)pos);
    }
    __threadfence(); grid.sync();
    for (int q = tid; q < nu; q += nthreads) {
      int i = BROOK_VLOAD_I(&cur[q]);
      int L = act[i];
      unsigned long long b = BROOK_VLOAD_U64(&best[i]);
      if (b != BROOK_MAXKEY) { if (!spec) cursor[L] = (long long)b; tgt[i] = order[b]; continue; }
      long long c = (long long)BROOK_VLOAD_U64(&wcur[i]) + W;
      if (c >= seg_end[L]) { if (!spec) cursor[L] = seg_end[L]; tgt[i] = -1; continue; }
      wcur[i] = c;
      if (!spec) cursor[L] = c;
      nxt[atomicAdd(&hdr[1], 1)] = i;
    }
    __threadfence(); grid.sync();
    if (tid == 0) { hdr[0] = BROOK_VLOAD_I(&hdr[1]); hdr[1] = 0; }
    int* t = cur; cur = nxt; nxt = t;
    if (shift < 26) shift += 2;
    __threadfence(); grid.sync();
  }
}

// Draft j (>= 1) of every active label: the first voxel of its order after draft j-1's that is
// alive and farther than ONE invalidation ball (scale * DBF + const of each earlier draft target,
// straight line) from every earlier draft target: an estimate of the invalidations not yet
// computed. kd_accept keeps a draft only if it is the target the sequential run finds. The search
// gives up after `max_rounds` windows (4^r * 4096 entries): no draft then, which costs speed,
// never correctness.
// No drafts for a label in its manual-target phase, or beyond its path budget.
__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) draft_lab(
    const unsigned char* alive, const int* order, const long long* seg_end,
    const int* act, const int* ih, const int* khdr, const int* pcount, const int* maxp,
    int K, int cap, int* tgtK, long long* dposK, const int* depth,
    long long* wcur, int* nd, int* un_a, int* un_b, int* bits, int* hdr, int max_windows,
    const float* DBF, const long long* aoff, const long long* adim, int na,
    float wx, float wy, float wz, float scale, float constant)
{
  // All drafts of a label in a few passes over a window of its order (BROOK_DRAFT_WIN entries from
  // wcur): phase 1 (one thread per order entry) sets a bit for every voxel that is alive and
  // outside the balls of the drafts the label has so far; phase 2 (one thread per label) walks
  // the set bits in order and keeps a voxel iff it is also outside the balls of the drafts kept in
  // this pass -- the greedy rule "first voxel of the order outside every earlier draft's ball" --
  // but examines no more than 32 of them: what the new drafts' balls cover is better struck out by
  // the next pass's phase 1, in parallel, than walked through. A pass keeps at least one draft
  // or moves the window on. Only the first khdr[3] labels of the active list draft.
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  int nact = ih[0], nl = khdr[3] < nact ? khdr[3] : nact;
  if (tid == 0) { hdr[0] = 0; hdr[1] = 0; }
  __threadfence(); grid.sync();

  for (int i = tid; i < nact; i += nthreads) {
    for (int j = 1; j < K; j++) { tgtK[j * cap + i] = -1; dposK[j * cap + i] = -1; }
    if (i >= nl || tgtK[i] < 0 || dposK[i] < 0) continue;
    int L = act[i];
    if (maxp[L] - pcount[L] < 2) continue;
    wcur[i] = dposK[i] + 1;
    if (wcur[i] >= seg_end[L]) continue;
    nd[i] = 1 | (max_windows << 8);                        // drafts so far | windows it may still open
    un_a[atomicAdd(&hdr[0], 1)] = i;
  }
  __threadfence(); grid.sync();

  int* cur = un_a; int* nxt = un_b;
  for (int pass = 0; pass < K + max_windows; pass++) {
    int nu = BROOK_VLOAD_I(&hdr[0]);
    if (nu == 0) break;
    long long items = (long long)nu * BROOK_DRAFT_WIN;
    for (long long q = tid; q < items; q += nthreads) {
      int slot = (int)(q / BROOK_DRAFT_WIN), off = (int)(q % BROOK_DRAFT_WIN), i = BROOK_VLOAD_I(&cur[slot]);
      int L = act[i];
      long long pos = (long long)BROOK_VLOAD_U64(&wcur[i]) + off;
      if (pos >= seg_end[L]) continue;
      long long v = (long long)order[pos];
      if (alive[v] == 0) continue;
      int a = lock_arena(v, aoff, na);
      long long a0 = aoff[a], sx = adim[3 * a], sy = adim[3 * a + 1];
      long long x, y, z; brook_unravel(v - a0, sx, sy, x, y, z);
      int far = 1, have = BROOK_VLOAD_I(&nd[i]) & 0xFF;
      for (int jj = 0; jj < have && far; jj++) {
        long long t = (long long)tgtK[jj * cap + i];
        long long tx, ty, tz; brook_unravel(t - a0, sx, sy, tx, ty, tz);
        float dx = wx * (float)(x - tx), dy = wy * (float)(y - ty), dz = wz * (float)(z - tz);
        float r = scale * DBF[t] + constant;
        if (dx * dx + dy * dy + dz * dz <= r * r) far = 0;
      }
      if (far) atomicOr(&bits[(long long)slot * (BROOK_DRAFT_WIN >> 5) + (off >> 5)], (int)(1u << (off & 31)));
    }
    __threadfence(); grid.sync();
    for (int s = tid; s < nu; s += nthreads) {
      int i = BROOK_VLOAD_I(&cur[s]);
      int L = act[i], st = BROOK_VLOAD_I(&nd[i]), had = st & 0xFF, n = had, left = st >> 8;
      int lim = maxp[L] - pcount[L]; if (lim > depth[L]) lim = depth[L];       // (the label's HORIZON)
      int a = lock_arena((long long)tgtK[i], aoff, na);
      unsigned int a0 = (unsigned int)aoff[a], sx = (unsigned int)adim[3 * a], sy = (unsigned int)adim[3 * a + 1];
      int* base = bits + (long long)s * (BROOK_DRAFT_WIN >> 5);
      long long w0 = (long long)BROOK_VLOAD_U64(&wcur[i]), next = w0 + BROOK_DRAFT_WIN;
      int seen = 0;
      for (int wd = 0; wd < (BROOK_DRAFT_WIN >> 5); wd++) {
        unsigned int word = (unsigned int)BROOK_VLOAD_I(&base[wd]);
        if (word == 0) continue;
        base[wd] = 0;
        for (int bb = 0; bb < 32 && n < lim && seen < 32; bb++) {
          if (!((word >> bb) & 1u)) continue;
          long long pos = w0 + ((long long)wd << 5) + bb;
          unsigned int v = (unsigned int)order[pos] - a0;          // 32-bit: native divisions
          unsigned int z = v / (sx * sy), rem = v - z * (sx * sy), y = rem / sx, x = rem - y * sx;
          int far = 1;
          for (int jj = had; jj < n && far; jj++) {                // (the earlier ones: phase 1)
            unsigned int t = (unsigned int)tgtK[jj * cap + i] - a0;
            unsigned int tz = t / (sx * sy), tr = t - tz * (sx * sy), ty = tr / sx, tx = tr - ty * sx;
            float dx = wx * (float)((int)x - (int)tx), dy = wy * (float)((int)y - (int)ty),
                  dz = wz * (float)((int)z - (int)tz);
            float r = scale * DBF[tgtK[jj * cap + i]] + constant;
            if (dx * dx + dy * dy + dz * dz <= r * r) far = 0;
          }
          if (far) { tgtK[n * cap + i] = order[pos]; dposK[n * cap + i] = pos; n++; }
          if (++seen == 32) next = pos + 1;                        // the next pass goes on from here
        }
      }
      if (n >= lim) continue;
      if (seen < 32 && --left <= 0) continue;                      // window exhausted: open the next, if any
      if (next >= seg_end[L]) continue;
      wcur[i] = next; nd[i] = n | (left << 8);
      nxt[atomicAdd(&hdr[1], 1)] = i;
    }
    __threadfence(); grid.sync();
    if (tid == 0) { hdr[0] = BROOK_VLOAD_I(&hdr[1]); hdr[1] = 0; }
    int* t = cur; cur = nxt; nxt = t;
    __threadfence(); grid.sync();
  }
}

// Ball invalidation for the new paths of every label at once, in kimimaro's claim order: the
// algorithm of invalidation.inval_coop (bands of the bottleneck distance M on a fixed grid, pulls
// at the band boundaries, round-synchronous pushes from a snapshot inside a band, touched list)
// plus the label gate and the arenas. All labels share the band counter; the grid is fixed, so
// a label only sees extra bands in which nothing of it moves: per label the flood is the
// per-component one, bit for bit.
// hdr: [0],[1] frontier counts, [2] touched, [3] invalidated count, [4] STICKY flag: a round
// budget was exhausted (never cleared here; the driver checks it once at the end),
// [5],[6] waiting-list counts, [7] smallest pulled M (bits)
// Waiting voxels keep their best offer (M, source) in their own `key`, which is unused while they
// are unclaimed: UNCLAIMED = top bit set (BROOK_MAXKEY = no offer), M's float bits in bits 32..62,
// the source below: the packed order of a claim's key, so atomicMin keeps what the exact pull
// would find, and a claim (top bit clear) always wins against it. Every claimed voxel expands
// with its final key before its band ends, so the record IS the pull's result -- unless one of
// the offers came from a key that improved afterwards (stale: it may be smaller than anything the
// final keys give). A voxel whose key improves is flagged (BROOK_F_IS); when it expands again it
// flags the waiting neighbours it may have misled (BROOK_F_ID), and only THOSE are pulled (26
// neighbours, exact) at the band boundary; all others are promoted from their record with no
// neighbour read at all. Bands, claim order and result are those of invalidation.inval_coop;
// the voxels of a thick first path just stop being read twice.
#define LINV_UNCLAIMED(k) (((k) >> 63) != 0ULL)
#define LINV_WAIT(mb, s)  ((((unsigned long long)(unsigned int)(mb) | 0x80000000ULL) << 32) | (unsigned long long)(unsigned int)(s))
#define LINV_BOUND(k)     ((int)(((k) >> 32) & 0x7FFFFFFFULL))
#define LINV_ALL          (BROOK_F_IQ | BROOK_F_IW | BROOK_F_IS | BROOK_F_ID)

__device__ __forceinline__ void linv_offer(
    long long v, float d, int muBits, int su, int hiBits, unsigned long long* key, int* inq,
    int* nxt, int* far, int* fcount, unsigned long long* pb, int lane_max, int* touched, int* hdr)
{
  int mb = __float_as_int(d);
  if (mb < muBits) mb = muBits;                                  // M = max(M[u], d)
  if (mb < hiBits) {
    unsigned long long candkey = ((unsigned long long)(unsigned int)mb << 32)
                                 | (unsigned long long)(unsigned int)su;
    unsigned long long old = atomicMin(key + v, candkey);
    if (old > candkey) {
      if (LINV_UNCLAIMED(old)) touched[atomicAdd(&hdr[2], 1)] = (int)v;
      else atomicOr(&inq[v], BROOK_F_IS);                          // it may have offered from the old key
      int was = atomicOr(&inq[v], BROOK_F_IQ);                   // -> frontier, from idle or waiting
      if (was & BROOK_F_IW) atomicAnd(&inq[v], ~BROOK_F_IW);
      if ((was & BROOK_F_IQ) == 0) nxt[atomicAdd(&hdr[1], 1)] = (int)v;
    }
  } else if (LINV_UNCLAIMED(BROOK_VLOAD_U64(&key[v]))) {
    unsigned long long wait = LINV_WAIT(mb, su);
    if (BROOK_VLOAD_U64(&key[v]) > wait) atomicMin(key + v, wait);        // (a claim still wins)
    unsigned int seen = (unsigned int)BROOK_VLOAD_I(&inq[v]), got = 0u;  // idle -> waiting; the railroad's
    int won = 0;                                                        // bits may change under the CAS
    while ((seen & (BROOK_F_IQ | BROOK_F_IW)) == 0u) {
      got = atomicCAS((unsigned int*)&inq[v], seen, seen | BROOK_F_IW);
      if (got == seen) { won = 1; break; }
      seen = got;
    }
    if (won) {
      int pos = atomicAdd(fcount, 1);
      far[pos] = (int)v;
      if (pos < lane_max) pb[pos] = BROOK_MAXKEY;                // its slot for a lane-parallel pull
    }
  }
}

// Expand voxel u from the (source, M) it held at the round boundary (label gate, arena frame).
// A voxel graph gates the offer u -> v (push, as inval_expand); an edge it blocks carries no
// offer, so the neighbour cannot have been misled by u either.
__device__ __forceinline__ void linv_expand(
    long long u, int su32, int muRaw, int hiBits, const unsigned char* alive, const float* DBF,
    const LABEL_T* lab, const unsigned int* vg, int has_vg,
    unsigned long long* key, const long long* aoff, const long long* adim, int na,
    float wx, float wy, float wz, float scale, float constant,
    int* inq, int* nxt, int* far, int* fcount, unsigned long long* pb, int lane_max, int* touched, int* hdr)
{
  long long su = (long long)su32;
  int muBits = muRaw & 0x7FFFFFFF, stale = muRaw < 0;
  LABEL_T lu = lab[u];
  int a = lock_arena(u, aoff, na);
  long long a0 = aoff[a], sx = adim[3 * a], sy = adim[3 * a + 1], sz = adim[3 * a + 2];
  long long sux, suy, suz; brook_unravel(su - a0, sx, sy, sux, suy, suz);
  float maxd = scale * DBF[su] + constant;
  long long x, y, z; brook_unravel(u - a0, sx, sy, x, y, z);
  for (int k = 0; k < 26; k++) {
    long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    long long v = a0 + BROOK_IDX(nx, ny, nz, sx, sy);
    if (alive[v] == 0 || lab[v] != lu) continue;
    if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
    if (stale && (BROOK_VLOAD_I(&inq[v]) & BROOK_F_IW)) atomicOr(&inq[v], BROOK_F_ID);
    float dvx = wx * (float)(nx - sux), dvy = wy * (float)(ny - suy), dvz = wz * (float)(nz - suz);
    float d = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
    if (d >= maxd) continue;
    linv_offer(v, d, muBits, su32, hiBits, key, inq, nxt, far, fcount, pb, lane_max, touched, hdr);
  }
}

__device__ __forceinline__ int linv_band(int mbits, float delta, int kmin) {
  float m = __int_as_float(mbits);
  int k = (int)(m / delta);
  while ((float)(k + 1) * delta <= m) k++;
  while (k > 0 && (float)k * delta > m) k--;
  return k < kmin ? kmin : k;
}

__global__ void __launch_bounds__(BROOK_COOP_BLOCK, BROOK_COOP_MIN_BLOCKS) inval_lab(
    unsigned char* alive, const float* DBF, const LABEL_T* lab, const unsigned int* vg, int has_vg,
    unsigned long long* key, int* snap, int* snapm,
    const long long* aoff, const long long* adim, int na,
    float wx, float wy, float wz, float scale, float constant, float delta,
    const int* store, const int* ihp,
    int* wl_a, int* wl_b, int* far_a, int* far_b, int* inq, int* touched, int* hdr, int max_rounds,
    int lane_max, float* smax, long long* sxyz, int* sar, unsigned long long* pb_a, unsigned long long* pb_b)
{
  // the sources: store[ihp[0] .. ihp[0] + ihp[1]) (the iteration header's [2], [3]); none when parked
  const int* path = store + ihp[0];
  int npath = ihp[3] ? 0 : ihp[1];
  cg::grid_group grid = cg::this_grid();
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  if (tid == 0) { hdr[0] = 0; hdr[1] = 0; hdr[2] = 0; hdr[3] = 0; hdr[5] = 0; hdr[6] = 0; hdr[7] = 0x7f800000; }
  __threadfence(); grid.sync();

  for (int i = tid; i < npath; i += nthreads) {
    long long p = (long long)path[i];
    if (alive[p] == 0) continue;                                // invalidated earlier
    unsigned long long old = atomicMin(key + p, (unsigned long long)(unsigned int)p);
    if (LINV_UNCLAIMED(old)) {                                  // dedupes repeated vertices
      atomicOr(&inq[p], BROOK_F_IQ);
      wl_a[atomicAdd(&hdr[0], 1)] = (int)p;
      touched[atomicAdd(&hdr[2], 1)] = (int)p;
    }
  }
  __threadfence(); grid.sync();

  int* cur = wl_a; int* nxt = wl_b; int* far = far_a; int* keep = far_b;
  unsigned long long* pbf = pb_a; unsigned long long* pbk = pb_b;
  int band = 0, round = 0, open = 1, presnap = 0;
  while (open) {
    int hiBits = __float_as_int((float)(band + 1) * delta);
    for (; round < max_rounds; round++) {                       // rounds inside the band
      int fc = BROOK_VLOAD_I(&hdr[0]);
      if (fc == 0) break;
      int lanes = (fc <= lane_max);
      if (!presnap) {                                           // (a lane boundary wrote the snapshot)
      for (int i = tid; i < fc; i += nthreads) {                // phase 1: snapshot, re-arm
        long long u = (long long)BROOK_VLOAD_I(&cur[i]);
        unsigned long long ku = BROOK_VLOAD_U64(&key[u]);
        int su = (int)(ku & 0xffffffffULL);
        int fl = atomicAnd(&inq[u], ~(BROOK_F_IQ | BROOK_F_IW | BROOK_F_IS));
        snap[i] = su; snapm[i] = (int)(ku >> 32) | ((fl & BROOK_F_IS) ? (int)0x80000000 : 0);
        if (lanes) {                                            // thin front: see dstep_lab
          int a = lock_arena(u, aoff, na);
          long long x, y, z; brook_unravel(u - aoff[a], adim[3 * a], adim[3 * a + 1], x, y, z);
          long long qx, qy, qz; brook_unravel((long long)su - aoff[a], adim[3 * a], adim[3 * a + 1], qx, qy, qz);
          sxyz[2 * i] = x | (y << 21) | (z << 42);
          sxyz[2 * i + 1] = qx | (qy << 21) | (qz << 42);
          smax[i] = scale * DBF[su] + constant; sar[i] = a;
        }
      }
      __threadfence(); grid.sync();
      }
      presnap = 0;
      if (lanes) {
        long long items = (long long)fc << 5;
        for (long long j = tid; j < items; j += nthreads) {     // phase 2, one lane per neighbour
          int k = (int)(j & 31);
          if (k >= 26) continue;
          int i = (int)(j >> 5);
          long long p = (long long)__ldcg(&sxyz[2 * i]), q = (long long)__ldcg(&sxyz[2 * i + 1]);
          int a = BROOK_VLOAD_I(&sar[i]);
          long long sx = adim[3 * a], sy = adim[3 * a + 1], sz = adim[3 * a + 2];
          long long nx = (p & 0x1FFFFF) + BROOK_NX[k], ny = ((p >> 21) & 0x1FFFFF) + BROOK_NY[k],
                    nz = (p >> 42) + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long v = aoff[a] + BROOK_IDX(nx, ny, nz, sx, sy);
          long long u = (long long)BROOK_VLOAD_I(&cur[i]);
          if (alive[v] == 0 || lab[v] != lab[u]) continue;
          if (!brook_vg_ok_push(vg, has_vg, u, k)) continue;
          int mraw = BROOK_VLOAD_I(&snapm[i]);
          if (mraw < 0 && (BROOK_VLOAD_I(&inq[v]) & BROOK_F_IW)) atomicOr(&inq[v], BROOK_F_ID);
          long long sux = q & 0x1FFFFF, suy = (q >> 21) & 0x1FFFFF, suz = q >> 42;
          float dvx = wx * (float)(nx - sux), dvy = wy * (float)(ny - suy), dvz = wz * (float)(nz - suz);
          float d = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
          if (d >= BROOK_VLOAD_F(&smax[i])) continue;
          linv_offer(v, d, mraw & 0x7FFFFFFF, BROOK_VLOAD_I(&snap[i]), hiBits, key, inq, nxt, far, &hdr[5], pbf, lane_max, touched, hdr);
        }
      } else
      for (int i = tid; i < fc; i += nthreads)                  // phase 2: expand the snapshot
        linv_expand((long long)BROOK_VLOAD_I(&cur[i]), BROOK_VLOAD_I(&snap[i]), BROOK_VLOAD_I(&snapm[i]), hiBits,
                    alive, DBF, lab, vg, has_vg, key, aoff, adim, na, wx, wy, wz, scale, constant,
                    inq, nxt, far, &hdr[5], pbf, lane_max, touched, hdr);
      __threadfence(); grid.sync();
      if (tid == 0) { hdr[0] = BROOK_VLOAD_I(&hdr[1]); hdr[1] = 0; }
      int* t = cur; cur = nxt; nxt = t;
      __threadfence(); grid.sync();
    }
    // ---- band boundary: next band | pull | promote | publish. The bands, the pulls and what is
    // promoted are those of invalidation.inval_coop; only WHO is pulled differs (the bounds, see
    // linv_offer's note). Thin waiting lists pull with one lane per (voxel, neighbour) and their
    // promotion writes the snapshot of the first round. hdr[7] was armed in the publish window.
    int nfar = BROOK_VLOAD_I(&hdr[5]);
    if (nfar == 0 || round >= max_rounds) { open = 0; continue; }
    for (int i = tid; i < nfar; i += nthreads) {                // the smallest bound names the next band
      unsigned long long kc = BROOK_VLOAD_U64(&key[BROOK_VLOAD_I(&far[i])]);
      if (LINV_UNCLAIMED(kc) && BROOK_VLOAD_I(&hdr[7]) > LINV_BOUND(kc)) atomicMin(&hdr[7], LINV_BOUND(kc));
    }
    __threadfence(); grid.sync();
    int lowest = BROOK_VLOAD_I(&hdr[7]);
    if (lowest < 0x7f800000) band = linv_band(lowest, delta, band + 1);
    hiBits = __float_as_int((float)(band + 1) * delta);
    int plan = (nfar <= lane_max);
    long long items = plan ? ((long long)nfar << 5) : (long long)nfar;
    for (long long j = tid; j < items; j += nthreads) {
      int i = plan ? (int)(j >> 5) : (int)j;
      int k0 = plan ? (int)(j & 31) : 0, k1 = plan ? k0 + 1 : 26;
      if (k0 >= 26) continue;
      long long c = (long long)BROOK_VLOAD_I(&far[i]);
      unsigned long long best = BROOK_MAXKEY, kc = BROOK_VLOAD_U64(&key[c]);
      int claimed = !LINV_UNCLAIMED(kc);
      int later = !claimed && (LINV_BOUND(kc) >= hiBits          // not in this band, or its record is exact:
                               || !(BROOK_VLOAD_I(&inq[c]) & BROOK_F_ID));                    // no pull
      if (!claimed && !later) {
        LABEL_T lc = lab[c];
        int a = lock_arena(c, aoff, na);
        long long a0 = aoff[a], sx = adim[3 * a], sy = adim[3 * a + 1], sz = adim[3 * a + 2];
        long long x, y, z; brook_unravel(c - a0, sx, sy, x, y, z);
        for (int k = k0; k < k1; k++) {
          long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
          long long u = a0 + BROOK_IDX(nx, ny, nz, sx, sy);
          if (lab[u] != lc) continue;
          if (!brook_vg_ok(vg, has_vg, u, k)) continue;        // pull: the edge u -> c
          unsigned long long ku = BROOK_VLOAD_U64(&key[u]);
          if (LINV_UNCLAIMED(ku)) continue;
          long long su = (long long)(ku & 0xffffffffULL);
          long long sux, suy, suz; brook_unravel(su - a0, sx, sy, sux, suy, suz);
          float dvx = wx * (float)(x - sux), dvy = wy * (float)(y - suy), dvz = wz * (float)(z - suz);
          float d = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
          if (d >= scale * DBF[su] + constant) continue;
          int mb = __float_as_int(d), mu = (int)(ku >> 32);
          if (mb < mu) mb = mu;
          unsigned long long cand = ((unsigned long long)(unsigned int)mb << 32) | (unsigned long long)(unsigned int)su;
          if (cand < best) best = cand;
        }
      }
      if (plan) {
        if (best != BROOK_MAXKEY) atomicMin(pbf + i, best);
      } else {
        snap[i] = claimed ? -1 : (later ? -3 : (best == BROOK_MAXKEY ? -2 : (int)(best & 0xffffffffULL)));
        snapm[i] = (int)(best >> 32);
      }
    }
    __threadfence(); grid.sync();
    for (int i = tid; i < nfar; i += nthreads) {                // promote / keep waiting / drop
      int c = BROOK_VLOAD_I(&far[i]);
      int s = -3, mb = 0;
      unsigned long long kc = BROOK_VLOAD_U64(&key[c]);
      if (!LINV_UNCLAIMED(kc)) continue;                        // claimed inside the last band
      int dirty = BROOK_VLOAD_I(&inq[c]) & BROOK_F_ID;
      if (LINV_BOUND(kc) < hiBits && !dirty) {                  // its record is the pull's result
        s = (int)(kc & 0xffffffffULL); mb = LINV_BOUND(kc);
      } else if (LINV_BOUND(kc) < hiBits) {                     // it was pulled
        if (plan) {
          unsigned long long b = BROOK_VLOAD_U64(&pbf[i]);
          s = (b == BROOK_MAXKEY) ? -2 : (int)(b & 0xffffffffULL); mb = (int)(b >> 32);
        } else {
          s = BROOK_VLOAD_I(&snap[i]); mb = BROOK_VLOAD_I(&snapm[i]);
        }
        if (s == -2) {                                          // no offer in reach (it may return)
          key[c] = BROOK_MAXKEY;
          atomicAnd(&inq[c], ~LINV_ALL);
          continue;
        }
      }
      if (s != -3 && mb < hiBits) {
        key[c] = ((unsigned long long)(unsigned int)mb << 32) | (unsigned long long)(unsigned int)s;
        touched[atomicAdd(&hdr[2], 1)] = c;
        int pos = atomicAdd(&hdr[1], 1);
        nxt[pos] = c;
        if (plan) {                                             // = phase 1 of its first round
          atomicAnd(&inq[c], ~LINV_ALL);
          snap[pos] = s; snapm[pos] = mb;
          int a = lock_arena((long long)c, aoff, na);
          long long x, y, z; brook_unravel((long long)c - aoff[a], adim[3 * a], adim[3 * a + 1], x, y, z);
          long long qx, qy, qz; brook_unravel((long long)s - aoff[a], adim[3 * a], adim[3 * a + 1], qx, qy, qz);
          sxyz[2 * pos] = x | (y << 21) | (z << 42);
          sxyz[2 * pos + 1] = qx | (qy << 21) | (qz << 42);
          smax[pos] = scale * DBF[s] + constant; sar[pos] = a;
        } else { atomicOr(&inq[c], BROOK_F_IQ); atomicAnd(&inq[c], ~(BROOK_F_IW | BROOK_F_ID)); }
      } else {
        if (s != -3) { key[c] = LINV_WAIT(mb, s); atomicAnd(&inq[c], ~BROOK_F_ID); }   // the pull's result is its record
        int pos = atomicAdd(&hdr[6], 1);
        keep[pos] = c;
        if (pos < lane_max) pbk[pos] = BROOK_MAXKEY;
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

  int n = BROOK_VLOAD_I(&hdr[2]);
  for (int i = tid; i < n; i += nthreads) {                     // invalidated set == touched
    int v = BROOK_VLOAD_I(&touched[i]);
    alive[v] = 0; key[v] = BROOK_MAXKEY; atomicAnd(&inq[v], ~LINV_ALL);
  }
  __threadfence(); grid.sync();
  if (tid == 0) {
    hdr[3] = n;
    if (BROOK_VLOAD_I(&hdr[0]) != 0 || BROOK_VLOAD_I(&hdr[5]) != 0) hdr[4] = 1;
  }
}
