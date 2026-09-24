// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti

#define BROOK_INF __int_as_float(0x7f800000)
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
// Draft windows (body "k") are arenas too, all boxes of one power-of-two size laid after the dense
// arenas: `na` carries the dense count in its low 16 bits and log2(window voxels) above them, so
// that a window's arena is found by a shift, not by a scan over hundreds of arenas.
__device__ __forceinline__ int lock_arena(long long u, const long long* aoff, int na) {
  int nd = na & 0xFFFF, sh = na >> 16;
  if (sh != 0 && u >= aoff[nd]) return nd + (int)((u - aoff[nd]) >> sh);
  int a = 0;
  while (a + 1 < nd && u >= aoff[a + 1]) a++;
  return a;
}

// Soma trimming (kimimaro: path = path[:1] + the vertices farther than soma_radius from the
// root). Same float64 arithmetic as numpy's norm(anisotropy_f32 * (path_u32 - root_f32)).
__device__ __forceinline__ bool lock_far(long long x, long long y, long long z, const long long* rxyz,
                                         float wx, float wy, float wz, double radius) {
  double dx = (double)wx * ((double)x - (double)rxyz[0]);
  double dy = (double)wy * ((double)y - (double)rxyz[1]);
  double dz = (double)wz * ((double)z - (double)rxyz[2]);
  double ss = dx * dx + dy * dy;
  ss = ss + dz * dz;
  return sqrt(ss) > radius;
}

// Backtrace straight from the distance field, one thread per label: from the label's nearest
// rail step to the predecessor = argmin neighbour dist within the label (rails skipped), first
// k among the minima (the rule of the warp backtrace in sssp_kernels.cuh), until the target (dist 0).
// pass 0 writes the length (-1: the label's flood failed), pass 1 writes target..rail at off[i].
// Soma labels (soma_r[L] >= 0) keep the output's first vertex and the vertices beyond the radius.
// A path is target-first, so when its rail end lies within the radius the trimmed remainder has
// no attachment left: the soma root is emitted in its place (the spoke a root-first path keeps
// as path[:1]). A window (L >= first_window) cannot address the root: such a draft is void.
// A voxel graph gates the predecessor pull u -> cur as backtrace_by_dist_warp does.
// The join: the nearest rail fixes the cost D and the last road voxel u (the first step), and
// every rail of the label that u may push to has distance D too. The path joins u's first one in
// dijkstra3d order (BROOK_D3RANK, the rule of backtrace_by_dist_warp) and steps from it to u; the
// soma rule tests that rail. ju / jr (windows, may be null): u's state index (-1: no join) and
// that rail's rank (64: none), for the draft checks (k2_lens, k2_route, k2_accept).
__global__ void backtrace_dist_lab(
    const float* dist, const float* field, const LABEL_T* lab, const unsigned int* vg, int has_vg,
    const unsigned long long* rail_best, const int* done,
    const int* act, const int* tgt, const int* ih,
    const long long* aoff, const long long* adim, int na,
    const double* soma_r, const long long* soma_root, float wx, float wy, float wz,
    int pass, const int* off, int* len, int* out, int maxlen, int* walk, const long long* seg_start, const long long* seg_end, const int* wfill,
    int first_window, int* ju, int* jr)
{
  // One block of 32 threads per active label (one warp): lane k probes neighbour k of the
  // current voxel and two synchronous min-reductions pick the smallest distance, then the lowest
  // k among the minima -- the scalar loop's ascending-k strict-< rule, so the path is the same.
  // A step costs one neighbour read instead of 26 dependent ones (the warp backtrace;
  // the walk is this kernel's latency). Everything that steers control flow (label, target,
  // current voxel, kept count) is identical on all lanes; lane 0 alone writes.
  int i = blockIdx.x, k = threadIdx.x;
  if (i >= ih[0] || (pass && ih[5])) return;
  int L = act[i]; long long t = tgt[i];
  if (pass == 0 && k == 0) len[i] = 0;
  if (pass == 0 && k == 0 && ju && L >= first_window) { ju[L - first_window] = -1; jr[L - first_window] = 64; }
  if (t < 0) return;
  if (done[L] != 1) { if (pass == 0 && k == 0) len[i] = -1; return; }
  // One walk: pass 0 keeps its vertices in the label's own stretch of an idle worklist buffer
  // (as long as the label has voxels, and a path has no more vertices than that); pass 1, once
  // the offsets are known, only copies them reversed.
  int* w = walk + seg_start[L] + wfill[i];
  long long room = seg_end[L] - seg_start[L] - wfill[i];
  if (pass) {
    if (k) return;
    int n = len[i]; long long base = off[i];
    for (int j = 0; j < n; j++) out[base + (n - 1 - j)] = w[j];
    return;
  }
  long long cur = (long long)(rail_best[L] & 0xFFFFFFFFULL);
  int a = lock_arena(cur, aoff, na);
  long long a0 = aoff[a], sx = adim[3 * a], sy = adim[3 * a + 1], sz = adim[3 * a + 2];
  long long x, y, z; brook_unravel(cur - a0, sx, sy, x, y, z);
  double sr = soma_r[L];
  int cnt = 0, kept = 0;
  while (cnt < maxlen) {
    int kb = -1;
    if (cur != t) {
      float du = BROOK_INF;
      if (k < 26) {
        long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
        if (nx >= 0 && ny >= 0 && nz >= 0 && nx < sx && ny < sy && nz < sz) {
          long long u = a0 + BROOK_IDX(nx, ny, nz, sx, sy);
          if (lab[u] == (LABEL_T)L && field[u] != 0.0f && brook_vg_ok(vg, has_vg, u, k)) du = dist[u];
        }
      }
      int db = __float_as_int(du);                        // dist >= 0: bits order like the floats
      int mb = __reduce_min_sync(0xFFFFFFFFu, db);
      if (mb < 0x7f800000) kb = __reduce_min_sync(0xFFFFFFFFu, (db == mb) ? k : 64);
    }
    if (cnt == 0 && kb >= 0) {                            // The join (see above), at u = cur + N[kb]
      long long ux = x + BROOK_NX[kb], uy = y + BROOK_NY[kb], uz = z + BROOK_NZ[kb];
      long long uidx = a0 + BROOK_IDX(ux, uy, uz, sx, sy);
      int key = 64;
      if (k < 26) {
        long long nx = ux + BROOK_NX[k], ny = uy + BROOK_NY[k], nz = uz + BROOK_NZ[k];
        if (nx >= 0 && ny >= 0 && nz >= 0 && nx < sx && ny < sy && nz < sz) {
          long long v = a0 + BROOK_IDX(nx, ny, nz, sx, sy);
          if (lab[v] == (LABEL_T)L && field[v] == 0.0f && brook_vg_ok_push(vg, has_vg, uidx, k)) key = BROOK_D3RANK[k];
        }
      }
      int r = __reduce_min_sync(0xFFFFFFFFu, key);        // < 26: the entry itself qualifies
      if (r < 26) {
        int ks = BROOK_D3INV[r];
        x = ux + BROOK_NX[ks]; y = uy + BROOK_NY[ks]; z = uz + BROOK_NZ[ks];
        cur = a0 + BROOK_IDX(x, y, z, sx, sy);
        kb = BROOK_OPP[ks];                                 // the step below goes from it to u
        if (ju && L >= first_window && k == 0) { ju[L - first_window] = (int)uidx; jr[L - first_window] = r; }
      }
    }
    bool far = sr < 0.0 || lock_far(x, y, z, soma_root + 3 * L, wx, wy, wz, sr);
    if (cnt == 0 && !far) {                                // the rail end is trimmed: attach to the root
      if (L >= first_window || kept >= room) { if (k == 0) len[i] = -1; return; }
      const long long* r = soma_root + 3 * L;
      if (k == 0) w[kept] = (int)(a0 + BROOK_IDX(r[0], r[1], r[2], sx, sy));
      kept++;
    }
    // kb < 0: the target, or a broken chain -> this vertex is the output's first one
    if (far || kb < 0) {
      if (kept >= room) { if (k == 0) len[i] = -1; return; }   // cannot happen on a valid field: eject
      if (k == 0) w[kept] = (int)cur;
      kept++;
    }
    cnt++;
    if (kb < 0) break;
    x += BROOK_NX[kb]; y += BROOK_NY[kb]; z += BROOK_NZ[kb];
    cur = a0 + BROOK_IDX(x, y, z, sx, sy);
  }
  if (k == 0) len[i] = kept;
}

// fix_branching=False: walk the parent field (global indices) target -> root; out = root..target.
// Emitted prefixes are not walked again (emitted-prefix rule). In a fixed parent tree an emitted path is
// a rootward chain, so everything rootward of an emitted vertex is emitted too: walking it again only
// repeats vertices and edges the label's skeleton already has, and re-offers invalidation sources that
// are dead (every path vertex is claimed by its own path's invalidation, which is complete before this
// path's begins). An emitted vertex no longer needs its parent, so the write pass turns it into a STOP
// (-2): the next walk ends ON the first emitted vertex it meets and keeps it as the attachment. Same
// union of vertices and edges, same radii, same path count; without the rule 54-94 % of the stored path
// vertices are such repeats (CREMI-A fine: 48 M stored for a union of 2.7 M). Not for soma labels: a trimmed path
// can bridge omitted vertices.
__global__ void backtrace_parent_lab(
    int* parent, const int* root, const int* act, const int* tgt, const int* ih,
    const long long* aoff, const long long* adim, int na,
    const double* soma_r, const long long* soma_root, float wx, float wy, float wz,
    int pass, const int* off, int* len, int* out, int maxlen, int* walk, const long long* seg_start, const long long* seg_end, const int* wfill,
    int stop_emitted)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ih[0] || (pass && ih[5])) return;
  int L = act[i]; int t = tgt[i];
  if (pass == 0) len[i] = 0;
  if (t < 0) return;
  int* w = walk + seg_start[L] + wfill[i];              // one walk, see backtrace_dist_lab
  long long room = seg_end[L] - seg_start[L] - wfill[i];
  double sr = soma_r[L];
  if (pass) {
    int n = len[i]; long long base = off[i];
    for (int j = 0; j < n; j++) { out[base + (n - 1 - j)] = w[j]; if (sr < 0.0 && stop_emitted) parent[w[j]] = -2; }
    return;
  }
  int a = lock_arena((long long)t, aoff, na);
  long long a0 = aoff[a], sx = adim[3 * a], sy = adim[3 * a + 1];
  int cur = t, src = root[L], cnt = 0, kept = 0;
  while (cnt < maxlen) {
    int p = (cur == src) ? -1 : parent[cur];
    bool keep = (sr < 0.0) || (p < 0);                   // p < 0: the output's first vertex
    if (!keep) {
      long long x, y, z; brook_unravel((long long)cur - a0, sx, sy, x, y, z);
      keep = lock_far(x, y, z, soma_root + 3 * L, wx, wy, wz, sr);
    }
    if (keep) { if (kept >= room) { len[i] = -1; return; } w[kept] = cur; kept++; }
    cnt++;
    if (p < 0) break;
    cur = p;
  }
  len[i] = kept;
}

// The iteration header ih (device int32[32]): everything the host would otherwise compute between the
// launches of an iteration, so that the iteration can run as the body of a CUDA-graph WHILE node.
//   [0] active labels   [1] iteration      [2] path-store write position   [3] vertices of this iteration
//   [4] path records    [5] PARKED: 1 = path store full, 2 = record store full (the host grows it,
//       runs the rest of the iteration itself and resumes)                  [6] labels finished
//   [7] paths           [8] labels ejected [9] store capacity [10] record capacity
//   [11] iterations until the loop hands control back (progress, growth)    [12] phases [13] rounds
//   [14] [15] the PENDING paths: store position and vertex count of the paths accepted by the
//       previous iteration, which this iteration invalidates ([16] [17] stay 0: inval_lab reads
//       its sources as ihp[0], ihp[1] and a parked flag at ihp[3])          [18] candidates rejected
//   [19] SWITCH: the loop stopped because an iteration's railroad flood touched no more than
//       [20] voxels (0 = never): from here on the iterations are latency-bound and the host
//       starts the two-branch body
//
// Pipelined speculation. Sequentially a label does  target -> flood -> backtrace -> invalidation
// -> target ...; the next target needs the invalidation, but the next FLOOD needs only the path
// (as a rail). So one iteration runs two branches at once:
//   A  the invalidation of the paths accepted by the previous iteration
//   B  a PREDICTED next target (first voxel of the label's farthest-first order that is alive and
//      outside the straight-line balls of the label's pending path: no connectivity, no claim
//      order, no rounds), its flood and its backtrace walk
// and then joins: the TRUE target is searched on the real invalidation; a candidate is accepted
// iff it IS the true target (then flood and walk are exactly the sequential ones: the rails were
// complete when the flood started), else it is dropped and the label tries again next iteration,
// now without a pending path, where the prediction is exact. Identical output by construction.
// lock_offsets: verdicts, then path offsets = pos + exclusive prefix sum of the accepted lengths.
// status: 0 accepted, 1 finished (no true target), 2 candidate rejected
__global__ void lock_offsets(const int* tgt_true, const int* tgt_hat, int* len, int* off,
                                         int* status, int* ih)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int nact = ih[0]; long long run = ih[2];
  for (int i = 0; i < nact; i++) {
    status[i] = tgt_true[i] < 0 ? 1 : (tgt_true[i] != tgt_hat[i] ? 2 : 0);
    if (status[i] != 0) len[i] = 0;
    off[i] = (int)run; run += len[i] > 0 ? len[i] : 0;
  }
  ih[3] = (int)(run - ih[2]);
  ih[5] = (run > (long long)ih[9]) ? 1 : ((long long)ih[4] + nact > (long long)ih[10] ? 2 : 0);
}

// the new path vertices become rails
__global__ void lock_rails(float* pdrf, const int* store, const int* ih)
{
  if (ih[5]) return;
  const int* seg = store + ih[2]; int n = ih[3];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) pdrf[seg[i]] = 0.0f;
}

// lock_advance: path records (label, length, start) in active-list order, finished labels leave,
// labels whose flood failed are ejected, rejected candidates just stay; per-label path counters
// and pending paths; the active list is compacted in place (stable), the header moves on, and
// the WHILE node learns whether to run the body again.
__global__ void lock_advance(
    int* act, const int* status, const int* len, const int* off, int* rec_lab, int* rec_len, int* rec_start,
    int* ejected, int* pcount, int* plast_off, int* plast_len,
    const int* counts, int has_counts, int* ih, unsigned long long handle, int* lane_stop)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int go = 0;
  if (ih[5] == 0) {
    int nact = ih[0], w = 0, nrec = ih[4];
    for (int i = 0; i < nact; i++) {
      int L = act[i];
      plast_len[L] = 0;
      if (status[i] == 1) { ih[6] += 1; continue; }
      if (status[i] == 2) { ih[18] += 1; act[w++] = L; continue; }
      if (len[i] < 0) { ejected[ih[8]] = L; ih[8] += 1; continue; }
      rec_lab[nrec] = L; rec_len[nrec] = len[i]; rec_start[nrec] = off[i]; nrec++;
      pcount[L] += 1; plast_off[L] = off[i]; plast_len[L] = len[i];
      act[w++] = L;
    }
    ih[7] += nrec - ih[4]; ih[4] = nrec;
    ih[14] = ih[2]; ih[15] = ih[3];                   // what the next iteration invalidates
    ih[0] = w; ih[1] += 1; ih[2] += ih[3]; ih[3] = 0;
    if (has_counts) { ih[12] += counts[6]; ih[13] += counts[7]; }
    ih[11] -= 1;
    go = (w > 0 && ih[11] > 0) ? 1 : 0;
    if (go && has_counts && ih[20] > 0 && counts[5] <= ih[20]) { ih[19] = 1; go = 0; }
  }
  // Lanes (Lockstep.run, k_enter): two loops run at once on two streams; the one whose chunk ends
  // first tells the other to come back too, so that neither waits idle. Schedule only.
  if (lane_stop[0]) go = 0;
  if (!go) lane_stop[0] = 1;
  if (handle) cudaGraphSetConditional(handle, (unsigned int)go);
}

// ===== K DRAFTS PER OBJECT (body "k": see Lockstep.run) =====
// Draft g = j * cap + i is the j-th draft of the label at active position i. tgtK / dposK: its
// target and that target's position in the label's farthest-first order; lenK: its walk.
// Draft 0 is the exact target and lives on the object's own voxels. Draft j >= 1 lives in a
// Window: a private arena (a cube of side 2^ws around its target, window w = i * (K - 1) + j - 1,
// pseudo-label ntab + w) holding COPIES of the object's penalty field, liveness and DBF. To the
// flood, backtrace and invalidation kernels a window is one more label at active position
// nact + w: nothing in them changes, every draft's flood and invalidation is private, and all of
// them run in the same launches. What a window cannot show is caught exactly:
//   Leaving it   window-face voxels of the object with the volume behind them are EXITS: the copy
//                makes them rails, so a flood that wants out ends on one (k2_lens); an exit as
//                near as the entry (k2_reset) or claimed by the invalidation (k2_mark) also
//                rejects the draft.
//   Route        the paths of the lower drafts are rails for the sequential flood of draft b but
//                not for its private one. Distances up to the entry's D_e are final when a flood
//                ends, so b's road is the sequential one iff no vertex v of a lower path has a
//                neighbour u (same object, not a rail) with (dist_b[u], v) < (D_e, e) in the
//                flood's packed order, and b's target is on no lower path (k2_route).
//   Join         the rail b's walk joins (backtrace_dist_lab: u_b's first rail neighbour in
//                dijkstra3d order) is a rail of the batch's start; a lower path's vertex next to
//                u_b that comes earlier in that order is the sequential join instead. k2_route
//                records the earliest per pair (rrank), k2_accept takes the earliest over the
//                retired drafts and k2_write puts it at the walk's rail end. Every candidate is a
//                dead rail, so nothing else changes (a soma label ends its batch there instead:
//                the new rail could lie within the radius, where the root takes its place).
//   Contact      b's invalidation ran on the liveness of the batch's start; the sequential one
//                runs without the voxels the lower drafts removed. A voxel takes part in a flood
//                only once it is claimed (an offer that is never claimed -- out of reach, or
//                dropped at a band boundary -- moves nothing: waiting voxels do not propagate,
//                and the band grid is fixed), and every claimed voxel is removed. So the two
//                floods are the same iff b removed no voxel that a lower draft removed
//                (k2_mark / k2_contact).
//   Target       draft b must be the first voxel of the order the lower drafts left alive
//                (k2_scan / k2_accept).
// Retirement (k2_accept) walks a label's drafts in order with the set A of the drafts retired so
// far (draft 0 first). Every draft ran against the batch's start, so its certificates are per
// Pair: mark[v] has bit a for every draft a that removed v, rbad[w] bit a for every draft whose
// path would change w's route, cmask[w] bit a for every draft that removed a voxel w removed too,
// and only the pairs with a in A count. A draft whose target a retired draft removed is never
// asked for by the sequential run: it is SKIPPED (it joins nothing; BROOK_KPOOL=0: it ends the
// batch, the prefix rule) and its paths, deletions and conflicts touch no one: the drafts behind
// it are checked against A alone. Anything else that fails is the REQUIRED target failing, and
// ends the label's batch. The retired drafts are applied (k2_apply, k2_write); the rest only
// ever touched window copies and scratch. Window and object indices order voxels alike (a
// translation), so every lowest-index tie-break is the sequential one.
#define KD_NONE 0xFFFFFFFFFFFFFFFFULL

// windows are boxes of power-of-two sides: ws packs log2 of the three sides, a byte each
#define K2_W1(ws) ((ws) & 0xFF)
#define K2_W2(ws) (((ws) >> 8) & 0xFF)
#define K2_W3(ws) (((ws) >> 16) & 0xFF)
// window voxel s -> the same voxel in the object's own arena (state index; coordinates; window)
__device__ __forceinline__ long long k2_home(long long s, long long wbase, int ws, const int* worg,
    const int* whome, const long long* aoff, const long long* adim, int* wout,
    long long* hx, long long* hy, long long* hz)
{
  int w1 = K2_W1(ws), w2 = K2_W2(ws), w3 = K2_W3(ws);
  long long q = s - wbase;
  int w = (int)(q >> (w1 + w2 + w3));
  long long x = worg[3 * w] + (q & ((1LL << w1) - 1)), y = worg[3 * w + 1] + ((q >> w1) & ((1LL << w2) - 1)),
            z = worg[3 * w + 2] + ((q >> (w1 + w2)) & ((1LL << w3) - 1));
  int a = whome[w];
  *wout = w; *hx = x; *hy = y; *hz = z;
  return aoff[a] + BROOK_IDX(x, y, z, adim[3 * a], adim[3 * a + 1]);
}
// is window voxel (local lx, ly, lz; object coordinates x, y, z) on a face with the volume behind it?
__device__ __forceinline__ int k2_exit(long long lx, long long ly, long long lz, long long x, long long y,
    long long z, int ws, long long sx, long long sy, long long sz)
{
  long long m1 = (1LL << K2_W1(ws)) - 1, m2 = (1LL << K2_W2(ws)) - 1, m3 = (1LL << K2_W3(ws)) - 1;
  return (lx == 0 && x > 0) || (lx == m1 && x < sx - 1) || (ly == 0 && y > 0) || (ly == m2 && y < sy - 1)
      || (lz == 0 && z > 0) || (lz == m3 && z < sz - 1);
}

// position of draft 0 (the exact target) in the order; -1 for a manual target: no further drafts
__global__ void kd_first(const int* act, const int* ih, const int* tgt, const long long* cursor,
    const int* pcount, const int* mb_count, long long* dposK)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ih[0]) return;
  int L = act[i];
  dposK[i] = (tgt[i] < 0 || pcount[L] < mb_count[L]) ? -1 : cursor[L];
}

// the batch's label list: the active labels, then one pseudo-label per window of the first capK
// labels; windows are placed (centred on the draft, clamped into the object's arena)
__global__ void k2_setup(const int* act, const int* ih, int* ihx, int K, int cap, int capK, int ws,
    long long wbase, int ntab, const int* tgtK, int* actx, int* tgtx, int* posof, int* worg, int* whome,
    int* wlab, int* wact, int* rbad, int* wbad, int* cmask, double* soma_r, long long* soma_root,
    const long long* aoff, const long long* adim, int na, int* khdr, int* rrank)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nact = ih[0];
  capK = khdr[3] < capK ? khdr[3] : capK;                     // the labels that draft (draft_lab)
  int nwa = (nact < capK ? nact : capK) * (K - 1);
  if (t == 0) { ihx[0] = nact + nwa; ihx[5] = 0; khdr[2] = nwa; }
  if (t < nact) { actx[t] = act[t]; tgtx[t] = tgtK[t]; posof[act[t]] = t; }
  if (t >= nwa) return;
  int w = t, i = w / (K - 1), j = w % (K - 1) + 1, L = act[i];
  actx[nact + w] = ntab + w; tgtx[nact + w] = -1;
  wact[w] = 0; rbad[w] = 0; wbad[w] = 0; cmask[w] = 0; wlab[w] = L;
  for (int q = 0; q < K; q++) rrank[w * K + q] = 64;
  long long tt = tgtK[j * cap + i];
  if (tt < 0) return;
  int a = lock_arena(tt, aoff, na);
  long long c[3]; brook_unravel(tt - aoff[a], adim[3 * a], adim[3 * a + 1], c[0], c[1], c[2]);
  long long l[3];
  int wsd[3] = {K2_W1(ws), K2_W2(ws), K2_W3(ws)};
  for (int d = 0; d < 3; d++) {
    long long W = 1LL << wsd[d];
    long long o = c[d] - W / 2, hi = adim[3 * a + d] - W;
    if (o > hi) o = hi;
    if (o < 0) o = 0;
    worg[3 * w + d] = (int)o; l[d] = c[d] - o;
    soma_root[3 * (ntab + w) + d] = soma_root[3 * L + d] - o;
  }
  soma_r[ntab + w] = soma_r[L];
  whome[w] = a; wact[w] = 1;
  tgtx[nact + w] = (int)(wbase + ((long long)w << (wsd[0] + wsd[1] + wsd[2]))
                         + (l[0] | (l[1] << wsd[0]) | (l[2] << (wsd[0] + wsd[1]))));
}

// fill the windows: the object's voxels carry the pseudo-label and copies of its fields (the voxel
// graph's words too, when there is one), exits are rails, everything else is background
__global__ void k2_gather(const int* khdr, int ws, long long wbase, int ntab, const int* worg,
    const int* whome, const int* wlab, const int* wact, const long long* aoff, const long long* adim,
    LABEL_T* lab, unsigned char* alive, float* dbf, float* pdrf, unsigned int* vg)
{
  int w1 = K2_W1(ws), w2 = K2_W2(ws), w3 = K2_W3(ws);
  long long total = (long long)khdr[2] << (w1 + w2 + w3);
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < total; q += stride) {
    int w = (int)(q >> (w1 + w2 + w3));
    if (!wact[w]) continue;
    long long lx = q & ((1LL << w1) - 1), ly = (q >> w1) & ((1LL << w2) - 1), lz = (q >> (w1 + w2)) & ((1LL << w3) - 1);
    long long s = wbase + q;
    int a = whome[w];
    long long sx = adim[3 * a], sy = adim[3 * a + 1], sz = adim[3 * a + 2];
    long long x = worg[3 * w] + lx, y = worg[3 * w + 1] + ly, z = worg[3 * w + 2] + lz;
    long long v = aoff[a] + BROOK_IDX(x, y, z, sx, sy);
    if (x >= sx || y >= sy || z >= sz || lab[v] != (LABEL_T)wlab[w]) { lab[s] = 0; alive[s] = 0; if (vg) vg[s] = 0u; continue; }
    lab[s] = (LABEL_T)(ntab + w); alive[s] = alive[v]; dbf[s] = dbf[v];
    if (vg) vg[s] = vg[v];
    pdrf[s] = k2_exit(lx, ly, lz, x, y, z, ws, sx, sy, sz) ? 0.0f : pdrf[v];
  }
}

// walk lengths per draft; -1 = the draft is void (flood failed, or its entry or join is an exit);
// jhome: the join's u in the object's coordinates (x | y << 21 | z << 42; -1: none)
__global__ void k2_lens(const int* ih, int K, int cap, int capK, int ws, long long wbase, int ntab,
    int* tgtK, const int* plen, int* lenK, const unsigned long long* rail_best, const int* worg,
    const int* whome, const long long* aoff, const long long* adim, const float* pdrf,
    const int* ju, const int* jr, long long* jhome)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t % cap, j = t / cap, nact = ih[0];
  if (j >= K || i >= nact) return;
  int g = j * cap + i;
  if (j == 0) { lenK[g] = tgtK[g] < 0 ? 0 : plen[i]; return; }
  if (i >= capK) tgtK[g] = -1;                               // no window: no draft
  if (tgtK[g] < 0) { lenK[g] = 0; return; }
  int w = i * (K - 1) + (j - 1), n = plen[nact + w];
  jhome[w] = -1;
  if (n >= 0) {
    int ww; long long hx, hy, hz;
    long long e = (long long)(rail_best[ntab + w] & 0xFFFFFFFFULL);
    long long hv = k2_home(e, wbase, ws, worg, whome, aoff, adim, &ww, &hx, &hy, &hz);
    if (pdrf[hv] != 0.0f) n = -1;                            // not a rail of the object: an exit
  }
  if (n >= 0 && ju[w] >= 0 && jr[w] < 26) {
    int ww; long long hx, hy, hz;
    k2_home((long long)ju[w], wbase, ws, worg, whome, aoff, adim, &ww, &hx, &hy, &hz);
    int a = whome[w], ks = BROOK_D3INV[jr[w]];
    long long rail = aoff[a] + BROOK_IDX(hx + BROOK_NX[ks], hy + BROOK_NY[ks], hz + BROOK_NZ[ks], adim[3 * a], adim[3 * a + 1]);
    // an exit joined (k2_reset's wbad stops such a draft too): void, so that no live exit becomes a
    // source of the batch's invalidation
    if (pdrf[rail] != 0.0f) n = -1;
    else jhome[w] = hx | (hy << 21) | (hz << 42);
  }
  lenK[g] = n;
}

// the sources of the batch's invalidation: the walks of every draft that is not void
__global__ void k2_src_offsets(const int* ih, int K, int cap, const int* tgtK, const int* lenK,
    int* soffK, int* ihv)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int nact = ih[0], run = 0;
  for (int i = 0; i < nact; i++) {
    for (int j = 0; j < K; j++) {
      int g = j * cap + i;
      int ok = (tgtK[g] >= 0 && lenK[g] >= 0 && (j == 0 || lenK[i] >= 0)) ? 1 : 0;
      soffK[g] = ok ? run : -1;
      if (ok) run += lenK[g];
    }
  }
  ihv[0] = 0; ihv[1] = run; ihv[2] = 0; ihv[3] = 0;
}
__global__ void k2_src_copy(const int* act, const int* ih, int K, int cap, const int* lenK,
    const int* soffK, const int* walk, const long long* wseg, int ntab, int* src)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t % cap, j = t / cap;
  if (j >= K || i >= ih[0]) return;
  int g = j * cap + i, n = lenK[g];
  if (soffK[g] < 0) return;
  long long base = wseg[j == 0 ? act[i] : ntab + i * (K - 1) + (j - 1)];
  for (int q = 0; q < n; q++) src[soffK[g] + q] = walk[base + q];
}

// Route check (see above). One thread per path vertex of the batch, against every higher draft
// of its label. Runs on the floods' distances: before they are reset. A voxel graph gates the
// neighbour u as the flood would (the edge u -> v must be permitted for u to capture the rail).
// Join: a vertex v next to u_b whose dijkstra3d rank from u_b (push-gated) beats b's own join
// lowers rrank[b, v's draft].
__global__ void k2_route(const int* ihv, const int* src, const int* posof, int K, int cap, int capK,
    int ws, long long wbase, int ntab, const int* tgtK, const int* lenK, const int* worg, const int* whome,
    const long long* aoff, const long long* adim, int na, const LABEL_T* lab, const unsigned int* vg, int has_vg,
    const float* pdrf, const float* dist, const unsigned long long* rail_best, int* rbad,
    const int* ju, const int* jr, const long long* jhome, int* rrank)
{
  long long n = ihv[1];
  int w1 = K2_W1(ws), w2 = K2_W2(ws), w3 = K2_W3(ws);
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
    long long s = src[t], hv, hx, hy, hz;
    int i, ja, a;
    if (s >= wbase) {
      int w;
      hv = k2_home(s, wbase, ws, worg, whome, aoff, adim, &w, &hx, &hy, &hz);
      i = w / (K - 1); ja = w % (K - 1) + 1; a = whome[w];
    } else {
      hv = s; i = posof[(int)lab[s]]; ja = 0;
      a = lock_arena(s, aoff, na);
      brook_unravel(s - aoff[a], adim[3 * a], adim[3 * a + 1], hx, hy, hz);
    }
    if (i >= capK) continue;
    for (int jb = ja + 1; jb < K; jb++) {
      int gb = jb * cap + i;
      if (tgtK[gb] < 0) break;
      if (lenK[gb] < 0) continue;
      int wb = i * (K - 1) + (jb - 1);
      if ((rbad[wb] >> ja) & 1) continue;
      if ((long long)tgtK[gb] == hv) { atomicOr(&rbad[wb], 1 << ja); continue; }
      long long jh = jhome[wb];
      if (jh >= 0 && a == whome[wb]) {
        long long dx = hx - (jh & 0x1FFFFF), dy = hy - ((jh >> 21) & 0x1FFFFF), dz = hz - (jh >> 42);
        if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1 && dz >= -1 && dz <= 1 && (dx | dy | dz) != 0) {
          int kk = BROOK_KOF[(dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)], rk = BROOK_D3RANK[kk];
          if (rk < jr[wb] && brook_vg_ok_push(vg, has_vg, (long long)ju[wb], kk)) atomicMin(&rrank[wb * K + ja], rk);
        }
      }
      unsigned long long rb = rail_best[ntab + wb];
      int De = (int)(rb >> 32);
      int we; long long ex, ey, ez;
      long long eh = k2_home((long long)(rb & 0xFFFFFFFFULL), wbase, ws, worg, whome, aoff, adim, &we, &ex, &ey, &ez);
      long long ox = worg[3 * wb], oy = worg[3 * wb + 1], oz = worg[3 * wb + 2];
      long long b0 = wbase + ((long long)wb << (w1 + w2 + w3));
      for (int k = 0; k < 26; k++) {
        long long lx = hx + BROOK_NX[k] - ox, ly = hy + BROOK_NY[k] - oy, lz = hz + BROOK_NZ[k] - oz;
        if (lx < 0 || ly < 0 || lz < 0 || lx >= (1LL << w1) || ly >= (1LL << w2) || lz >= (1LL << w3)) continue;
        long long u = b0 + (lx | (ly << w1) | (lz << (w1 + w2)));
        if (lab[u] != (LABEL_T)(ntab + wb) || pdrf[u] == 0.0f) continue;
        if (!brook_vg_ok(vg, has_vg, u, k)) continue;
        int du = __float_as_int(dist[u]);
        if (du < De || (du == De && hv < eh)) { atomicOr(&rbad[wb], 1 << ja); break; }
      }
    }
  }
}

// Restore the railroad state over the touched voxels; on the way, an exit that the flood reached
// no later than its entry voids the draft (a route through it could tie with the entry).
__global__ void k2_reset(float* dist, int* inq, int* infar, const int* touched, const int* counts,
    int ws, long long wbase, int ntab, const int* worg, const int* whome, const long long* aoff,
    const long long* adim, const float* pdrf, const unsigned long long* rail_best, int* wbad)
{
  int n = counts[5];
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
    long long v = touched[t];
    if (v >= wbase && pdrf[v] == 0.0f) {
      int w; long long hx, hy, hz;
      long long hv = k2_home(v, wbase, ws, worg, whome, aoff, adim, &w, &hx, &hy, &hz);
      unsigned long long rb = rail_best[ntab + w];
      if (pdrf[hv] != 0.0f && rb != KD_NONE && __float_as_int(dist[v]) <= (int)(rb >> 32)) wbad[w] = 1;
    }
    dist[v] = BROOK_INF; atomicAnd(&inq[v], ~(BROOK_F_DQ | BROOK_F_DF));
  }
}

// Contact check, pass 1: every voxel a draft's invalidation removed gets that draft's BIT, in the
// object's own arena; a window draft that claimed an exit is void.
__global__ void k2_mark(const int* touched, const int* hdr, int K, int ws, long long wbase,
    const int* worg, const int* whome, const long long* aoff, const long long* adim, int* mark, int* wbad)
{
  int n = hdr[3];
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
    long long v = touched[t];
    if (v < wbase) { if (!(mark[v] & 1)) atomicOr(&mark[v], 1); continue; }
    int w; long long x, y, z;
    long long hv = k2_home(v, wbase, ws, worg, whome, aoff, adim, &w, &x, &y, &z);
    int val = 1 << (w % (K - 1) + 1), a = whome[w];
    atomicOr(&mark[hv], val);
    if (k2_exit(x - worg[3 * w], y - worg[3 * w + 1], z - worg[3 * w + 2], x, y, z, ws,
                adim[3 * a], adim[3 * a + 1], adim[3 * a + 2])) wbad[w] = 1;
  }
}
// pass 2: the drafts of its object that removed a voxel this draft removed too
__global__ void k2_contact(const int* touched, const int* hdr, int K, int ws, long long wbase,
    const int* worg, const int* whome, const long long* aoff, const long long* adim,
    const int* mark, int* cmask)
{
  int n = hdr[3];
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
    long long v = touched[t];
    if (v < wbase) continue;
    int w; long long x, y, z;
    long long hv = k2_home(v, wbase, ws, worg, whome, aoff, adim, &w, &x, &y, &z);
    int others = mark[hv] & ~(1 << (w % (K - 1) + 1));
    if (others & ~cmask[w]) atomicOr(&cmask[w], others);
  }
}

// Target check. death(v): 0 = dead before the batch or removed by draft 0 (already applied),
// jd = j + 1 = the LOWEST window draft j that removed it, K + 1 = alive. fbd[i, d] = first order position (from
// draft 0's) whose voxel has death d.
__global__ void kd_scan_offsets(const int* ih, int K, int cap, const int* tgtK, const int* lenK,
    const long long* dposK, long long* scoff, int* khdr)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int nact = ih[0]; long long run = 0;
  for (int i = 0; i < nact; i++) {
    scoff[i] = run;
    int last = 0;
    for (int j = 1; j < K; j++) { int g = j * cap + i; if (tgtK[g] < 0) break; if (lenK[g] >= 0) last = j; }
    if (last > 0 && dposK[i] >= 0) run += dposK[last * cap + i] - dposK[i] + 1;
  }
  scoff[nact] = run;
  khdr[1] = run > 0x7fffffffLL ? 0x7fffffff : (int)run;
}
__global__ void kd_scan_init(const int* ih, int K, unsigned long long* fbd)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < ih[0] * (K + 2)) fbd[t] = KD_NONE;
}
__global__ void k2_scan(const int* ih, const int* khdr, int K, const long long* scoff,
    const long long* dposK, const int* order, const unsigned char* alive, const int* mark,
    unsigned long long* fbd)
{
  int nact = ih[0]; long long total = khdr[1];
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    int lo = 0, hi = nact - 1;                                   // the label whose range holds t
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (scoff[mid] <= t) lo = mid; else hi = mid - 1; }
    long long pos = dposK[lo] + (t - scoff[lo]);
    int v = order[pos];
    int m = mark[v], d = K + 1;
    if (alive[v] == 0) d = 0;
    else if (m != 0) { d = 1; while (!(m & 1)) { m >>= 1; d++; } }
    unsigned long long* slot = fbd + (long long)lo * (K + 2) + d;
    if (*slot > (unsigned long long)pos) atomicMin(slot, (unsigned long long)pos);
  }
}
// acc[i]: retired drafts of the label (-1: its first flood failed, the label is ejected);
// kinfo[4i..]: the last retired draft, the draft that ended the batch (0: none), whether it ended
// on the TARGET rule (a survivor the drafter took for removed), the drafts skipped.
// The TARGET rule with a set A: the draft's own target must be alive after A (exact: its mark
// holds no bit of A), and no voxel before it may be: fbd knows a voxel by its LOWEST remover, so a
// voxel counts as alive unless that one is in A -- never fewer voxels than are really alive, so
// the first of them being the draft's target proves it (equal to the exact rule when A is a prefix).
// A retired draft joins the earliest rail in dijkstra3d order over its window's and the retired
// drafts' candidates (repick[g]: that rank, -1: its own join; the JOIN rule above).
// kstat: [2] void (flood / exit) [3] route (and a soma label's re-pick) [4] contact / invalidation exit
// [5] target [6] skipped [7] joins re-picked
__global__ void k2_accept(const int* ih, int K, int cap, const int* tgtK, const int* lenK,
    const long long* dposK, const unsigned long long* fbd, const int* rbad, const int* wbad,
    const int* cmask, const int* mark, int pool, int* acc, int* accept, int* kinfo, int* kstat,
    const double* soma_r, int ntab, const int* jr, const int* rrank, int* repick)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ih[0]) return;
  int a = 0, last = 0, stop = 0, miss = 0, skips = 0;
  for (int j = 0; j < K; j++) { accept[j * cap + i] = 0; repick[j * cap + i] = -1; }
  if (tgtK[i] >= 0) {
    a = lenK[i] < 0 ? -1 : 1;
    int A = 1;
    if (a > 0) accept[i] = 1;
    for (int j = 1; j < K && a > 0; j++) {
      int g = j * cap + i, w = i * (K - 1) + (j - 1);
      if (tgtK[g] < 0) break;
      if (mark[tgtK[g]] & A) {                               // a retired draft removed this target
        if (pool) { atomicAdd(&kstat[6], 1); skips++; continue; }
        atomicAdd(&kstat[5], 1); stop = j; break;
      }
      unsigned long long first = fbd[(long long)i * (K + 2) + K + 1];
      for (int d = 2; d <= K; d++) {
        if ((A >> (d - 1)) & 1) continue;
        unsigned long long f = fbd[(long long)i * (K + 2) + d]; if (f < first) first = f;
      }
      if (first != (unsigned long long)dposK[g]) { atomicAdd(&kstat[5], 1); stop = j; miss = 1; break; }
      if (lenK[g] < 0) { atomicAdd(&kstat[2], 1); stop = j; break; }
      if (rbad[w] & A) { atomicAdd(&kstat[3], 1); stop = j; break; }
      if (wbad[w] || (cmask[w] & A)) { atomicAdd(&kstat[4], 1); stop = j; break; }
      int best = jr[w];
      for (int q = 0; q < j; q++) if (((A >> q) & 1) && rrank[w * K + q] < best) best = rrank[w * K + q];
      if (best < jr[w]) {
        if (soma_r[ntab + w] >= 0.0) { atomicAdd(&kstat[3], 1); stop = j; break; }
        repick[g] = best; atomicAdd(&kstat[7], 1);
      }
      A |= 1 << j; accept[g] = 1; last = j; a++;
    }
  }
  acc[i] = a; kinfo[4 * i] = last; kinfo[4 * i + 1] = stop; kinfo[4 * i + 2] = miss; kinfo[4 * i + 3] = skips;
}

// store offsets of the accepted walks (label-major, drafts in order), capacity checks (PARKED)
__global__ void kd_offsets(int* ih, int K, int cap, const int* lenK, const int* acc,
    const int* accept, int* offK)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int nact = ih[0]; long long run = ih[2], nrec = ih[4];
  for (int i = 0; i < nact; i++) for (int j = 0; j < K && acc[i] > 0; j++) {
    int g = j * cap + i;
    if (!accept[g]) continue;
    offK[g] = (int)run; run += lenK[g] > 0 ? lenK[g] : 0; nrec++;
  }
  ih[3] = (int)(run - ih[2]);
  ih[5] = (run > (long long)ih[9]) ? 1 : (nrec > (long long)ih[10] ? 2 : 0);
}
// what the accepted window drafts removed dies in the object; the marks go
__global__ void k2_apply(const int* ih, const int* touched, const int* hdr, int K, int cap, int ws,
    long long wbase, const int* worg, const int* whome, const long long* aoff, const long long* adim,
    const int* accept, unsigned char* alive, int* mark)
{
  if (ih[5]) return;
  int n = hdr[3];
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
    long long v = touched[t];
    if (v < wbase) { mark[v] = 0; continue; }
    int w; long long x, y, z;
    long long hv = k2_home(v, wbase, ws, worg, whome, aoff, adim, &w, &x, &y, &z);
    if (accept[(w % (K - 1) + 1) * cap + w / (K - 1)]) alive[hv] = 0;
    mark[hv] = 0;
  }
}
// the accepted walks, as voxels of the object, reversed into the path store; they become rails.
// A re-picked join (k2_accept) replaces the walk's rail end (its first vertex): u + N[rank].
__global__ void k2_write(const int* act, const int* ih, int K, int cap, int ws, long long wbase,
    int ntab, const int* lenK, const int* offK, const int* accept, const int* walk, const long long* wseg,
    const int* worg, const int* whome, const long long* aoff, const long long* adim, int* store, float* pdrf,
    const long long* jhome, const int* repick)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t % cap, j = t / cap;
  if (j >= K || i >= ih[0] || ih[5]) return;
  int g = j * cap + i, n = lenK[g];
  if (!accept[g] || n <= 0) return;
  long long base = wseg[j == 0 ? act[i] : ntab + i * (K - 1) + (j - 1)];
  for (int q = 0; q < n; q++) {
    long long v = walk[base + q];
    if (j > 0) { int w; long long x, y, z; v = k2_home(v, wbase, ws, worg, whome, aoff, adim, &w, &x, &y, &z); }
    if (j > 0 && q == 0 && repick[g] >= 0) {
      int wn = i * (K - 1) + j - 1, a = whome[wn], ks = BROOK_D3INV[repick[g]];
      long long h = jhome[wn];
      v = aoff[a] + BROOK_IDX((h & 0x1FFFFF) + BROOK_NX[ks], ((h >> 21) & 0x1FFFFF) + BROOK_NY[ks],
                              (h >> 42) + BROOK_NZ[ks], adim[3 * a], adim[3 * a + 1]);
    }
    store[offK[g] + (n - 1 - q)] = (int)v;
    pdrf[v] = 0.0f;
  }
}
__global__ void kd_advance(int* act, int K, int cap, const int* tgtK, const int* lenK, const int* offK,
    const long long* dposK, const int* acc, int* rec_lab, int* rec_len, int* rec_start, int* ejected,
    int* pcount, int* plast_len, long long* cursor, const int* counts, int has_counts, int* ih, int* kstat,
    unsigned long long handle, int* lane_stop, int* depth, int adapt, const int* accept, const int* kinfo)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int go = 0;
  if (ih[5] == 0) {
    int nact = ih[0], w = 0, nrec = ih[4];
    for (int i = 0; i < nact; i++) {
      int L = act[i], a = acc[i];
      plast_len[L] = 0;
      if (tgtK[i] < 0) { ih[6] += 1; continue; }
      if (a < 0) { ejected[ih[8]] = L; ih[8] += 1; continue; }
      // The horizon of a label (schedule only): a draft behind a rejected one is work the batch
      // waits for and throws away, so a label drafts one more than it last got accepted, and one
      // more again whenever everything it drafted was (additive increase up to K, reset on a
      // rejection). What is drafted never changes what is accepted: the checks do not know it.
      if (adapt) {
        int nd = 0;
        for (int j = 0; j < K; j++) if (tgtK[j * cap + i] >= 0) nd++;
        int stop = kinfo[4 * i + 1];
        if (stop > 0) depth[L] = stop + 1 > 2 ? stop + 1 : 2;
        else if (nd >= depth[L] && depth[L] < K) depth[L] += 1;
      }
      for (int j = 0; j < K; j++) {
        int g = j * cap + i;
        if (!accept[g]) continue;
        rec_lab[nrec] = L; rec_len[nrec] = lenK[g]; rec_start[nrec] = offK[g]; nrec++;
      }
      for (int j = 0; j < K; j++) if (tgtK[j * cap + i] >= 0) kstat[0] += 1;       // drafted
      kstat[1] += a;                                                               // accepted
      pcount[L] += a;
      if (a > 1 && dposK[kinfo[4 * i] * cap + i] >= 0) cursor[L] = dposK[kinfo[4 * i] * cap + i];
      act[w++] = L;
    }
    ih[7] += nrec - ih[4]; ih[4] = nrec;
    ih[14] = ih[2]; ih[15] = 0;                       // nothing pending: the batch invalidated itself
    ih[0] = w; ih[1] += 1; ih[2] += ih[3]; ih[3] = 0;
    if (has_counts) { ih[12] += counts[6]; ih[13] += counts[7]; }
    ih[11] -= 1;
    go = (w > 0 && ih[11] > 0) ? 1 : 0;
  }
  if (lane_stop[0]) go = 0;                           // (see lock_advance)
  if (!go) lane_stop[0] = 1;
  if (handle) cudaGraphSetConditional(handle, (unsigned int)go);
}

// Restore the railroad state's clean values over the voxels the last flood touched.
__global__ void reset_touched(
    float* dist, int* inq, int* infar, const int* touched, const int* counts, const int* ih)
{
  if (ih[5]) return;                       // parked: pass 1 of the backtrace still needs dist
  int n = counts[5];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int v = touched[i];
    dist[v] = BROOK_INF; atomicAnd(&inq[v], ~(BROOK_F_DQ | BROOK_F_DF));
  }
}
