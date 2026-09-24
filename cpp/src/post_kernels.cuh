// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// Follows Kimimaro 5.8.1 remove_loops, join_close_components and remove_ticks (kimimaro/post.py; cpp/licenses/Kimimaro.txt).
#pragma once
#include "common.cuh"
namespace brook::cuda::post {

// per component c: vertices vs[c]..vs[c+1] (local ids 0..nv-1, ascending = the label's order),
// edge rows es[c]..es[c+1] in `edges` (in: the component's rows; out: ne[c] surviving rows).
// Incremental state, so that an iteration costs its depth-first search and the cycle it found, not
// a dozen passes over the component (e.g. a 38k-vertex component with 41 cycles):
//   rows      tombstoned in place (row[0] = -1), new rows appended, `first` = first live row
//   neighbour sets as doubly linked lists in insertion order (entries from a free pool); an entry
//             knows its first row and how many rows repeat it (kimimaro's arrays may hold
//             duplicate rows: they count for the branch test and go together)
//   cnt       rows touching a node, kept up to date
// visited / mark / succ are reset for the nodes that were touched only.
// Dangling trees are skipped. A neighbour c of u whose side of the edge is a tree can never close a
// cycle, is pushed by u alone (so it is never popped as "already visited"), and leaves the rest of
// the stack and the path as they were: not pushing it changes nothing the search reports. Leaves
// are stripped incrementally (SKP on the entry parent -> leaf); an edge added to a stripped node
// re-attaches its chain up to the core first.
// The search then walks only the neighbourhood of the remaining cycles, not the whole skeleton.
__global__ void remove_loops_k(
    int ncomp, const long long* vs, const long long* es, int* ne,
    const float* xyz, const float* radii, int* edges,
    int* rows, int* ent, int* nodes, int* stk, int* path)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncomp) return;
  long long v0 = vs[c], e0 = es[c];
  int nv = (int)(vs[c + 1] - v0), E0 = (int)(es[c + 1] - e0), cap = 2 * E0;
  if (E0 == 0) return;
  const float* P = xyz + 3 * v0; const float* R = radii + v0;
  int* ROW = rows + 4 * e0;                               // cap rows
  int* NBR = ent + 12 * e0, *NXT = NBR + cap, *PRV = NXT + cap, *RID = PRV + cap, *MUL = RID + cap, *SKP = MUL + cap;
  int* head = nodes + 11 * v0, *tail = head + nv, *cnt = tail + nv, *visited = cnt + nv,
     *succ = visited + nv, *mark = succ + nv, *vlist = mark + nv,
     *cdeg = vlist + nv, *spar = cdeg + nv, *queue = spar + nv, *inq = queue + nv;   // core degree, strip parent (-2 = in the core), work queue (each node at most once)
  int* S = stk + 3 * (2 * e0 + c);
  int* PATH = path + v0 + c;

  for (int v = 0; v < nv; v++) { head[v] = -1; tail[v] = -1; cnt[v] = 0; visited[v] = 0; succ[v] = -1; mark[v] = 0; cdeg[v] = 0; spar[v] = -2; inq[v] = 0; }
  int qn = 0;
  #define PUSHQ(x) { if (!inq[x]) { inq[x] = 1; queue[qn++] = (x); } }
  int nfree = 0, used = 0;                                 // entries: a bump pointer, then a free list in NXT
  int freehead = -1;
  int nrows = 0, nlive = 0, first = 0, app0 = 0, app1 = 0;

  // ---- helpers as macros over the local state ----
  #define FIND(a, b, out) { out = -1; for (int _e = head[a]; _e >= 0; _e = NXT[_e]) if (NBR[_e] == (b)) { out = _e; break; } }
  #define NEWENT(out) { if (freehead >= 0) { out = freehead; freehead = NXT[freehead]; } else { out = used++; } }
  #define LINK(a, b, rid) { int _n; NEWENT(_n); NBR[_n] = (b); RID[_n] = (rid); MUL[_n] = 1; SKP[_n] = 0; NXT[_n] = -1; PRV[_n] = tail[a]; \
                            if (tail[a] >= 0) NXT[tail[a]] = _n; else head[a] = _n; tail[a] = _n; }
  #define UNLINK(a, e) { if (PRV[e] >= 0) NXT[PRV[e]] = NXT[e]; else head[a] = NXT[e]; \
                         if (NXT[e] >= 0) PRV[NXT[e]] = PRV[e]; else tail[a] = PRV[e]; NXT[e] = freehead; freehead = e; }
  #define ADDROW(x, y) { int _r = nrows++; ROW[2 * _r] = (x); ROW[2 * _r + 1] = (y); nlive++; cnt[x]++; cnt[y]++; \
                         int _f; FIND(x, y, _f); \
                         if (_f >= 0) { MUL[_f]++; int _g; FIND(y, x, _g); MUL[_g]++; } \
                         else { LINK(x, y, _r); LINK(y, x, _r); cdeg[x]++; cdeg[y]++; } }
  // re-attach a stripped node and its chain of strip parents to the core
  #define ATTACH(x0) { int _x = (x0); while (spar[_x] != -2) { int _u = spar[_x]; spar[_x] = -2; \
                         if (_u >= 0) { int _f; FIND(_u, _x, _f); if (_f >= 0) SKP[_f] = 0; cdeg[_u]++; int _g; FIND(_x, _u, _g); cdeg[_x]++; (void)_g; } \
                         if (_u < 0) break; _x = _u; } }
  // strip leaves from the queue: the entry parent -> leaf is skipped by the search from now on
  #define STRIP() { while (qn > 0) { int _c = queue[--qn]; inq[_c] = 0; if (spar[_c] != -2 || cdeg[_c] > 1) continue; \
                      int _u = -1; for (int _e = head[_c]; _e >= 0; _e = NXT[_e]) if (spar[NBR[_e]] == -2) { _u = NBR[_e]; break; } \
                      spar[_c] = _u; cdeg[_c] = 0; \
                      if (_u >= 0) { int _f; FIND(_u, _c, _f); SKP[_f] = 1; cdeg[_u]--; if (cdeg[_u] <= 1) PUSHQ(_u); } } }

  for (int i = 0; i < E0; i++) { int x = edges[2 * (e0 + i)], y = edges[2 * (e0 + i) + 1]; ADDROW(x, y); }
  app0 = app1 = nrows;
  for (int v = 0; v < nv; v++) if (cdeg[v] <= 1) PUSHQ(v);
  STRIP();

  for (;;) {
    if (nlive == 0) break;
    while (ROW[2 * first] < 0) first++;
    // ---- _find_cycle ----
    int sp = 1, plen = 0, node = -1, parent = -1, depth = 0, found = 0, nvis = 0;
    S[0] = ROW[2 * first]; S[1] = -1; S[2] = 0;
    while (sp > 0) {
      sp--; node = S[3 * sp]; parent = S[3 * sp + 1]; depth = S[3 * sp + 2];
      if (plen > depth) plen = depth;
      PATH[plen++] = node;
      if (visited[node]) { found = 1; break; }
      visited[node] = 1; vlist[nvis++] = node;
      for (int e = head[node]; e >= 0; e = NXT[e]) {
        int child = NBR[e];
        if (child == parent || SKP[e]) continue;
        S[3 * sp] = child; S[3 * sp + 1] = node; S[3 * sp + 2] = depth + 1; sp++;
      }
    }
    for (int j = 0; j < nvis; j++) visited[vlist[j]] = 0;
    if (!found || plen <= 1) break;
    int i0 = 0;
    for (i0 = 0; i0 < plen - 1; i0++) if (PATH[i0] == node) break;
    if (plen - i0 < 3) break;                                   // (incl. the stale-entry case)
    const int* CY = PATH + i0; int cl = plen - i0;               // CY[0] == CY[cl - 1]; cl - 1 edges
    int nb = 0;
    for (int j = 0; j + 1 < cl; j++) { int v = CY[j]; mark[v] = 1; succ[v] = CY[j + 1]; if (cnt[v] >= 3) vlist[nb++] = v; }
    for (int i = 1; i < nb; i++) {                               // branch nodes ascending (few)
      int x = vlist[i], j = i - 1;
      while (j >= 0 && vlist[j] > x) { vlist[j + 1] = vlist[j]; j--; }
      vlist[j + 1] = x;
    }
    int add_a = -1, add_b = -1, add_many = 0, inter = -1;
    if (nb == 0) {
      for (int j = 0; j + 1 < cl; j++) mark[CY[j]] |= 2;
    } else if (nb == 1) {
      int bn = vlist[0], end = -1; float best = -1.0f;
      for (int j = 0; j + 1 < cl; j++) {                          // first maximum in ascending node order
        int v = CY[j];
        float dx = P[3 * v] - P[3 * bn], dy = P[3 * v + 1] - P[3 * bn + 1], dz = P[3 * v + 2] - P[3 * bn + 2];
        float d = (dx * dx + dy * dy) + dz * dz;
        if (end < 0 || d > best || (d == best && v < end)) { best = d; end = v; }
      }
      for (int j = 0; j + 1 < cl; j++) mark[CY[j]] |= 2;
      add_a = bn; add_b = end;
    } else if (nb == 2) {
      int L = cl - 1, p0 = -1, p1 = -1;                           // path = CY[1..]; the two branch positions
      for (int j = 0; j < L; j++) { int v = CY[j + 1]; if (cnt[v] >= 3) { if (p0 < 0) p0 = j; else p1 = j; } }
      int inner = ((double)(p1 - p0) < (double)L / 2.0);          // the side with fewer nodes stays
      for (int j = 0; j + 1 < cl; j++) {
        int lo = j - 1, hi = j;                                   // cycle edge j joins path[j-1] and path[j]; j == 0: the junction
        int on_inner = (lo >= 0 && lo >= p0 && hi <= p1);
        int on_short = inner ? on_inner : !on_inner;
        if (!on_short) mark[CY[j]] |= 2;
      }
    } else {
      float cx = 0.0f, cy = 0.0f, cz = 0.0f;
      for (int i = 0; i < nb; i++) { int v = vlist[i]; cx += P[3 * v]; cy += P[3 * v + 1]; cz += P[3 * v + 2]; }
      cx /= (float)nb; cy /= (float)nb; cz /= (float)nb;
      float best = 0.0f;
      for (int v = 0; v < nv; v++) {                              // All nodes of the component, first minimum
        float dx = P[3 * v] - cx, dy = P[3 * v + 1] - cy, dz = P[3 * v + 2] - cz;
        float d = (dx * dx + dy * dy) + dz * dz;
        if (inter < 0 || d < best) { best = d; inter = v; }
      }
      float far = 0.0f;
      for (int i = 0; i < nb; i++) {
        int v = vlist[i];
        float dx = P[3 * v] - P[3 * inter], dy = P[3 * v + 1] - P[3 * inter + 1], dz = P[3 * v + 2] - P[3 * inter + 2];
        float d = (dx * dx + dy * dy) + dz * dz;
        if (i == 0 || d > far) far = d;
      }
      if (sqrtf(far) > R[inter]) mark[CY[0]] |= 2;                // a tiny snip: the cycle's first edge only
      else { for (int j = 0; j + 1 < cl; j++) mark[CY[j]] |= 2; add_many = 1; }
    }
    // ---- remove_row: every row becomes (lo, hi) (only last iteration's new rows can be unsorted);
    //      the flagged cycle edges go with all their duplicates, the others keep their order ----
    for (int r = app0; r < app1; r++) if (ROW[2 * r] > ROW[2 * r + 1]) { int t = ROW[2 * r]; ROW[2 * r] = ROW[2 * r + 1]; ROW[2 * r + 1] = t; }
    for (int j = 0; j + 1 < cl; j++) {
      int a = CY[j], b = CY[j + 1];
      if (!(mark[a] & 2)) continue;
      int e; FIND(a, b, e);
      if (e < 0) continue;
      int m = MUL[e];
      if (m == 1) { ROW[2 * RID[e]] = -1; }
      else {
        int lo = a < b ? a : b, hi = a < b ? b : a;
        for (int r = first; r < nrows; r++) if (ROW[2 * r] == lo && ROW[2 * r + 1] == hi) ROW[2 * r] = -1;
      }
      nlive -= m; cnt[a] -= m; cnt[b] -= m;
      int g; FIND(b, a, g);
      UNLINK(a, e); UNLINK(b, g);
      cdeg[a]--; cdeg[b]--;
      if (cdeg[a] <= 1) PUSHQ(a);
      if (cdeg[b] <= 1) PUSHQ(b);
    }
    // the branch nodes to reconnect were taken BEFORE the removal changed the counts
    if (nrows + nb + 1 > cap) {                                   // compact the rows (rare), keep their order
      int w = 0;
      for (int r = 0; r < nrows; r++) if (ROW[2 * r] >= 0) { ROW[2 * w] = ROW[2 * r]; ROW[2 * w + 1] = ROW[2 * r + 1]; w++; }
      nrows = w; first = 0;
      for (int v = 0; v < nv; v++) for (int e = head[v]; e >= 0; e = NXT[e]) RID[e] = -1;
      for (int r = 0; r < nrows; r++) {
        int f; FIND(ROW[2 * r], ROW[2 * r + 1], f); if (RID[f] < 0) { RID[f] = r; int g2; FIND(ROW[2 * r + 1], ROW[2 * r], g2); RID[g2] = r; }
      }
    }
    app0 = nrows;
    if (add_b >= 0) { ATTACH(add_a); ATTACH(add_b); ADDROW(add_a, add_b); }
    if (add_many) { ATTACH(inter); for (int i = 0; i < nb; i++) { int v = vlist[i]; if (v != inter) { ATTACH(v); ADDROW(v, inter); } } }
    app1 = nrows;
    STRIP();
    for (int j = 0; j + 1 < cl; j++) { mark[CY[j]] = 0; succ[CY[j]] = -1; }
  }
  int w = 0;
  for (int r = 0; r < nrows; r++) if (ROW[2 * r] >= 0) { edges[2 * (e0 + w)] = ROW[2 * r]; edges[2 * (e0 + w) + 1] = ROW[2 * r + 1]; w++; }
  ne[c] = w;
}


__device__ __forceinline__ int cell_cmp(int la, long long xa, long long ya, long long za,
                                        int lb, long long xb, long long yb, long long zb) {
  if (la != lb) return la < lb ? -1 : 1;
  if (xa != xb) return xa < xb ? -1 : 1;
  if (ya != yb) return ya < yb ? -1 : 1;
  if (za != zb) return za < zb ? -1 : 1;
  return 0;
}
#define JSLOTS 8
// mode 0: count the candidates of vertex i into counts[i]; mode 1: write them from offs[i] on.
// A candidate = (component of i, a HIGHER component in reach, exact distance, the two vertices).
__global__ void join_scan(
    int n, int mode, const int* kl, const long long* kx, const long long* ky, const long long* kz,
    const float* xyz, const long long* gid, const long long* comp, const double* bound,
    long long* counts, const long long* offs,
    long long* oA, long long* oB, double* oD, long long* oV, long long* oU)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int L = kl[i];
  double bnd = bound[L];
  float b32 = (float)bnd; b32 = b32 * b32 * 1.0001f + 1e-30f;      // float32 only rejects
  long long sB[JSLOTS]; double sD[JSLOTS]; long long sU[JSLOTS]; int ns = 0;
  long long emitted = 0, base = mode ? offs[i] : 0;
  float px = xyz[3 * i], py = xyz[3 * i + 1], pz = xyz[3 * i + 2];
  for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
    long long cx = kx[i] + dx, cy = ky[i] + dy, cz = kz[i] + dz;
    int lo = 0, hi = n;                                            // lower bound of the cell
    while (lo < hi) { int mid = (lo + hi) >> 1; if (cell_cmp(kl[mid], kx[mid], ky[mid], kz[mid], L, cx, cy, cz) < 0) lo = mid + 1; else hi = mid; }
    for (int j = lo; j < n && cell_cmp(kl[j], kx[j], ky[j], kz[j], L, cx, cy, cz) == 0; j++) {
      if (comp[j] <= comp[i]) continue;                            // every unordered pair once
      float fx = xyz[3 * j] - px, fy = xyz[3 * j + 1] - py, fz = xyz[3 * j + 2] - pz;
      if ((fx * fx + fy * fy) + fz * fz > b32) continue;
      double ex = (double)xyz[3 * j] - (double)px, ey = (double)xyz[3 * j + 1] - (double)py, ez = (double)xyz[3 * j + 2] - (double)pz;
      double d = sqrt((ex * ex + ey * ey) + ez * ez);
      if (!(d < bnd)) continue;
      int s = -1;
      for (int t = 0; t < ns; t++) if (sB[t] == comp[j]) { s = t; break; }
      if (s >= 0) {                                                // same vertex i: ties by the other vertex's id
        if (d < sD[s] || (d == sD[s] && gid[j] < gid[sU[s]])) { sD[s] = d; sU[s] = j; }
      } else if (ns < JSLOTS) { sB[ns] = comp[j]; sD[ns] = d; sU[ns] = j; ns++; }
      else {                                                       // no slot left: straight to the list
        if (mode) { long long o = base + emitted; oA[o] = comp[i]; oB[o] = comp[j]; oD[o] = d; oV[o] = i; oU[o] = j; }
        emitted++;
      }
    }
  }
  for (int t = 0; t < ns; t++) {
    if (mode) { long long o = base + emitted; oA[o] = comp[i]; oB[o] = sB[t]; oD[o] = sD[t]; oV[o] = i; oU[o] = sU[t]; }
    emitted++;
  }
  if (!mode) counts[i] = emitted;
}

// One thread per label: kimimaro's greedy fusion on the label's pair list [ps, pe).
// cx / cy: local component indices (order of the components = list order), rewritten as clusters
// fuse (the fused cluster keeps the id of cx); pos: list position; ok: the radius test of the pair.
__global__ void join_fuse(
    int nlab, const long long* ps, const long long* cs, const double* radius,
    int* cx, int* cy, const double* dd, const long long* va, const long long* vb,
    const long long* gid, const float* rad, int* alive, int* pos, int* slot,
    long long* out_a, long long* out_b, int* nout)
{
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nlab) return;
  long long p0 = ps[l], p1 = ps[l + 1], c0 = cs[l];
  int nc = (int)(cs[l + 1] - c0);
  int* POS = pos + c0; int* SLOT = slot + c0;
  for (int k = 0; k < nc; k++) { POS[k] = k; SLOT[k] = -1; }
  int fusions = 0, made = 0;
  for (;;) {
    long long best = -1; float bd = 0.0f; int bi = 0, bj = 0;
    for (long long p = p0; p < p1; p++) {
      if (!alive[p]) continue;
      double d = dd[p];
      if (d > (double)(float)(rad[va[p]] + rad[vb[p]])) continue;      // the nearest pair fails: blocked
      float d32 = (float)d;
      int ia = POS[cx[p]], ib = POS[cy[p]];
      if (ia > ib) { int t = ia; ia = ib; ib = t; }
      if (best < 0 || d32 < bd || (d32 == bd && (ia < bi || (ia == bi && ib < bj)))) { best = p; bd = d32; bi = ia; bj = ib; }
    }
    if (best < 0 || (double)bd > radius[l]) break;
    int X = cx[best], Y = cy[best];
    out_a[c0 + made] = va[best]; out_b[c0 + made] = vb[best]; made++;
    alive[best] = 0;
    fusions++;
    // Y joins X: relabel, then keep for every other cluster the nearer of the two pairs
    for (long long p = p0; p < p1; p++) {
      if (!alive[p]) continue;
      if (cx[p] == Y) cx[p] = X;
      if (cy[p] == Y) cy[p] = X;
      if (cx[p] == cy[p]) { alive[p] = 0; continue; }
      if (cx[p] != X && cy[p] != X) continue;
      int k = (cx[p] == X) ? cy[p] : cx[p];
      long long q = SLOT[k];
      if (q < 0) { SLOT[k] = (int)(p - p0); continue; }
      q += p0;
      long long lo_p = gid[va[p]] < gid[vb[p]] ? gid[va[p]] : gid[vb[p]], hi_p = gid[va[p]] < gid[vb[p]] ? gid[vb[p]] : gid[va[p]];
      long long lo_q = gid[va[q]] < gid[vb[q]] ? gid[va[q]] : gid[vb[q]], hi_q = gid[va[q]] < gid[vb[q]] ? gid[vb[q]] : gid[va[q]];
      int p_wins = dd[p] < dd[q] || (dd[p] == dd[q] && (lo_p < lo_q || (lo_p == lo_q && hi_p < hi_q)));
      if (p_wins) { alive[q] = 0; SLOT[k] = (int)(p - p0); } else alive[p] = 0;
    }
    for (long long p = p0; p < p1; p++) if (alive[p]) { SLOT[cx[p]] = -1; SLOT[cy[p]] = -1; }
    POS[X] = -fusions;                                              // the newest fusion heads the list
  }
  nout[l] = made;
}


__global__ void remove_ticks_k(
    int ncomp, const long long* vs, const long long* es, double threshold,
    const float* xyz, const long long* csr_off, const int* csr_dst, const int* csr_row,
    int* stk, float* stk_d, int* bc, int* su, int* sv, double* slen, int* sflag,
    int* chain_first, int* chain_last, int* chain_next, int* chain_dead, int* chain_of_edge, int* keep,
    int* hx, double* hlen)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncomp) return;
  long long v0 = vs[c], e0 = es[c];
  int nv = (int)(vs[c + 1] - v0), ne = (int)(es[c + 1] - e0);
  const float* P = xyz + 3 * v0;
  const long long* OFF = csr_off + v0;                       // absolute offsets into csr_dst / csr_row (local ids / rows)
  int* S = stk + 4 * v0; float* SD = stk_d + v0;             // (node, parent, root, chain) + running length
  int* BC = bc + v0;
  int* SU = su + v0, *SV = sv + v0, *SF = sflag + v0; double* SL = slen + v0;   // sflag: bit 0 alive, bit 1 eligible
  int* CF = chain_first + v0, *CL = chain_last + v0, *CN = chain_next + v0, *CD = chain_dead + v0;
  int* COE = chain_of_edge + e0; int* KEEP = keep + e0;
  // The shortest eligible super-edge comes from a binary HEAP keyed (length, lo, hi); an entry carries its
  // key and the super-edge's VERSION (a fused edge is a new version: stale entries are dropped when they
  // surface). The two super-edges left at a worn-down branch point come from per-node lists of SLOTS
  // (slot 2s / 2s+1 = super-edge s at its two ends; a slot's owner is rewritten when its edge is absorbed).
  // Linear scans are quadratic for components with many ticks.
  int* HID = hx + 14 * v0, *HVER = HID + 2 * nv, *HLO = HVER + 2 * nv, *HHI = HLO + 2 * nv,
     *VER = HHI + 2 * nv, *OWN = VER + nv, *SNX = OWN + 2 * nv, *IH = SNX + 2 * nv;
  double* HL = hlen + 2 * v0;
  int hn = 0;
  #define DEG(v) ((int)(OFF[(v) + 1] - OFF[(v)]))
  #define HLESS(i, j) (HL[i] < HL[j] || (HL[i] == HL[j] && (HLO[i] < HLO[j] || (HLO[i] == HLO[j] && HHI[i] < HHI[j]))))
  #define HSWAP(i, j) { double _d = HL[i]; HL[i] = HL[j]; HL[j] = _d; int _t; \
                        _t = HID[i]; HID[i] = HID[j]; HID[j] = _t; _t = HVER[i]; HVER[i] = HVER[j]; HVER[j] = _t; \
                        _t = HLO[i]; HLO[i] = HLO[j]; HLO[j] = _t; _t = HHI[i]; HHI[i] = HHI[j]; HHI[j] = _t; }
  #define HPUSH(s_) { int _i = hn++; HID[_i] = (s_); HVER[_i] = VER[s_]; HL[_i] = SL[s_]; \
                      HLO[_i] = SU[s_] < SV[s_] ? SU[s_] : SV[s_]; HHI[_i] = SU[s_] < SV[s_] ? SV[s_] : SU[s_]; \
                      while (_i > 0) { int _p = (_i - 1) >> 1; if (!HLESS(_i, _p)) break; HSWAP(_i, _p); _i = _p; } }
  #define HPOP() { hn--; if (hn > 0) { HSWAP(0, hn); int _i = 0; for (;;) { int _l = 2 * _i + 1, _r = _l + 1, _m = _i; \
                      if (_l < hn && HLESS(_l, _m)) _m = _l; if (_r < hn && HLESS(_r, _m)) _m = _r; \
                      if (_m == _i) break; HSWAP(_i, _m); _i = _m; } } }
  int start = -1;
  for (int v = 0; v < nv; v++) { int d = DEG(v); BC[v] = d >= 3 ? d : 0; IH[v] = -1; if (start < 0 && d == 1) start = v; }
  if (start < 0) return;
  // ---- the distance super-graph ----
  int ns = 0, sp = 0;
  S[0] = start; S[1] = -1; S[2] = start; S[3] = -1; SD[0] = 0.0f; sp = 1;
  while (sp > 0) {
    sp--;
    int node = S[4 * sp], parent = S[4 * sp + 1], root = S[4 * sp + 2], chain = S[4 * sp + 3];
    float dist = SD[sp];
    int d = DEG(node);
    if ((d == 1 || d >= 3) && node != root) {
      SU[chain] = root; SV[chain] = node; SL[chain] = (double)dist; VER[chain] = 0;
      SF[chain] = 1 | ((DEG(root) == 1 || d == 1) ? 2 : 0);
      OWN[2 * chain] = chain; SNX[2 * chain] = IH[root]; IH[root] = 2 * chain;
      OWN[2 * chain + 1] = chain; SNX[2 * chain + 1] = IH[node]; IH[node] = 2 * chain + 1;
      if (SF[chain] & 2) HPUSH(chain);
      dist = 0.0f; root = node;
    }
    for (long long q = OFF[node]; q < OFF[node + 1]; q++) {
      int child = csr_dst[q];
      if (child == parent) continue;
      float dx = P[3 * node] - P[3 * child], dy = P[3 * node + 1] - P[3 * child + 1], dz = P[3 * node + 2] - P[3 * child + 2];
      dx *= dx; dy *= dy; dz *= dz;
      int ch = chain;
      if (node == root) { ch = ns++; CF[ch] = ch; CL[ch] = ch; CN[ch] = -1; CD[ch] = 0; SF[ch] = 0; }   // a new chain leaves a critical point
      COE[csr_row[q]] = ch;
      S[4 * sp] = child; S[4 * sp + 1] = node; S[4 * sp + 2] = root; S[4 * sp + 3] = ch; SD[sp] = dist + sqrtf((dx + dy) + dz); sp++;
    }
  }
  int nalive = 0; for (int s = 0; s < ns; s++) if (SF[s] & 1) nalive++;
  // ---- kimimaro's loop ----
  while (nalive > 1) {
    while (hn > 0 && (!(SF[HID[0]] & 1) || VER[HID[0]] != HVER[0])) HPOP();
    if (hn == 0) break;
    int best = HID[0]; double bl = SL[best];
    int e1 = SU[best], e2 = SV[best];
    if (BC[e1] == 1 && BC[e2] == 1) break;
    if (bl >= threshold) break;
    HPOP();
    for (int ch = CF[best]; ch >= 0; ch = CN[ch]) CD[ch] = 1;
    SF[best] = 0; nalive--;
    BC[e1] -= 1; BC[e2] -= 1;
    for (int t = 0; t < 2; t++) {
      int v = t ? e2 : e1;
      if (BC[v] != 2) continue;
      int a = -1, b = -1;
      for (int k = IH[v]; k >= 0; k = SNX[k]) {
        int o = OWN[k];
        if (!(SF[o] & 1) || (SU[o] != v && SV[o] != v) || o == a) continue;
        if (a < 0) a = o; else { b = o; break; }
      }
      if (a < 0 || b < 0) { BC[v] = 0; continue; }
      int ea = SU[a] == v ? SV[a] : SU[a], eb = SU[b] == v ? SV[b] : SU[b];
      SL[a] = (0.0 + SL[a]) + SL[b];
      SU[a] = ea; SV[a] = eb; SF[a] = 3;                         // every fused edge becomes eligible
      CN[CL[a]] = CF[b]; CL[a] = CL[b];
      SF[b] = 0; nalive--;
      for (int k = IH[eb]; k >= 0; k = SNX[k]) if (OWN[k] == b) OWN[k] = a;   // b's slot at its far end now means a
      VER[a] += 1; HPUSH(a);
      BC[v] = 0;
    }
  }
  for (int r = 0; r < ne; r++) KEEP[r] = !CD[COE[r]];
}

}
