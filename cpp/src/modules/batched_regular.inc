// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti

#define BROOK_INF __int_as_float(0x7f800000)

// Per-label maximum of a non-negative field, packed with the voxel: key = value bits << 32 |
// (0xFFFFFFFF - flat) so the atomicMax picks the largest value and, on ties, the LOWEST flat
// index (the per-component rule). Non-finite values and label 0 are skipped.
__global__ void seg_argmax(
    const float* val, const LABEL_T* lab, long long n, unsigned long long* best)
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  LABEL_T L = lab[i];
  if (L == 0) return;
  float v = val[i];
  if (!(v < BROOK_INF)) return;
  unsigned long long key = ((unsigned long long)(unsigned int)__float_as_int(v) << 32)
                           | (unsigned long long)(0xFFFFFFFFu - (unsigned int)i);
  if (__ldcg(&best[L]) >= key) return;                   // cheap pre-check, then the atomic
  atomicMax(&best[L], key);
}

// Neighbour token: bit k = neighbour k (BROOK_N* order) is inside the volume, has my label and,
// with a voxel graph, the step u -> u + off[k] is permitted (the push gate of the single-component
// floods, brook_vg_ok_push: bit BROOK_BIT[k] of vg[u]). The floods read only this token, so a
// graph gates them exactly as it gates sssp_coop.
__global__ void nbr_mask(
    const LABEL_T* lab, const unsigned int* vg, long long sx, long long sy, long long sz, unsigned int* mask)
{
  long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= sx * sy * sz) return;
  LABEL_T L = lab[u];
  unsigned int m = 0;
  if (L != 0) {
    unsigned int edges = vg ? vg[u] : 0xFFFFFFFFu;
    long long x, y, z; brook_unravel(u, sx, sy, x, y, z);
    for (int k = 0; k < 26; k++) {
      if (!((edges >> BROOK_BIT[k]) & 1u)) continue;
      long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
      if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
      if (lab[BROOK_IDX(nx, ny, nz, sx, sy)] == L) m |= 1u << k;
    }
  }
  mask[u] = m;
}

// Free-space seeding of the somata's DAFs (the free-space seed for every soma label at once:
// the same float32 expression on the same coordinate differences, so the same bits). root[L] < 0:
// not a soma. Voxels of L within radius[L] of root[L] get the closed-form free-space path length.
__global__ void freespace_seed_lab(
    float* dist, const LABEL_T* lab, const long long* root, const float* radius,
    long long sx, long long sy, long long sz, float wx, float wy, float wz)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= sx * sy * sz) return;
  LABEL_T L = lab[v];
  if (L == 0 || root[L] < 0) return;
  float safe_radius = radius[L];
  long long x, y, z; brook_unravel(v, sx, sy, x, y, z);
  long long sxx, syy, szz; brook_unravel(root[L], sx, sy, sxx, syy, szz);
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

// mark[v] = 1 where a soma label's DBF is its maximum (the candidates of the soma root)
__global__ void dbf_maxima_lab(
    const float* dbf, const LABEL_T* lab, const float* dbf_max, const unsigned char* soma,
    long long n, unsigned char* mark)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  LABEL_T L = lab[v];
  mark[v] = (L != 0 && soma[L] && dbf[v] == dbf_max[L]) ? 1 : 0;
}

// Volume-wide PDRF with per-label parameters (the same float32 operation order as the scalar
// formula; compiled with --fmad=false). Background -> +inf.
// A voxel of the label its DAF flood never reached (a voxel graph can cut it off, e.g. a filled
// soma void under a cc3d graph) enters with DAF 0, as the per-component normalization does.
// A foreground voxel at distance zero (background painted into a label by the avocado
// correction, whose field is not refreshed) is +inf to the formula, as the component
// preparation's normalization makes it.
__global__ void pdrf_lab(
    const float* dbf, const float* daf, const LABEL_T* lab, const float* M, const float* inv,
    const unsigned char* use, int nsq, float scale, long long n, float* out)
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  LABEL_T L = lab[i];
  if (L == 0 || use[L] == 0) { out[i] = BROOK_INF; return; }
  float d = dbf[i];
  if (d == 0.0f) d = BROOK_INF;
  float p = 1.0f - d * M[L];
  for (int k = 0; k < nsq; k++) p = p * p;
  p = p * scale;
  if (use[L] == 2) { float d = daf[i]; if (isinf(d)) d = 0.0f; p = p + d * inv[L]; }
  out[i] = p;
}

// Predecessor = argmin neighbour dist within the component, first k on ties (the rule of
// vw_assign_parents); -1 for the source / unreached voxels. Volume flat indices (int32).
// A voxel graph gates the pull as vw_assign_parents does: the edge u -> v must be permitted.
__global__ void parents_lab(
    const float* dist, const LABEL_T* lab, const unsigned char* use, const unsigned int* vg, int has_vg,
    long long sx, long long sy, long long sz, int* parent)
{
  long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long n = sx * sy * sz;
  if (v >= n) return;
  LABEL_T L = lab[v];
  if (L == 0 || use[L] == 0 || !(dist[v] < BROOK_INF) || dist[v] == 0.0f) { parent[v] = -1; return; }
  long long x, y, z; brook_unravel(v, sx, sy, x, y, z);
  float mindu = BROOK_INF; long long argu = -1;
  for (int k = 0; k < 26; k++) {
    long long nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    long long u = BROOK_IDX(nx, ny, nz, sx, sy);
    if (lab[u] != L) continue;
    if (!brook_vg_ok(vg, has_vg, u, k)) continue;
    float du = dist[u];
    if (du < mindu) { mindu = du; argu = u; }
  }
  parent[v] = (int)argu;
}

// Single-thread walk of the volume parent field (pointer chase: inherently serial).
__global__ void backtrace_vol(
    const int* parent, long long target, long long source, int* out, int* out_len, int maxlen)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long cur = target;
  int cnt = 0;
  while (cnt < maxlen) {
    out[cnt++] = (int)cur;
    if (cur == source) break;
    int p = parent[cur];
    if (p < 0) break;
    cur = p;
  }
  for (int i = 0; i < cnt / 2; i++) { int t = out[i]; out[i] = out[cnt - 1 - i]; out[cnt - 1 - i] = t; }
  *out_len = cnt;
}
