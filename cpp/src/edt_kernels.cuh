// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "common.cuh"
namespace brook::cuda {
template<class Index> struct EdtIndexType { using type = Index; };

#define BROOK_INF __int_as_float(0x7f800000)

// Pass 1: along x (stride 1). One thread per (y,z) line.
// acc[k] = the spacing added k times IN FLOAT32, one rounding per step: the edt package (edt.hpp,
// squared_edt_1d_multi_seg) walks the line with d[i] = d[i-1] + w and squares at the end, so for
// a spacing that is not exactly representable (e.g. 7.91 um CT) k * w is NOT what it computes.
// Integer spacings give the same result either way; fractional ones do not.
template<int Sampling=1,class Index=long long,class Labels,class Distance>
__global__ void edt_pass1_x(
    Distance D, Labels labels,
    typename EdtIndexType<Index>::type sx, typename EdtIndexType<Index>::type sy, typename EdtIndexType<Index>::type sz, const float* acc_all, typename EdtIndexType<Index>::type acc_pitch, int black_border)
{
  Index line = (Index)blockIdx.x * blockDim.x + threadIdx.x;
  Index nlines = sy * sz;
  if (line >= nlines) return;
  Index y = line % sy, z = line / sy;
  Index base = sx * (y + sy * z);
  const float* acc = acc_all + z * acc_pitch;      // acc_pitch 0: one spacing; else one table per z slab

  Index i = 0;
  while (i < sx) {
    auto L = labels[base + i];
    Index c = i, d = i;
    while (d + 1 < sx && labels[base + d + 1] == L) d++;
    bool leftw  = (c > 0) || black_border;
    bool rightw = (d < sx - 1) || black_border;
    Index lpos = c - 1;   // -1 if c==0
    Index rpos = d + 1;   // sx if d==sx-1
    for (Index p = c + (Sampling - c % Sampling) % Sampling; p <= d; p += Sampling) {
      float best = BROOK_INF;
      if (leftw)  { float dd = acc[p - lpos]; float v = dd * dd; if (v < best) best = v; }
      if (rightw) { float dd = acc[rpos - p]; float v = dd * dd; if (v < best) best = v; }
      D[base + p] = best;
    }
    i = d + 1;
  }
}

// Passes 2/3: FH lower envelope along axis (1=y, 2=z). One thread per line.
// Scratch V/H/Z are per-line (pitch elements each).
template<int Sampling=1,class Index=long long,class Labels,class Distance>
__global__ void edt_pass_fh(
    Distance D, Labels labels,
    int* Vg, float* Hg, double* Zg,
    typename EdtIndexType<Index>::type sx, typename EdtIndexType<Index>::type sy, typename EdtIndexType<Index>::type sz, int axis,
    float w, const float* wslab, int black_border, typename EdtIndexType<Index>::type pitch, typename EdtIndexType<Index>::type line0, typename EdtIndexType<Index>::type nbatch,
    typename EdtIndexType<Index>::type segment = 0)
{
  // lines are processed in batches so the scratch is O(batch), not O(volume)
  Index tid = (Index)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nbatch) return;
  Index line = line0 + tid;
  Index n, stride, base, nlines;
  if (axis == 1) {            // along y
    nlines = (sx / Sampling) * sz; if (line >= nlines) return;
    Index x = (line % (sx / Sampling)) * Sampling, z = line / (sx / Sampling);
    n = sy; stride = sx; base = x + sx * sy * z;
    if (wslab) w = wslab[z];                       // a stack of independent planes (border targets)
  } else if (segment > 0) {   // along z, in independent segments: a stack of samples, each an open volume
    Index per = (sx / Sampling) * (sy / Sampling), k = sz / segment;
    nlines = per * k; if (line >= nlines) return;
    Index l = line % per, seg = line / per;
    Index x = (l % (sx / Sampling)) * Sampling, yy = (l / (sx / Sampling)) * Sampling;
    n = segment; stride = sx * sy; base = x + sx * yy + sx * sy * segment * seg;
  } else {                    // along z
    nlines = (sx / Sampling) * (sy / Sampling); if (line >= nlines) return;
    Index x = (line % (sx / Sampling)) * Sampling, yy = (line / (sx / Sampling)) * Sampling;
    n = sz; stride = sx * sy; base = x + sx * yy;
  }

  double w2 = (double)(w * w);          // edt.hpp squares the spacing in float32, then widens
  int*    V = Vg + tid * pitch;
  float*  H = Hg + tid * pitch;
  double* Z = Zg + tid * pitch;

  Index i = 0;
  while (i < n) {
    auto L = labels[base + i * stride];
    Index c = i, d = i;
    while (d + 1 < n && labels[base + (d + 1) * stride] == L) d++;
    bool leftw  = (c > 0) || black_border;
    bool rightw = (d < n - 1) || black_border;
    Index lpos = c - 1, rpos = d + 1;

    int k = -1;   // envelope top index

    // Add parabolas in increasing position. Skip +inf heights (never minimal).
    // left wall (height 0)
    for (int pass = 0; pass < 1; pass++) {
      if (!leftw) break;
      Index pp = lpos; double ph = 0.0;
      // k == -1 here, so just place
      k = 0; V[0] = (int)pp; H[0] = 0.0f; Z[0] = -1e300;
    }
    // segment voxels (finite heights only)
    for (Index p = c; p <= d; p++) {
      float hf = D[base + p * stride];
      if (!(hf < BROOK_INF)) continue;     // skip +inf (and NaN)
      double ph = (double)hf; Index pp = p;
      bool placed = false;
      while (k >= 0 && !placed) {
        double s = ((ph + w2 * (double)pp * (double)pp)
                  - ((double)H[k] + w2 * (double)V[k] * (double)V[k]))
                 / (2.0 * w2 * ((double)pp - (double)V[k]));
        if (s <= Z[k]) { k--; }
        else { k++; V[k] = (int)pp; H[k] = (float)ph; Z[k] = s; placed = true; }
      }
      if (!placed) { k = 0; V[0] = (int)pp; H[0] = (float)ph; Z[0] = -1e300; }
    }
    // right wall (height 0)
    if (rightw) {
      Index pp = rpos; double ph = 0.0;
      bool placed = false;
      while (k >= 0 && !placed) {
        double s = ((ph + w2 * (double)pp * (double)pp)
                  - ((double)H[k] + w2 * (double)V[k] * (double)V[k]))
                 / (2.0 * w2 * ((double)pp - (double)V[k]));
        if (s <= Z[k]) { k--; }
        else { k++; V[k] = (int)pp; H[k] = (float)ph; Z[k] = s; placed = true; }
      }
      if (!placed) { k = 0; V[0] = (int)pp; H[0] = 0.0f; Z[0] = -1e300; }
    }

    // Evaluate the envelope at q in [c, d].
    if (k < 0) {
      for (Index q = c + (Sampling - c % Sampling) % Sampling; q <= d; q += Sampling) D[base + q * stride] = BROOK_INF;
    } else {
      int kk = 0;
      for (Index q = c + (Sampling - c % Sampling) % Sampling; q <= d; q += Sampling) {
        while (kk < k && Z[kk + 1] < (double)q) kk++;
        double dq = (double)q - (double)V[kk];
        D[base + q * stride] = (float)(w2 * dq * dq + (double)H[kk]);
      }
    }
    i = d + 1;
  }
}

}
