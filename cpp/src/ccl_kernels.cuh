// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "common.cuh"
#include <type_traits>
namespace brook::cuda {
template<class Index> using UnsignedIndex = std::conditional_t<sizeof(Index)==4, unsigned int, unsigned long long>;

// Union-find index type: int when the volume has < 2^31 voxels (native 32-bit atomicCAS at
// full rate and half the memory traffic of the 64-bit form), long long otherwise.


typedef long long i64;

template<class Index>
__device__ __forceinline__ Index brook_find(Index* parent, Index i) {
  Index curr = parent[i];
  if (curr != i) {
    Index next, prev = i;
    while (curr > (next = parent[curr])) {
      parent[prev] = next;   // racy path compression (ECL-CC)
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

template<class Index>
__device__ __forceinline__ void brook_union(Index* parent, Index a, Index b) {
  Index va = brook_find(parent, a);
  Index vb = brook_find(parent, b);
  bool repeat;
  do {
    repeat = false;
    if (va != vb) {
      Index ret;
      if (va < vb) {
        ret = (Index) atomicCAS((UnsignedIndex<Index>*)&parent[vb], (UnsignedIndex<Index>)vb, (UnsignedIndex<Index>)va);
        if (ret != vb) { vb = ret; repeat = true; }
      } else {
        ret = (Index) atomicCAS((UnsignedIndex<Index>*)&parent[va], (UnsignedIndex<Index>)va, (UnsignedIndex<Index>)vb);
        if (ret != va) { va = ret; repeat = true; }
      }
    }
  } while (repeat);
}

template<class Label, class Index, class Output>
__global__ void ccl_init(Index* parent, i64 n) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) parent[i] = (Index)i;
}

template<class Label, class Index, class Output>
__global__ void ccl_union(
    const Label* labels, Index* parent, i64 sx, i64 sy, i64 sz) {
  i64 v = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  i64 n = sx * sy * sz;
  if (v >= n) return;
  Label lv = labels[v];
  if (lv == 0) return;
  i64 x, y, z;
  brook_unravel(v, sx, sy, x, y, z);
  #pragma unroll
  for (int k = 0; k < 26; k++) {
    i64 nx = x + BROOK_NX[k], ny = y + BROOK_NY[k], nz = z + BROOK_NZ[k];
    if (nx < 0 || ny < 0 || nz < 0 || nx >= sx || ny >= sy || nz >= sz) continue;
    i64 u = BROOK_IDX(nx, ny, nz, sx, sy);
    if (u <= v) continue;                 // each undirected edge once
    if (labels[u] == lv) brook_union(parent, (Index)v, (Index)u);
  }
}

// Final flatten MUST use a read-only walk: the compressing brook_find writes to
// intermediate nodes (parent[prev]=next), which races with other threads that
// have already flattened those nodes to their root and would *regress* them to
// a non-root, mislabelling foreground voxels as background. Read-only walk +
// single self-write is race-safe and deterministic (pointers only decrease and
// roots are stable once unions are done).
template<class Label, class Index, class Output>
__global__ void ccl_flatten(Index* parent, i64 n) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Index curr = parent[i];
  Index next;
  while (curr != (next = parent[curr])) curr = next;   // walk to root, no writes
  parent[i] = curr;
}

// ---- relabel 1..N without volume-sized temporaries --------------------------------
// The relabel pass uses no volume-sized temporary arrays beyond the labels and output.
template<class Label, class Index, class Output>
__global__ void ccl_count_roots(
    const Label* labels, const Index* parent, i64 n, unsigned long long* count) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && labels[i] != 0 && parent[i] == (Index)i) atomicAdd(count, 1ULL);
}

template<class Label, class Index, class Output>
__global__ void ccl_collect_roots(
    const Label* labels, const Index* parent, i64 n, Index* roots, unsigned long long* cursor) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && labels[i] != 0 && parent[i] == (Index)i)
    roots[atomicAdd(cursor, 1ULL)] = (Index)i;      // any order; sorted on the host side (N entries)
}

// roots (sorted ascending) get -(rank) in place: parent[root] = -(id), id = 1..N
template<class Label, class Index, class Output>
__global__ void ccl_mark_ranks(Index* parent, const Index* roots, i64 nroots) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nroots) parent[roots[i]] = (Index)(-(i + 1));
}

// same, with explicit ids (streamed CCL: the global id of each slab-local root)
template<class Label, class Index, class Output>
__global__ void ccl_mark_ids(
    Index* parent, const Index* roots, const Index* ids, i64 nroots) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nroots) parent[roots[i]] = (Index)(-ids[i]);
}

// cc id written straight in the output dtype: a root holds -(id), everyone else its root
template<class Label, class Index, class Output>
__global__ void ccl_relabel(
    const Label* labels, const Index* parent, Output* cc, i64 n) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (labels[i] == 0) { cc[i] = 0; return; }
  Index p = parent[i];
  if (p >= 0) p = parent[p];
  cc[i] = (Output)(-(i64)p);
}

}
