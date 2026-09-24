// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"
#include "kernel_module.hpp"
#include <unordered_map>

namespace brook {
struct SegmentedMaximum { std::vector<float> value;std::vector<int64_t> index; };
class BatchedFlooder {
  Context &ctx_;
  Array labels_;
  KernelModule module_;
  Buffer<int> a_,b_,inq_;
  Buffer<float> snapshot_;
  Buffer<long long> coordinates_;
  Buffer<unsigned long long> stats_;
  Buffer<unsigned> neighbors_;
 public:
  // graph: an optional voxel connectivity graph (uint32, the labels' shape); its push bits are folded
  // into the neighbour token, so every flood steps exactly the edges the single-component floods do.
  BatchedFlooder(Context &ctx,const Array &labels,const Array *graph=nullptr);
  void flood(Array &distance,const std::vector<int64_t> &sources,int mode,const Buffer<float> &weights,
             const Array *field=nullptr,bool seeded=false,bool banded=false);
  SegmentedMaximum argmax(const Array &values,size_t count);
};
struct BatchedPreparation {
  std::vector<uint8_t> active,soma;
  std::vector<float> dbf_max,max_daf;
  std::vector<int64_t> root,target;
  std::vector<int> filled;
  // Per-call precheck outputs only; sparse in component count, consumed before tracing.
  std::unordered_map<size_t,Array> filled_masks;
  Array daf,pdrf,parents;
  // The tracing graph (uint32) laid out like the fields; empty without a voxel graph.
  Array graph;
};
BatchedPreparation prepare_batched(Context &ctx,const Components &cc,const Array &dbf,
    const std::vector<Box> &boxes,const std::vector<std::vector<Point>> &borders,const SkeletonizeOptions &options,
    const Array *graph=nullptr);
}
