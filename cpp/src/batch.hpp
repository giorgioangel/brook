// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "packed.hpp"

namespace brook {
// A batch of independent samples skeletonized together. The packed arrays hold every skeleton of
// every sample, in sample order; `labels` is each skeleton's label in its own sample, `sample` its
// sample index and `sample_offsets` (size B+1) the skeleton range of each sample. Vertices are in
// each sample's own physical frame.
struct BatchedSkeletons {
  PackedSkeletons packed;
  std::vector<int64_t> labels,sample,sample_offsets{0};
};
// Per-sample results are identical to skeletonize(sample, options) for every sample. Manual target
// points are per sample: `targets_before`/`targets_after` hold one list of points per sample, in
// that sample's own coordinates, or are empty when no sample has any (`options.targets_*` must be
// empty). `graphs`: empty, or one voxel connectivity graph per sample (an empty Array for a sample
// without one): sample i is then skeletonized with voxel_graph = graphs[i].
BatchedSkeletons skeletonize_batch(Context &ctx,std::vector<Array> samples,const SkeletonizeOptions &options,
    const std::vector<std::vector<Point>> &targets_before={},const std::vector<std::vector<Point>> &targets_after={},
    std::vector<Array> graphs={});
// Skeletons [first, last) of a batch as a packed view sharing the batch's device storage.
PackedSkeletons packed_slice(const BatchedSkeletons &batch,int64_t first,int64_t last);
}
