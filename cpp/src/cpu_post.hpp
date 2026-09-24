// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "cpu_transform.hpp"
#include "skeleton.hpp"
namespace brook {
// Precision flags preserve NumPy's per-array dtype and promotion rules. Float32
// values occupy exact double representations internally, but every float32
// arithmetic step explicitly rounds before the next operation.
struct CpuGeometry {
  int64_t label = 0;
  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<uint32_t, 2>> edges;
  std::vector<double> radii;
  std::array<float, 3> anisotropy = {1, 1, 1};
  bool vertex64 = false, radius64 = false;
  CpuGeometry() = default;
  CpuGeometry(Skeleton input)
      : label(input.label), edges(std::move(input.edges)), anisotropy(input.anisotropy) {
    for (auto v : input.vertices)
      vertices.push_back({v[0], v[1], v[2]});
    radii.assign(input.radii.begin(), input.radii.end());
  }
};
struct CpuQueryDtypeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// Source indices refer to the concatenation of input vertices, before any merging.
struct CpuMetadata {
  int64_t owner = -1;
  bool physical = false;
  CpuTransform transform = cpu_identity;
};
struct CpuSkeleton {
  CpuGeometry skeleton;
  std::vector<uint64_t> sources;
  CpuMetadata metadata;
};
CpuSkeleton cpu_postprocess(CpuGeometry input, double dust, double ticks,
                            const CpuMetadata &metadata = {}, bool tick_float32 = false);
// Where the pieces of a join came from, so a caller can reproduce NumPy's edge dtype.
struct CpuJoinTrace {
  std::vector<int64_t> pieces; // input index of each piece; -1 if the input had several components
  bool joined = false;         // at least one connecting edge was added
};
CpuSkeleton cpu_join_precise(std::vector<CpuGeometry> inputs, double radius,
                             bool restrict_by_radius,
                             const std::vector<CpuMetadata> &metadata = {},
                             bool radius_float32 = false, bool radius_double_compare = false,
                             CpuJoinTrace *trace = nullptr);
CpuSkeleton cpu_join(std::vector<Skeleton> inputs, double radius, bool restrict_by_radius);
struct CpuCrop { std::array<double,6> bounds{};bool apply=false; };
CpuSkeleton cpu_fuse(std::vector<CpuGeometry> inputs,
                     const std::vector<CpuMetadata> &metadata,
                     const std::vector<CpuCrop> &crop_boxes = {},
                     std::vector<size_t> *attribute_inputs = nullptr);
std::pair<CpuSkeleton,double> cpu_physical_length(CpuGeometry input,
                                                  const CpuMetadata &metadata);
} // namespace brook
