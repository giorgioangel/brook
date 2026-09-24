// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"

namespace brook {
// Border-target candidates of one volume, or of `samples` volumes of shape (sx, sy, sz / samples)
// stacked along z with distinct label ids per sample. Faces are processed per axis pair in a stack
// of [face, separator, face] slabs per sample sized to that pair's face, so thin volumes do not pay
// for square slabs. Distinct ids keep consecutive samples' faces from merging.
constexpr int BORDER_PAIR_SLABS=3;
struct BorderGroup {
  int pair,sample,face;                 // face 0..5 of the sample: axis pair face/2, side face%2
  uint64_t owner,order;                 // component label; the first live pixel in face-major order
  std::vector<uint32_t> scans;          // candidate pixels (y * B + x) at the face's maximal distance
  float sumx=0,sumy=0;uint32_t size=0;  // tied groups: the face component's pixel sums (x outer, y inner, float32)
};
struct BorderCandidates {
  std::array<int64_t,3> shape{};int samples=1;
  std::array<int64_t,3> A{},B{};        // per pair: face extents along its second and first axis
  std::vector<BorderGroup> groups;
};
BorderCandidates batched_border_candidates(Context &ctx,const Array &input,std::array<float,3> aniso,int samples=1);
}
