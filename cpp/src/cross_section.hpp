// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"

namespace brook {
struct CrossSectionOptions {
  std::array<float,3> anisotropy={1,1,1};
  std::array<double,3> position_spacing={1,1,1};
  bool float64_spacing=false;
  Point origin{}; // host volume's voxel origin in a larger image
  int smoothing_window=1,step=1;
  bool fill_holes=false,multipass=false,repair_contacts=false,visualize=false,in_place=false;
  bool shape_is_crop=false;
};
struct CrossSectionInput {
  Skeleton skeleton;
  std::optional<uint64_t> unsigned_label;
  // Python accepts float64 coordinates as well as native float32 skeletons.
  std::vector<std::array<double,3>> precise_vertices;
  bool float64_vertices=false;
  bool physical=true;
  std::vector<float> areas;
  std::vector<uint8_t> contacts;
};
struct CrossSectionResult {
  bool processed=false;
  std::vector<float> areas;
  std::vector<uint8_t> contacts;
  std::vector<uint32_t> sections;
  Point section_shape{},section_origin{};
};
std::vector<CrossSectionResult> cross_sectional_area_host(const brook_volume &labels,
    const std::vector<CrossSectionInput> &skeletons,const CrossSectionOptions &options);
}
