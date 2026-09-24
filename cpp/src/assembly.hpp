// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"

namespace brook {
// Rows are contiguous: vertices (N,3), edges (M,2), radii (N).
// Array's internal dimensions describe the same storage in Fortran order.
// Offsets are indexed by connected-component ID, including the unused ID zero.
struct ComponentAssembly {
  Array vertices,edges,radii;
  std::vector<int64_t> vertex_offsets,edge_offsets;
};
ComponentAssembly assemble_paths(Context &ctx,const Components &cc,const Array &dbf,
    const ArenaLayout &layout,const std::vector<uint8_t> &handled,
    const int *store,int positions,const int *record_start,const int *record_length,int records,
    std::array<float,3> anisotropy);
void download_components(const ComponentAssembly &packed,const Components &cc,
    const std::vector<uint8_t> &handled,std::array<float,3> anisotropy,
    std::vector<std::optional<Skeleton>> &out);
}
