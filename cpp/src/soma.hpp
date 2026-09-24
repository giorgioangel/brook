// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "batched.hpp"

namespace brook {
struct SomaPreparation {
  Components components;
  Array dbf;
  std::vector<Box> boxes;
  BatchedPreparation fields;
  ArenaLayout layout;
};
// graph: the tracing voxel graph of the volume (uint32), when there is one. Private soma
// preparations run on its crop and private arenas carry that crop in fields.graph.
SomaPreparation prepare_somata(Context &ctx,const Components &cc,const Array &dbf,BatchedPreparation fields,
    const std::vector<Box> &boxes,const std::vector<std::vector<Point>> &borders,
    const std::vector<std::vector<Point>> &before,const std::vector<std::vector<Point>> &after,
    const SkeletonizeOptions &options,const Array *graph=nullptr);
}
