// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "batched.hpp"
#include "assembly.hpp"

namespace brook {
struct LockstepResult {
  std::vector<std::optional<Skeleton>> components;
  std::vector<uint8_t> handled;
  std::optional<ComponentAssembly> device;
  int iterations=0,paths=0,ejected=0;
};
LockstepResult trace_lockstep(Context &ctx,Components cc,Array dbf,BatchedPreparation preparation,const ArenaLayout &layout,
    const std::vector<Box> &boxes,const std::vector<std::vector<Point>> &borders,
    const std::vector<std::vector<Point>> &before,const std::vector<std::vector<Point>> &after,
    const SkeletonizeOptions &options,bool keep_device=false);
}
