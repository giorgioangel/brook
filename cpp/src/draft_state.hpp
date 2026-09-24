// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"

namespace brook {
// All captured loops must be closed before growing these buffers. Worklists,
// keys, flags and distances are clean at an iteration boundary; only the four
// persistent voxel fields are copied. Managed worklists must not be prefilled.
struct DraftVoxelBuffers {
  Array &labels,&dbf,&penalty;
  Buffer<uint8_t> &alive;
  Buffer<unsigned long long> &key;
  Buffer<int> &flags;
  Buffer<float> &distance;
  std::array<Buffer<int>*,7> worklists;
  Array *graph=nullptr;   // the voxel graph (uint32), when the trace has one: grown like the fields
};
void append_draft_reserve(Context &ctx,DraftVoxelBuffers state,int dense,int foreground,int reserve);
}
