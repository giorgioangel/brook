// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include "skeleton.hpp"
#include <cstdlib>

namespace brook {
Array upload_stream_region(Context &ctx,const brook_volume &region);
size_t default_stream_budget(size_t requested);
std::vector<Box> analyze_streamed(Context &ctx,const brook_volume &input,size_t count,size_t budget);
struct HostComponents {
  std::array<int64_t,3> shape{};
  brook_dtype dtype=BROOK_U8;
  struct FreeHost {void operator()(uint8_t *p) const noexcept {std::free(p);}};
  std::unique_ptr<uint8_t,FreeHost> labels;
  std::vector<int64_t> mapping{0},roots;
  brook_volume view() const;
};
HostComponents connected_components_streamed(Context &ctx,const brook_volume &labels,size_t budget=0,
                                             const std::vector<int64_t> *object_ids=nullptr);
std::vector<Skeleton> skeletonize_streamed(Context &ctx,const brook_volume &labels,
                                         const SkeletonizeOptions &options,size_t budget=0);
// Host input, host F-contiguous float output. budget==0 selects the environment/
// free-memory default. At least one complete XY plane and one XZ tile must fit;
// the budget is a scheduling target, with that irreducible minimum.
void edt_streamed(Context &ctx,const brook_volume &labels,std::array<float,3> spacing,
                  bool black_border,size_t budget,float *output,size_t capacity);
}
