// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"

namespace brook {
struct TargetOrder {
  Buffer<int> order;
  Buffer<long long> start,end;
  std::vector<long long> host_start,host_end;
};
TargetOrder build_target_order(Context &ctx,const Array &labels,const Array &daf,const std::vector<uint8_t> &eligible,
                              const std::vector<int64_t> &counts,bool fix_branching);
Buffer<int> build_component_target_order(Context &ctx,const Array &mask,const Array &daf,size_t foreground);
void initialize_alive(Context &ctx,const Array &labels,const std::vector<uint8_t> &eligible,Buffer<uint8_t> &alive);
std::vector<int> removed_per_label(Context &ctx,const Array &labels,const Buffer<int> &touched,int removed,size_t count);
}
