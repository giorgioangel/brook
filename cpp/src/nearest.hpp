// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"

namespace brook {
std::vector<std::array<int64_t,3>> nearest_label_voxels_host(const brook_volume &labels,
    const std::vector<brook_label_query> &queries,bool bounded=true,Context *fallback=nullptr);
std::vector<std::array<int64_t,3>> nearest_label_voxels(Context &ctx,const Array &labels,
    const std::vector<brook_label_query> &queries);
std::vector<std::array<int64_t,3>> nearest_label_voxels(Context &ctx,const brook_volume &labels,
    const std::vector<brook_label_query> &queries);
}
