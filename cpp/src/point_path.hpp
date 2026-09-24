// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"

namespace brook {
std::vector<int64_t> point_path_heap(const std::vector<float> &field,std::array<int64_t,3> shape,
                                    int64_t source,int64_t target);
Skeleton connect_points(Context &ctx,Array labels,Point start,Point end,std::array<float,3> spacing,
                        double scale=100000,double exponent=4);
}
