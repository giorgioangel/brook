// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "packed.hpp"

namespace brook {
Array skeleton_components(Context &ctx,const PackedSkeletons &packed);
PackedSkeletons remove_dust(Context &ctx,const PackedSkeletons &packed,double threshold);
PackedSkeletons select_vertices(Context &ctx,const PackedSkeletons &packed,const Array &keep);
PackedSkeletons prune_unused(Context &ctx,const PackedSkeletons &packed);
PackedSkeletons remove_loops(Context &ctx,const PackedSkeletons &packed);
PackedSkeletons remove_ticks(Context &ctx,const PackedSkeletons &packed,double threshold);
PackedSkeletons join_components(Context &ctx,const PackedSkeletons &packed);
PackedSkeletons postprocess(Context &ctx,const PackedSkeletons &packed,double dust_threshold,double tick_threshold);
}
