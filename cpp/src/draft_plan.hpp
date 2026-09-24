// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"

namespace brook {
struct DraftEstimate { std::vector<int> labels;std::vector<double> remaining; };
struct DraftWindowPlan {
  std::array<int,3> log2;
  int drafting_labels=0,windows=0,voxels=0;
};
DraftEstimate estimate_draft_paths(Context &ctx,const int *active,int count,const long long *cursor,
    const long long *ends,const int *order,const uint8_t *alive,const int *paths,const long long *voxels);
std::array<int,3> plan_draft_window(Context &ctx,const int *starts,const int *lengths,const int *store,
    const float *dbf,int records,std::array<int64_t,3> original_shape,std::array<float,3> anisotropy,
    double scale,double constant,int forced_side=0);
std::optional<DraftWindowPlan> fit_draft_window(std::array<int,3> requested,int reserve,
    size_t label_table,size_t label_bytes,int labels,int drafts);
int draft_reserve_budget(size_t free_bytes,int dense,int foreground,size_t label_bytes,
    bool retained_inputs,int64_t wanted=INT64_MAX);
}
