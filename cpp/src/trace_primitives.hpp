// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"

namespace brook {
struct ComponentSummary { float maximum;int valid,first; };
ComponentSummary summarize_component(Context &ctx,const Array &mask,const Array &dbf);
std::vector<int> maximum_positions(Context &ctx,const Array &dbf,float maximum);
struct DistanceField { Array distance; int64_t maximum; };
DistanceField euclidean_distance_field(Context &ctx,const Array &mask,int64_t source,
                                      std::array<float,3> anisotropy,float free_radius=0,
                                      const Array *voxel_graph=nullptr,bool reference_weights=false,
                                      size_t foreground_bound=0,const std::array<double,3> *step_spacing=nullptr);
std::pair<Array,Array> parental_field(Context &ctx,const Array &field,int64_t source,
                                    bool rail_terminal=false,const Array *voxel_graph=nullptr);
Array assign_parents(Context &ctx,const Array &field,const Array &distance,int64_t source,
                     bool rail_terminal=false,const Array *voxel_graph=nullptr);
std::pair<Array,Array> parental_field_banded(Context &ctx,const Array &mask,const Array &field,int64_t source);
Array pdrf(Context &ctx,const Array &dbf,const Array &daf,double dbf_max,double max_daf,
           double scale,double exponent);
// Component-owned, nonoverlapping DBF/DAF are normalized in place before PDRF.
Array normalize_pdrf(Context &ctx,Array &dbf,Array &daf,double dbf_max,double max_daf,
                     double scale,double exponent);
std::pair<Array,int> fill_voids(Context &ctx,const Array &mask);

struct TraceWorkspace {
  int n, lane_capacity;
  bool reuse_path;
  std::shared_ptr<Allocation> arena;
  size_t arena_offset=0;
  Buffer<unsigned long long> key,minfar,rail_best,stats,pba,pbb;
  Buffer<int> snap,snapm,hdr,wa,wb,fa,fb,inq,touched,infar,counts,out,out_len;
  Buffer<float> dist,threshold,sdu;
  Buffer<long long> sxyz;
  // Declared before device storage: device release completes queued copies
  // before the host staging allocation is destroyed.
  std::vector<long long> staged_path;
  Buffer<long long> path_seeds;
  TraceWorkspace(Context &ctx,int size,bool need_railroad=true);
  void stage_path(Context &ctx,const std::vector<int64_t> &path);
};
std::vector<int64_t> railroad(Context &ctx,const Array &field,int64_t target,
                              TraceWorkspace &ws,const Array *voxel_graph=nullptr);
int invalidate(Context &ctx,Array &mask,const Array &dbf,float scale,float constant,
               std::array<float,3> anisotropy,const std::vector<int64_t> &path,
               TraceWorkspace &ws,const Array *voxel_graph=nullptr);
int invalidate_device_path(Context &ctx,Array &mask,const Array &dbf,float scale,float constant,
               std::array<float,3> anisotropy,const std::vector<int64_t> &path,
               TraceWorkspace &ws,const Array *voxel_graph,const long long *device_path);
std::vector<int64_t> backtrace(Context &ctx,const Array &parents,int64_t target,int64_t source);
std::vector<int64_t> backtrace_workspace(Context &ctx,const Array &parents,int64_t target,int64_t source,TraceWorkspace &ws);
}
