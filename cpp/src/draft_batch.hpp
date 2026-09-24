// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "draft_plan.hpp"
#include "kernel_module.hpp"
#include "skeleton.hpp"
#include "lockstep_order.hpp"

namespace brook {
struct DraftLane {
  Buffer<int> active,header,wcur_a,wcur_b,target_header,work_header,counts,ejected;
  Buffer<long long> cursor_scratch,sxyz;
  Buffer<unsigned long long> best,pba,pbb;
  Buffer<float> sdu,stb;
  Buffer<int> sar,a,b,far_a,far_b,touched,snap,snapm,store,rec_label,rec_length,rec_start;
  size_t work_capacity=0;
  DraftLane(Context &ctx,int cap,int lanes,size_t store_capacity,size_t record_capacity);
  void reserve_worklists(Context &ctx,size_t count);
};
struct DraftShared {
  void *labels;
  void *graph=nullptr;int has_graph=0;   // the voxel graph (uint32) of every arena, when there is one
  uint8_t *alive;
  float *dbf,*penalty,*distance;
  unsigned long long *key;
  int *flags,*path_count,*last_offset,*last_length,*manual_start,*manual_count,*manual,*max_paths,*wide_store,*lane_stop;
  long long *cursor,*order_end;
  int *order;
  int total,lanes,target_grid,draft_grid,railroad_grid,invalidation_grid;
  float wx,wy,wz,scale,constant;
};
struct DraftBatch {
  int drafts,cap,table,dense,reserve,cap_windows,windows,expanded_cap,groups,encoded_sides,arenas;
  DraftWindowPlan plan;
  Buffer<long long> arena_offsets,arena_dimensions,window_start,window_end,soma_root,draft_position,scan_offset;
  Buffer<double> soma_radius;
  Buffer<float> threshold,delta;
  Buffer<unsigned long long> rail_best,minfar,first_dead;
  Buffer<int> done,target,length,source_offset,output_offset,accepted,accepted_count,header,stats,expanded_header,invalidation_header;
  Buffer<int> expanded_active,expanded_target,expanded_length,expanded_offset,zero_fill,position_of,window_origin,window_home,window_label,window_active;
  Buffer<int> route_bad,window_bad,contact_mask,info,number_drafted,bits,depth,walk,sources;
  Buffer<int> join_u,join_rank,route_rank,repick;   // the join rule (lockstep_regular.inc, JOIN)
  Buffer<long long> join_home;
  DraftBatch(Context &ctx,int drafts,int cap,int table,int dense,int foreground,int reserve,const DraftWindowPlan &plan,
      const ArenaLayout &layout,const TargetOrder &order,const std::vector<double> &soma_radius,const std::vector<long long> &soma_root);
  void body(Context &ctx,const KernelModule &module,const DraftShared &shared,DraftLane &lane,
      int rounds,int pool,int adaptive,int share,unsigned long long condition=0);
  void tail(Context &ctx,const KernelModule &module,const DraftShared &shared,DraftLane &lane,
      int adaptive,unsigned long long condition=0);
};
}
