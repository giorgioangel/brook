// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include <optional>
#include <memory>

namespace brook {
struct PackedSkeletons;
using Point=std::array<int64_t,3>;
struct Skeleton {
  int64_t label=0;
  std::vector<std::array<float,3>> vertices;
  std::vector<std::array<uint32_t,2>> edges;
  std::vector<float> radii;
  std::array<float,3> anisotropy={1,1,1};
};
struct VoxelSkeleton {
  std::vector<std::array<uint32_t,3>> vertices;
  std::vector<std::array<uint32_t,2>> edges;
  std::vector<float> radii;
};
struct TraceParameters {
  double scale=10,constant=10,pdrf_scale=5000,pdrf_exponent=16;
  double soma_detection=1100,soma_acceptance=4000,soma_scale=0.5,soma_constant=0;
  std::optional<int64_t> max_paths;
};
struct SkeletonizeOptions {
  TraceParameters teasar;
  std::array<float,3> anisotropy={1,1,1};
  double dust=1000;
  bool fix_branching=true,fix_borders=true,fill_holes=false,fix_avocados=false;
  double avocado_detection=0;
  std::optional<std::vector<int64_t>> object_ids;
  std::vector<Point> targets_before,targets_after;
  Array voxel_graph;
  std::optional<bool> black_border_override;
  // Direct component APIs retain double precision when forming EDF step weights.
  std::optional<std::array<double,3>> trace_step_spacing;
  bool component_api=false;
};
struct Box {
  Point lo={INT32_MAX,INT32_MAX,INT32_MAX},hi={-1,-1,-1};
  int64_t count=0;
};
struct TraceZeroDivision : std::runtime_error {TraceZeroDivision():std::runtime_error("float division by zero") {}};
struct ComponentPreparation {
  Array mask,dbf,daf,penalty,parents;
  Point root{},target{};
  std::vector<Point> before,after;
  int64_t valid=0;
  int filled=0;
  bool soma=false;
  double soma_radius=0;
  float maximum=0;
};
struct ArenaLayout {
  std::vector<int64_t> offsets;
  std::vector<std::array<int64_t,3>> dimensions;
  std::vector<Point> origins;
  std::vector<int> label_arena;
  // The base arena each arena belongs to (its sample's whole volume; empty: arena 0 for all).
  // Private soma arenas are crops of their base; assembly keys vertices in the base's frame.
  std::vector<int> base;
  // Per label: the first z slice of its sample in a stack of samples (empty: 0 for all). Assembly
  // subtracts it before scaling, so vertices are the sample's own coordinates times the anisotropy.
  std::vector<int64_t> label_z;
};
ComponentPreparation prepare_component(Context &ctx,Array mask,Array dbf,const SkeletonizeOptions &options,
    std::vector<Point> before,std::vector<Point> after,std::optional<Point> root,const Array *voxel_graph=nullptr,
    const std::pair<Array,int> *void_check=nullptr);
std::pair<Array,int64_t> fill_all_holes(Context &ctx,Array labels,const std::vector<Box> &boxes);
std::pair<Array,int64_t> fill_all_holes(Context &ctx,Array labels);
// segment > 0: the z axis is a stack of independent samples of that depth, corrected one by one.
// A voxel graph (one sample) gates the refreshed distance field as it gates the initial one.
void fix_avocados(Context &ctx,Components &cc,Array &dbf,double threshold,std::array<float,3> spacing,bool black_border,int64_t segment=0,
    const Array *voxel_graph=nullptr);
std::vector<Box> analyze(Context &ctx,const Array &components,size_t count);
// After a fill painted voxels: roots[id-1] becomes the id's first voxel in memory order (Kimimaro's first_label on the crop;
// the lockstep's find_root source). An id the fill emptied keeps its root, which still lies in its own sample of a stack;
// nothing traces it.
void refresh_roots(Context &ctx,const Array &components,std::vector<int64_t> &roots);
std::pair<Array,Array> crop_component(Context &ctx,const Array &components,const Array &dbf,int64_t label,const Box &box,
    const Array *voxel_graph=nullptr,Array *graph_crop=nullptr);
std::vector<std::vector<Point>> border_targets(Context &ctx,const Array &components,size_t count,std::array<float,3> anisotropy);
std::vector<std::vector<Point>> border_targets_host(Context &ctx,const brook_volume &components,size_t count,std::array<float,3> anisotropy);
// `samples` volumes of shape (sx, sy, sz / samples) stacked along z: each sample's own faces, points in the stack's frame.
std::vector<std::vector<Point>> border_targets_stacked(Context &ctx,const Array &components,int samples,size_t count,std::array<float,3> anisotropy);
Array filter_labels(Context &ctx,const Array &labels,const std::vector<int64_t> &ids);
// The component a manual target point falls on; 0 is background, which no component traces. A
// negative coordinate counts from the far edge, and a point outside the volume is rejected. Only
// the lookup wraps: the trace flattens the point as the caller wrote it.
int64_t component_at(Context &ctx,const Array &components,Point point);
std::vector<Skeleton> skeletonize(Context &ctx,Array labels,const SkeletonizeOptions &options,PackedSkeletons *device_result=nullptr);
// skeletonize() in two halves, so a batch can trace many prepared samples in one lockstep call.
struct SomaPreparation;
struct LockstepResult;
struct PreparedSample {
  Components cc;Array dbf;std::vector<Box> boxes;size_t count=0;
  std::vector<std::vector<Point>> borders,extra_before,extra_after;
  Array trace_graph;                                  // graph mode: the tracing graph (else empty)
  std::shared_ptr<SomaPreparation> lockstep;          // the lockstep tracer's inputs, when it applies
  bool empty=true;
};
PreparedSample prepare_sample(Context &ctx,Array labels,const SkeletonizeOptions &options,bool lockstep_wanted=true);
std::vector<Skeleton> finish_sample(Context &ctx,PreparedSample &sample,LockstepResult &locked,const SkeletonizeOptions &options,
    PackedSkeletons *device_result);
Skeleton trace_component(Context &ctx,Array mask,Array dbf,const SkeletonizeOptions &options,
                         std::vector<Point> before,std::vector<Point> after,std::optional<Point> root,const Array *voxel_graph=nullptr,
                         const std::pair<Array,int> *void_check=nullptr,bool *assembled=nullptr,VoxelSkeleton *voxel_result=nullptr);
void consolidate(Skeleton &skeleton);
void consolidate(VoxelSkeleton &skeleton);
Point unravel(int64_t index,const std::array<int64_t,3> &shape);
int64_t flatten(Point point,const std::array<int64_t,3> &shape);
}
