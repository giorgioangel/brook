// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"

namespace brook {
struct PackedSkeletons {
  std::vector<int64_t> labels,vertex_offsets{0},edge_offsets{0};
  Array vertices,edges,radii;
  std::array<float,3> anisotropy{1,1,1};
};
struct SkeletonPart {
  int64_t label;
  Array vertices,edges,radii;
  std::array<float,3> origin{0,0,0};
};
PackedSkeletons empty_packed(Context &ctx,std::array<float,3> anisotropy);
PackedSkeletons upload_packed(Context &ctx,std::vector<int64_t> labels,std::vector<int64_t> vertex_offsets,
    std::vector<int64_t> edge_offsets,const float *vertices,const uint32_t *edges,const float *radii,std::array<float,3> anisotropy);
PackedSkeletons skeletonize_packed(Context &ctx,Array labels,const SkeletonizeOptions &options);
PackedSkeletons pack_skeletons(Context &ctx,const std::vector<Skeleton> &skeletons,std::array<float,3> anisotropy);
std::vector<Skeleton> unpack_skeletons(const PackedSkeletons &packed);
std::vector<SkeletonPart> skeleton_parts(const PackedSkeletons &packed,std::array<float,3> origin={0,0,0});
PackedSkeletons merge_parts(Context &ctx,const std::vector<SkeletonPart> &parts,std::array<float,3> anisotropy,
                            bool sort_labels=true,bool drop_empty=false);
PackedSkeletons merge_fragments(Context &ctx,const std::vector<PackedSkeletons> &fragments,
                                const std::vector<std::array<float,3>> &origins);
Array row_view(const Array &array,int64_t first,int64_t count,int width);
}
