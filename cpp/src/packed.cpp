// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "packed.hpp"
#include <algorithm>

namespace brook {
Array row_view(const Array &array,int64_t first,int64_t count,int width) {
  if(first<0 || count<0 || width<1 || uint64_t(first)>array.size()/width || uint64_t(count)>array.size()/width-uint64_t(first))
    throw std::invalid_argument("packed row view is out of bounds");
  Array result=array;result.byte_offset+=size_t(first)*width*dtype_size(array.dtype);
  result.shape={width,count,1};
  int64_t item=int64_t(dtype_size(array.dtype));
  if(width==1) return tensor_view(std::move(result),{count},{item});
  return tensor_view(std::move(result),{count,width},{width*item,item});
}
PackedSkeletons empty_packed(Context &ctx,std::array<float,3> aniso) {
  PackedSkeletons out;out.anisotropy=aniso;
  out.vertices=row_view(allocate(ctx,{3,0,1},BROOK_F32),0,0,3);
  out.edges=row_view(allocate(ctx,{2,0,1},BROOK_U32),0,0,2);
  out.radii=row_view(allocate(ctx,{0,1,1},BROOK_F32),0,0,1);return out;
}
PackedSkeletons upload_packed(Context &ctx,std::vector<int64_t> labels,std::vector<int64_t> voff,std::vector<int64_t> eoff,
    const float *vertices,const uint32_t *edges,const float *radii,std::array<float,3> aniso) {
  ctx.activate();
  if(voff.size()!=labels.size()+1 || eoff.size()!=labels.size()+1 || voff.front()!=0 || eoff.front()!=0 ||
      !std::is_sorted(voff.begin(),voff.end()) || !std::is_sorted(eoff.begin(),eoff.end())) throw std::invalid_argument("invalid packed offsets");
  int64_t nv=voff.back(),ne=eoff.back();
  if((nv && (!vertices || !radii)) || (ne && !edges)) throw std::invalid_argument("null packed data");
  auto v=allocate(ctx,{3,nv,1},BROOK_F32),e=allocate(ctx,{2,ne,1},BROOK_U32),r=allocate(ctx,{nv,1,1},BROOK_F32);
  if(v.bytes()) BROOK_CUDA(cudaMemcpyAsync(v.data(),vertices,v.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  if(e.bytes()) BROOK_CUDA(cudaMemcpyAsync(e.data(),edges,e.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  if(r.bytes()) BROOK_CUDA(cudaMemcpyAsync(r.data(),radii,r.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  ctx.synchronize();return {std::move(labels),std::move(voff),std::move(eoff),row_view(v,0,nv,3),row_view(e,0,ne,2),row_view(r,0,nv,1),aniso};
}
PackedSkeletons pack_skeletons(Context &ctx,const std::vector<Skeleton> &skeletons,std::array<float,3> aniso) {
  ctx.activate();auto out=empty_packed(ctx,aniso);
  std::vector<std::array<float,3>> vertices;std::vector<std::array<uint32_t,2>> edges;std::vector<float> radii;
  for(const auto &s:skeletons) {
    if(s.vertices.size()!=s.radii.size()) throw std::invalid_argument("skeleton radius count mismatch");
    for(auto edge:s.edges) if(edge[0]>=s.vertices.size() || edge[1]>=s.vertices.size()) throw std::invalid_argument("skeleton edge is out of bounds");
    out.labels.push_back(s.label);vertices.insert(vertices.end(),s.vertices.begin(),s.vertices.end());
    edges.insert(edges.end(),s.edges.begin(),s.edges.end());radii.insert(radii.end(),s.radii.begin(),s.radii.end());
    out.vertex_offsets.push_back(vertices.size());out.edge_offsets.push_back(edges.size());
  }
  auto v=allocate(ctx,{3,int64_t(vertices.size()),1},BROOK_F32),e=allocate(ctx,{2,int64_t(edges.size()),1},BROOK_U32);
  auto r=allocate(ctx,{int64_t(radii.size()),1,1},BROOK_F32);
  if(v.bytes()) BROOK_CUDA(cudaMemcpyAsync(v.data(),vertices.data(),v.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  if(e.bytes()) BROOK_CUDA(cudaMemcpyAsync(e.data(),edges.data(),e.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  if(r.bytes()) BROOK_CUDA(cudaMemcpyAsync(r.data(),radii.data(),r.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  ctx.synchronize();
  out.vertices=row_view(v,0,vertices.size(),3);out.edges=row_view(e,0,edges.size(),2);out.radii=row_view(r,0,radii.size(),1);return out;
}
std::vector<Skeleton> unpack_skeletons(const PackedSkeletons &p) {
  std::vector<std::array<float,3>> vertices(p.vertex_offsets.back());
  std::vector<std::array<uint32_t,2>> edges(p.edge_offsets.back());std::vector<float> radii(vertices.size());
  p.vertices.copy_to_host(vertices.data(),vertices.size()*sizeof(vertices[0]));
  p.edges.copy_to_host(edges.data(),edges.size()*sizeof(edges[0]));p.radii.copy_to_host(radii.data(),radii.size()*sizeof(float));
  std::vector<Skeleton> out;out.reserve(p.labels.size());
  for(size_t i=0;i<p.labels.size();++i) {
    Skeleton s;s.label=p.labels[i];s.anisotropy=p.anisotropy;
    s.vertices.assign(vertices.begin()+p.vertex_offsets[i],vertices.begin()+p.vertex_offsets[i+1]);
    s.radii.assign(radii.begin()+p.vertex_offsets[i],radii.begin()+p.vertex_offsets[i+1]);
    s.edges.assign(edges.begin()+p.edge_offsets[i],edges.begin()+p.edge_offsets[i+1]);out.push_back(std::move(s));
  }
  return out;
}
std::vector<SkeletonPart> skeleton_parts(const PackedSkeletons &p,std::array<float,3> origin) {
  if(p.vertex_offsets.size()!=p.labels.size()+1 || p.edge_offsets.size()!=p.labels.size()+1 ||
      p.vertex_offsets.front()!=0 || p.edge_offsets.front()!=0) throw std::invalid_argument("invalid packed offsets");
  std::vector<SkeletonPart> out;out.reserve(p.labels.size());
  for(size_t i=0;i<p.labels.size();++i) {
    auto nv=p.vertex_offsets[i+1]-p.vertex_offsets[i],ne=p.edge_offsets[i+1]-p.edge_offsets[i];
    out.push_back({p.labels[i],row_view(p.vertices,p.vertex_offsets[i],nv,3),row_view(p.edges,p.edge_offsets[i],ne,2),
      row_view(p.radii,p.vertex_offsets[i],nv,1),origin});
  }
  return out;
}
PackedSkeletons merge_fragments(Context &ctx,const std::vector<PackedSkeletons> &fragments,
    const std::vector<std::array<float,3>> &origins) {
  if(fragments.size()!=origins.size()) throw std::invalid_argument("fragment/origin count mismatch");
  std::vector<SkeletonPart> parts;
  for(size_t i=0;i<fragments.size();++i) {
    auto next=skeleton_parts(fragments[i],origins[i]);parts.insert(parts.end(),std::make_move_iterator(next.begin()),std::make_move_iterator(next.end()));
  }
  return merge_parts(ctx,parts,fragments.empty()?std::array<float,3>{1,1,1}:fragments.front().anisotropy);
}
}
