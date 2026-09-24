// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "skeleton.hpp"
#include "preamble_kernels.cuh"
#include <algorithm>
#include <climits>

namespace brook {
namespace {
template<class Label> std::vector<Box> run_analyze(Context &ctx,const Array &input,size_t count) {
  size_t k=count+1;
  Buffer<int> lower(ctx,k*3),upper(ctx,k*3);
  Buffer<unsigned long long> counts(ctx,k);
  std::vector<int> low(k*3,INT32_MAX),high(k*3,-1);
  lower.set(ctx,low.data(),low.size());upper.set(ctx,high.data(),high.size());counts.clear(ctx);
  auto s=input.shape;
  if(input.size()) cuda::analyze_rows<<<static_cast<unsigned>((s[1]*s[2]+255)/256),256,0,ctx.stream>>>(
    static_cast<const Label*>(input.data()),s[0],s[1],s[2],lower.data(),upper.data(),counts.data());
  BROOK_CUDA(cudaGetLastError());
  low=lower.get(ctx,k*3);high=upper.get(ctx,k*3);auto sizes=counts.get(ctx,k);
  std::vector<Box> result(k);
  int64_t foreground=0;
  for(size_t i=1;i<k;++i) {
    for(int axis=0;axis<3;++axis) { result[i].lo[axis]=low[3*i+axis];result[i].hi[axis]=high[3*i+axis]; }
    result[i].count=sizes[i];foreground+=sizes[i];
  }
  result[0].count=static_cast<int64_t>(input.size())-foreground;
  return result;
}
// first[id - 1]: the id's first voxel in memory order (atomics only at run starts, as component_positions).
template<class Label> __global__ void first_voxels(const Label *labels,long long n,unsigned long long count,unsigned long long *first) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n) return;
  Label id=labels[i];
  if(id && static_cast<unsigned long long>(id)<=count && (i==0 || labels[i-1]!=id)) atomicMin(first+(static_cast<unsigned long long>(id)-1),static_cast<unsigned long long>(i));
}
template<class Label> void run_first_voxels(Context &ctx,const Array &input,Buffer<unsigned long long> &first) {
  first_voxels<<<static_cast<unsigned>((input.size()+255)/256),256,0,ctx.stream>>>(static_cast<const Label*>(input.data()),
    static_cast<long long>(input.size()),static_cast<unsigned long long>(first.size()),first.data());
  BROOK_CUDA(cudaGetLastError());
}
template<class Label> __global__ void crop(const Label *cc,const float *dbf,uint8_t *mask,float *out,
    long long vx,long long vy,long long sx,long long sy,long long sz,long long ox,long long oy,long long oz,Label label,
    const uint32_t *graph,uint32_t *graph_crop) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=sx*sy*sz) return;
  long long x,y,z;cuda::brook_unravel(i,sx,sy,x,y,z);
  long long j=(x+ox)+vx*((y+oy)+vy*(z+oz));
  bool live=cc[j]==label;mask[i]=live;out[i]=live?dbf[j]:0.0f;
  if(graph) graph_crop[i]=graph[j];
}
template<class Label> std::pair<Array,Array> run_crop(Context &ctx,const Array &cc,const Array &dbf,int64_t label,const Box &box,
    const Array *graph,Array *graph_crop) {
  std::array<int64_t,3> shape;
  for(int axis=0;axis<3;++axis) shape[axis]=box.hi[axis]-box.lo[axis]+1;
  auto mask=allocate(ctx,shape,BROOK_U8),distance=allocate(ctx,shape,BROOK_F32);
  if(graph) *graph_crop=allocate(ctx,shape,BROOK_U32);
  crop<<<static_cast<unsigned>((mask.size()+255)/256),256,0,ctx.stream>>>(
    static_cast<const Label*>(cc.data()),static_cast<const float*>(dbf.data()),static_cast<uint8_t*>(mask.data()),
    static_cast<float*>(distance.data()),cc.shape[0],cc.shape[1],shape[0],shape[1],shape[2],
    box.lo[0],box.lo[1],box.lo[2],static_cast<Label>(label),
    graph?static_cast<const uint32_t*>(graph->data()):nullptr,graph?static_cast<uint32_t*>(graph_crop->data()):nullptr);
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return {std::move(mask),std::move(distance)};
}
}
std::vector<Box> analyze(Context &ctx,const Array &cc,size_t count) {
  ctx.activate();
  for(auto axis:cc.shape) if(axis>INT32_MAX) throw std::invalid_argument("bounding-box axis exceeds int32");
  switch(cc.dtype) {
    case BROOK_U8:return run_analyze<uint8_t>(ctx,cc,count);
    case BROOK_U16:return run_analyze<uint16_t>(ctx,cc,count);
    case BROOK_U32:return run_analyze<uint32_t>(ctx,cc,count);
    case BROOK_U64:return run_analyze<uint64_t>(ctx,cc,count);
    default:throw std::invalid_argument("invalid component label dtype");
  }
}
void refresh_roots(Context &ctx,const Array &cc,std::vector<int64_t> &roots) {
  ctx.activate();
  if(roots.empty() || !cc.size()) return;
  Buffer<unsigned long long> first(ctx,roots.size());
  BROOK_CUDA(cudaMemsetAsync(first.data(),0xff,roots.size()*sizeof(unsigned long long),ctx.stream));
  switch(cc.dtype) {
    case BROOK_U8:run_first_voxels<uint8_t>(ctx,cc,first);break;
    case BROOK_U16:run_first_voxels<uint16_t>(ctx,cc,first);break;
    case BROOK_U32:run_first_voxels<uint32_t>(ctx,cc,first);break;
    case BROOK_U64:run_first_voxels<uint64_t>(ctx,cc,first);break;
    default:throw std::invalid_argument("invalid component label dtype");
  }
  auto found=first.get(ctx,roots.size());
  for(size_t id=0;id<roots.size();++id) if(found[id]!=ULLONG_MAX) roots[id]=static_cast<int64_t>(found[id]);
}
std::pair<Array,Array> crop_component(Context &ctx,const Array &cc,const Array &dbf,int64_t label,const Box &box,
    const Array *graph,Array *graph_crop) {
  ctx.activate();
  if(graph && (!graph_crop || graph->dtype!=BROOK_U32 || graph->shape!=cc.shape))
    throw std::invalid_argument("invalid voxel graph crop");
  switch(cc.dtype) {
    case BROOK_U8:return run_crop<uint8_t>(ctx,cc,dbf,label,box,graph,graph_crop);
    case BROOK_U16:return run_crop<uint16_t>(ctx,cc,dbf,label,box,graph,graph_crop);
    case BROOK_U32:return run_crop<uint32_t>(ctx,cc,dbf,label,box,graph,graph_crop);
    case BROOK_U64:return run_crop<uint64_t>(ctx,cc,dbf,label,box,graph,graph_crop);
    default:throw std::invalid_argument("invalid component label dtype");
  }
}
}
