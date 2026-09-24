// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "voxel_graph.hpp"
#include "ccl_kernels.cuh"
#include "graph.hpp"
// Keep header-instantiated toolkit kernels private to this translation unit.
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_graph_components
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
namespace cub=brook_cccl_graph_components::cub;
namespace thrust=brook_cccl_graph_components::thrust;

namespace brook {
namespace {
struct GraphColoringTag {};
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
template<class Index> __global__ void unite(const void *graph,Index *parent,long long sx,long long sy,long long sz,bool compact_2d) {
  long long v=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(v>=sx*sy*sz) return;
  long long x,y,z;cuda::brook_unravel(v,sx,sy,x,y,z);
  uint32_t edges=compact_2d?static_cast<const uint8_t*>(graph)[v]:static_cast<const uint32_t*>(graph)[v];
  for(int k=0;k<26;++k) {
    int bit=cuda::BROOK_BIT[k];if(compact_2d && bit>=6) bit-=2;
    if(!(edges&(1u<<bit))) continue;
    long long nx=x+cuda::BROOK_NX[k],ny=y+cuda::BROOK_NY[k],nz=z+cuda::BROOK_NZ[k];
    if(nx<0 || ny<0 || nz<0 || nx>=sx || ny>=sy || nz>=sz) continue;
    long long u=nx+sx*(ny+sy*nz);
    if(u<v) cuda::brook_union(parent,Index(v),Index(u));
  }
}
template<class Index> struct RootFlag {
  const Index *parent;
  __device__ Index operator()(Index i) const { return parent[i]==i?1:0; }
};
template<class Index> __global__ void add_prefix(Index *prefix,long long begin,long long count) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count) prefix[begin+i]+=prefix[begin-1];
}
template<class Label,class Index> __global__ void color(const Label *labels,const Index *parent,const Index *prefix,
    Index *output,long long n,unsigned long long *maximum) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n) return;
  if(labels[i]!=labels[0] && !__ldcg(maximum+1)) atomicExch(maximum+1,1ULL);
  Index id=labels[i]>0?prefix[parent[i]]:0;output[i]=id;
  if(id && static_cast<unsigned long long>(id)>__ldcg(maximum)) atomicMax(maximum,static_cast<unsigned long long>(id));
}
// Compact IDs. The root scan above ranks every root, background singletons included, so the ids
// `color` assigns are sparse (gaps of the background before each component) and every per-id
// table downstream would be sized by the background. The used ids are renumbered 1..count in
// their order, which keeps the components' relative order (the order of their roots).
template<class Index> __global__ void mark_present(const Index *colors,Index *present,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n && colors[i]) present[colors[i]]=Index(1);
}
template<class Index> __global__ void recolor(Index *colors,const Index *compact,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n && colors[i]) colors[i]=compact[colors[i]];
}
// last_run[id]: the start of the id's last run of voxels (its original label is read there);
// roots[id - 1]: the component's lowest foreground voxel (its union-find root unless background joins it).
template<class Index> __global__ void component_positions(const Index *colors,long long n,long long *last_run,Index *roots) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n) return;
  Index id=colors[i];if(!id) return;
  if(i==0 || colors[i-1]!=id) atomicMax(last_run+id,i);
  atomicMin(roots+(id-1),Index(i));
}
// keys[c]: the sparse id of compact id c, the rank of its graph component among all of them (background
// ones included) by lowest voxel: cc3d.color_connectivity_graph's numbering, which Kimimaro keeps through
// its avocado stage. Present sparse ids map one to one to compact ids, so each slot has one writer.
template<class Index> __global__ void sparse_keys(const Index *present,const Index *compact,long long *keys,long long n,long long count) {
  long long s=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(s<1 || s>n || !present[s]) return;
  long long c=compact[s];
  if(c>=1 && c<=count) keys[c]=s;
}
template<class Label> __global__ void gather_mapping(const Label *labels,const long long *positions,long long *mapping,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) mapping[i]=i && positions[i]>=0?static_cast<long long>(labels[positions[i]]):0;
}
template<class Index,class Output> __global__ void pack_colors(const Index *input,Output *output,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n) output[i]=Output(input[i]);
}
template<class Label,class Index> Components run(Context &ctx,const Array &labels,const Array &vg,bool capture) {
  const long long n=static_cast<long long>(labels.size());auto s=labels.shape;
  // parent / prefix hold n + 1 entries: after the coloring they are reused as the presence flags
  // and the compact numbering of the sparse ids, which run up to n.
  Buffer<Index> parent(ctx,n+1),prefix(ctx,n+1),colors(ctx,n);
  Buffer<unsigned long long> maximum(ctx,2);
  using Sequence=thrust::counting_iterator<Index>;
  auto indicators=thrust::make_transform_iterator(Sequence(0),RootFlag<Index>{parent.data()});
  const long long chunk=INT32_MAX;
  size_t scratch_bytes=0,flag_bytes=0;
  BROOK_CUDA(cub::DeviceScan::InclusiveSum(nullptr,scratch_bytes,indicators,prefix.data(),int(std::min(n,chunk)),ctx.stream));
  BROOK_CUDA(cub::DeviceScan::InclusiveSum(nullptr,flag_bytes,parent.data(),prefix.data(),int(std::min(n+1,chunk)),ctx.stream));
  Buffer<uint8_t> scratch(ctx,std::max(scratch_bytes,flag_bytes));
  auto body=[&] {
    maximum.clear(ctx);
    cuda::ccl_init<GraphColoringTag,Index,uint8_t><<<blocks(n),256,0,ctx.stream>>>(parent.data(),n);
    unite<<<blocks(n),256,0,ctx.stream>>>(vg.data(),parent.data(),s[0],s[1],s[2],vg.dtype==BROOK_U8);
    cuda::ccl_flatten<GraphColoringTag,Index,uint8_t><<<blocks(n),256,0,ctx.stream>>>(parent.data(),n);
    BROOK_CUDA(cudaGetLastError());
    for(long long start=0;start<n;start+=chunk) {
      int count=int(std::min(chunk,n-start));
      auto input=thrust::make_transform_iterator(Sequence(Index(start)),RootFlag<Index>{parent.data()});
      BROOK_CUDA(cub::DeviceScan::InclusiveSum(scratch.data(),scratch_bytes,input,prefix.data()+start,count,ctx.stream));
      if(start) add_prefix<<<blocks(count),256,0,ctx.stream>>>(prefix.data(),start,count);
    }
    color<<<blocks(n),256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),parent.data(),prefix.data(),colors.data(),n,maximum.data());
    BROOK_CUDA(cudaGetLastError());
    // compact the ids (see mark_present): prefix[id] = the compact id of a present sparse id
    BROOK_CUDA(cudaMemsetAsync(parent.data(),0,size_t(n+1)*sizeof(Index),ctx.stream));
    mark_present<<<blocks(n),256,0,ctx.stream>>>(colors.data(),parent.data(),n);
    BROOK_CUDA(cudaGetLastError());
    for(long long start=0;start<n+1;start+=chunk) {
      int count=int(std::min(chunk,n+1-start));
      BROOK_CUDA(cub::DeviceScan::InclusiveSum(scratch.data(),flag_bytes,parent.data()+start,prefix.data()+start,count,ctx.stream));
      if(start) add_prefix<<<blocks(count),256,0,ctx.stream>>>(prefix.data(),start,count);
    }
    recolor<<<blocks(n),256,0,ctx.stream>>>(colors.data(),prefix.data(),n);
    BROOK_CUDA(cudaGetLastError());
  };
  if(capture) { FixedGraph graph(ctx);graph.capture(body);graph.run(); }
  else { body();ctx.synchronize(); }
  auto info=maximum.get(ctx,2);
  Index compact=0;
  BROOK_CUDA(cudaMemcpyAsync(&compact,prefix.data()+n,sizeof(Index),cudaMemcpyDeviceToHost,ctx.stream));ctx.synchronize();
  const unsigned long long count=static_cast<unsigned long long>(compact);
  if(count>UINT32_MAX) throw std::invalid_argument("voxel graph component count exceeds uint32");
  brook_dtype dtype=count<=UINT8_MAX?BROOK_U8:(count<=UINT16_MAX?BROOK_U16:BROOK_U32);
  Components result;result.labels=allocate(ctx,labels.shape,dtype);result.input_uniform=info[1]==0;
  Buffer<long long> positions(ctx,count+1),mapping(ctx,count+1);Buffer<Index> roots(ctx,count);
  BROOK_CUDA(cudaMemsetAsync(positions.data(),0xff,(count+1)*sizeof(long long),ctx.stream));
  if(count) BROOK_CUDA(cudaMemsetAsync(roots.data(),0x7f,count*sizeof(Index),ctx.stream));   // above every voxel index
  component_positions<<<blocks(n),256,0,ctx.stream>>>(colors.data(),n,positions.data(),roots.data());
  gather_mapping<<<blocks(count+1),256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),positions.data(),mapping.data(),static_cast<long long>(count+1));
  if(dtype==BROOK_U8) pack_colors<<<blocks(n),256,0,ctx.stream>>>(colors.data(),static_cast<uint8_t*>(result.labels.data()),n);
  else if(dtype==BROOK_U16) pack_colors<<<blocks(n),256,0,ctx.stream>>>(colors.data(),static_cast<uint16_t*>(result.labels.data()),n);
  else pack_colors<<<blocks(n),256,0,ctx.stream>>>(colors.data(),static_cast<uint32_t*>(result.labels.data()),n);
  // after the body, parent holds the presence flags of the sparse ids and prefix their compact numbering
  Buffer<long long> keys(ctx,count+1);keys.clear(ctx);
  if(count) sparse_keys<<<blocks(n+1),256,0,ctx.stream>>>(parent.data(),prefix.data(),keys.data(),n,static_cast<long long>(count));
  BROOK_CUDA(cudaGetLastError());
  auto m=mapping.get(ctx,count+1);auto r=roots.get(ctx,count);auto k=keys.get(ctx,count+1);
  result.mapping.assign(m.begin(),m.end());result.roots.assign(r.begin(),r.end());result.keys.assign(k.begin(),k.end());return result;
}
template<class Label> Components dispatch(Context &ctx,const Array &labels,const Array &graph,bool capture) {
  return labels.size()<(size_t{1}<<31)?run<Label,int>(ctx,labels,graph,capture):run<Label,long long>(ctx,labels,graph,capture);
}
__global__ void widen_graph(const uint8_t *input,uint32_t *output,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) output[i]=input[i];
}
}
Array graph_for_trace(Context &ctx,const Array &graph) {
  if(graph.dtype==BROOK_U32) return graph;
  if(graph.dtype!=BROOK_U8 || graph.shape[2]!=1)
    throw std::invalid_argument("voxel_graph requires uint32, or uint8 for an XY plane");
  auto result=allocate(ctx,graph.shape,BROOK_U32);
  if(graph.size()) widen_graph<<<blocks(graph.size()),256,0,ctx.stream>>>(static_cast<const uint8_t*>(graph.data()),
    static_cast<uint32_t*>(result.data()),graph.size());
  BROOK_CUDA(cudaGetLastError());return result;
}
Components graph_components(Context &ctx,const Array &labels,const Array &graph,bool capture) {
  ctx.activate();
  if((graph.dtype!=BROOK_U32 && !(graph.dtype==BROOK_U8 && graph.shape[2]==1)) || graph.shape!=labels.shape)
    throw std::invalid_argument("voxel_graph must match the label shape and use uint32, or uint8 for an XY plane");
  if(!labels.size()) return {allocate(ctx,labels.shape,BROOK_U8),{0},{}};
  switch(labels.dtype) {
    case BROOK_U8:return dispatch<uint8_t>(ctx,labels,graph,capture);
    case BROOK_U16:return dispatch<uint16_t>(ctx,labels,graph,capture);
    case BROOK_U32:return dispatch<uint32_t>(ctx,labels,graph,capture);
    case BROOK_U64:return dispatch<uint64_t>(ctx,labels,graph,capture);
    case BROOK_I8:return dispatch<int8_t>(ctx,labels,graph,capture);
    case BROOK_I16:return dispatch<int16_t>(ctx,labels,graph,capture);
    case BROOK_I32:return dispatch<int32_t>(ctx,labels,graph,capture);
    case BROOK_I64:return dispatch<int64_t>(ctx,labels,graph,capture);
    default:throw std::invalid_argument("voxel graph labels must be integers");
  }
}
}
