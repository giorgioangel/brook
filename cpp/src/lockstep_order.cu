// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "lockstep_order.hpp"
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_lockstep_order
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <numeric>
#include <cstdlib>
namespace cub=brook_cccl_lockstep_order::cub;
namespace thrust=brook_cccl_lockstep_order::thrust;

namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
template<class T> Buffer<T> upload(Context &ctx,const std::vector<T> &values) {
  Buffer<T> out(ctx,values.size());out.set(ctx,values.data(),values.size());ctx.synchronize();return out;
}
struct Reverse { int n;__device__ int operator()(int i) const { return n-1-i; } };
struct ComponentKey {
  const uint8_t *mask;const float *daf;
  __device__ unsigned long long operator()(int i) const {
    if(!mask[i]) return 0;
    float d=daf[i];if(isinf(d) || d==0) d=0;
    // DAF is nonnegative. Complement both words for descending distance/index.
    // The index is below 2^31, so a foreground key can never be the zero sentinel.
    return (static_cast<unsigned long long>(0xffffffffu-__float_as_uint(d))<<32)|(0xffffffffu-unsigned(i));
  }
};
struct NonzeroKey {__device__ bool operator()(unsigned long long key) const {return key!=0;}};
template<class Label> struct Selected {
  const Label *labels;const uint8_t *eligible;int lo,hi;
  __device__ bool operator()(int i) const { int label=int(labels[i]);return label>=lo && label<=hi && eligible[label]; }
};
template<class Label> __global__ void keys(const Label *labels,const float *daf,const int *indices,
    unsigned long long *out,int count,bool single) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  // An unreached voxel (a voxel graph can cut one off) orders as DAF 0, as ComponentKey does.
  int v=indices[i];float d=daf[v];if(isinf(d)) d=0;unsigned distance=0xffffffffu-__float_as_uint(d);
  out[i]=single?(static_cast<unsigned long long>(distance)<<32)|(0xffffffffu-unsigned(v)):
    (static_cast<unsigned long long>(labels[v])<<32)|distance;
}
__global__ void unpack(const unsigned long long *keys,int *out,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) out[i]=int(0xffffffffu-unsigned(keys[i]));
}
template<class Label> TargetOrder run(Context &ctx,const Array &labels,const Array &daf,
    const std::vector<uint8_t> &eligible,const std::vector<int64_t> &counts,bool fix) {
  TargetOrder result;result.host_start.resize(counts.size(),0);result.host_end.resize(counts.size(),0);
  std::vector<int> active;int64_t foreground=0;
  for(size_t id=1;id<eligible.size();++id) if(eligible[id]) {
    active.push_back(int(id));result.host_start[id]=foreground;foreground+=counts[id];result.host_end[id]=foreground;
  }
  result.order=Buffer<int>(ctx,foreground);
  result.start=upload(ctx,result.host_start);result.end=upload(ctx,result.host_end);
  auto flags=upload(ctx,eligible);
  size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
  double future=double(labels.size())*(12+(fix?4:0));
  const char *bounded=std::getenv("BROOK_ORDER_BUDGET");
  bool bounded_sort=!bounded || std::string(bounded)!="0";
  size_t budget=bounded_sort?std::min(free/4,size_t(future)):size_t(std::max(double(size_t{16}<<20),0.85*double(free)-future));
  size_t group=bounded_sort?std::max(budget/64,size_t{1}<<20):std::max(size_t{1},budget/32);
  ctx.stats["target_sort_group_voxels"]=int64_t(group);
  const Label *lab=static_cast<const Label*>(labels.data());
  auto sequence=thrust::make_transform_iterator(thrust::counting_iterator<int>(0),Reverse{int(labels.size())});
  for(size_t first=0;first<active.size();) {
    size_t end=first;int64_t count=0;
    while(end<active.size() && (end==first || count+counts[active[end]]<=int64_t(group))) count+=counts[active[end++]];
    Buffer<int> indices(ctx,count),selected(ctx,1);
    Selected<Label> predicate{lab,flags.data(),active[first],active[end-1]};size_t bytes=0;
    BROOK_CUDA(cub::DeviceSelect::If(nullptr,bytes,sequence,indices.data(),selected.data(),int(labels.size()),predicate,ctx.stream));
    Buffer<uint8_t> scratch(ctx,bytes);
    BROOK_CUDA(cub::DeviceSelect::If(scratch.data(),bytes,sequence,indices.data(),selected.data(),int(labels.size()),predicate,ctx.stream));
    if(selected.get(ctx,1)[0]!=count) throw std::runtime_error("target order and component counts disagree");
    scratch={};
    Buffer<unsigned long long> input_keys(ctx,count),output_keys(ctx,count);
    bool single=end-first==1;
    keys<<<blocks(count),256,0,ctx.stream>>>(lab,static_cast<const float*>(daf.data()),indices.data(),input_keys.data(),int(count),single);
    BROOK_CUDA(cudaGetLastError());
    int *output=result.order.data()+result.host_start[active[first]];
    bytes=0;
    if(single) {
      indices={};
      BROOK_CUDA(cub::DeviceRadixSort::SortKeys(nullptr,bytes,input_keys.data(),output_keys.data(),int(count),0,64,ctx.stream));
      scratch=Buffer<uint8_t>(ctx,bytes);
      BROOK_CUDA(cub::DeviceRadixSort::SortKeys(scratch.data(),bytes,input_keys.data(),output_keys.data(),int(count),0,64,ctx.stream));
      unpack<<<blocks(count),256,0,ctx.stream>>>(output_keys.data(),output,int(count));
      BROOK_CUDA(cudaGetLastError());
    } else {
      BROOK_CUDA(cub::DeviceRadixSort::SortPairs(nullptr,bytes,input_keys.data(),output_keys.data(),indices.data(),output,int(count),0,64,ctx.stream));
      scratch=Buffer<uint8_t>(ctx,bytes);
      BROOK_CUDA(cub::DeviceRadixSort::SortPairs(scratch.data(),bytes,input_keys.data(),output_keys.data(),indices.data(),output,int(count),0,64,ctx.stream));
    }
    ctx.synchronize();first=end;
  }
  return result;
}
template<class Label> __global__ void set_alive(const Label *labels,const uint8_t *eligible,uint8_t *alive,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) alive[i]=eligible[labels[i]];
}
template<class Label> void alive(Context &ctx,const Array &labels,const Buffer<uint8_t> &eligible,Buffer<uint8_t> &out) {
  set_alive<<<blocks(labels.size()),256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),eligible.data(),out.data(),int(labels.size()));
  BROOK_CUDA(cudaGetLastError());
}
template<class Label> __global__ void count_removed(const Label *labels,const int *touched,int n,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) atomicAdd(counts+labels[touched[i]],1);
}
}

TargetOrder build_target_order(Context &ctx,const Array &labels,const Array &daf,const std::vector<uint8_t> &eligible,
                              const std::vector<int64_t> &counts,bool fix) {
  switch(labels.dtype) {
    case BROOK_U8:return run<uint8_t>(ctx,labels,daf,eligible,counts,fix);
    case BROOK_U16:return run<uint16_t>(ctx,labels,daf,eligible,counts,fix);
    case BROOK_U32:return run<uint32_t>(ctx,labels,daf,eligible,counts,fix);
    case BROOK_U64:return run<uint64_t>(ctx,labels,daf,eligible,counts,fix);
    default:throw std::invalid_argument("invalid target order label dtype");
  }
}
Buffer<int> build_component_target_order(Context &ctx,const Array &mask,const Array &daf,size_t foreground) {
  if(mask.dtype!=BROOK_U8 || daf.dtype!=BROOK_F32 || mask.shape!=daf.shape || mask.device!=daf.device ||
     mask.size()>=size_t{1}<<31 || foreground>mask.size()) throw std::invalid_argument("invalid component target order inputs");
  Buffer<int> order(ctx,foreground);
  if(!foreground) return order;
  Buffer<unsigned long long> input(ctx,foreground),output(ctx,foreground);
  Buffer<int> selected(ctx,1);
  auto sequence=thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
    ComponentKey{static_cast<const uint8_t*>(mask.data()),static_cast<const float*>(daf.data())});
  size_t bytes=0;
  BROOK_CUDA(cub::DeviceSelect::If(nullptr,bytes,sequence,input.data(),selected.data(),int(mask.size()),NonzeroKey{},ctx.stream));
  Buffer<uint8_t> scratch(ctx,bytes);
  BROOK_CUDA(cub::DeviceSelect::If(scratch.data(),bytes,sequence,input.data(),selected.data(),int(mask.size()),NonzeroKey{},ctx.stream));
  if(selected.get(ctx,1)[0]!=int(foreground)) throw std::runtime_error("component target order and foreground count disagree");
  scratch={};bytes=0;
  BROOK_CUDA(cub::DeviceRadixSort::SortKeys(nullptr,bytes,input.data(),output.data(),int(foreground),0,64,ctx.stream));
  scratch=Buffer<uint8_t>(ctx,bytes);
  BROOK_CUDA(cub::DeviceRadixSort::SortKeys(scratch.data(),bytes,input.data(),output.data(),int(foreground),0,64,ctx.stream));
  unpack<<<blocks(foreground),256,0,ctx.stream>>>(output.data(),order.data(),int(foreground));
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return order;
}
void initialize_alive(Context &ctx,const Array &labels,const std::vector<uint8_t> &eligible,Buffer<uint8_t> &out) {
  auto flags=upload(ctx,eligible);
  switch(labels.dtype) {
    case BROOK_U8:alive<uint8_t>(ctx,labels,flags,out);break;
    case BROOK_U16:alive<uint16_t>(ctx,labels,flags,out);break;
    case BROOK_U32:alive<uint32_t>(ctx,labels,flags,out);break;
    case BROOK_U64:alive<uint64_t>(ctx,labels,flags,out);break;
    default:throw std::invalid_argument("invalid alive label dtype");
  }
  ctx.synchronize();
}
std::vector<int> removed_per_label(Context &ctx,const Array &labels,const Buffer<int> &touched,int removed,size_t count) {
  Buffer<int> out(ctx,count);out.clear(ctx);
  if(removed) switch(labels.dtype) {
    case BROOK_U8:count_removed<<<blocks(removed),256,0,ctx.stream>>>(static_cast<const uint8_t*>(labels.data()),touched.data(),removed,out.data());break;
    case BROOK_U16:count_removed<<<blocks(removed),256,0,ctx.stream>>>(static_cast<const uint16_t*>(labels.data()),touched.data(),removed,out.data());break;
    case BROOK_U32:count_removed<<<blocks(removed),256,0,ctx.stream>>>(static_cast<const uint32_t*>(labels.data()),touched.data(),removed,out.data());break;
    case BROOK_U64:count_removed<<<blocks(removed),256,0,ctx.stream>>>(static_cast<const uint64_t*>(labels.data()),touched.data(),removed,out.data());break;
    default:throw std::invalid_argument("invalid removed label dtype");
  }
  BROOK_CUDA(cudaGetLastError());return out.get(ctx,count);
}
}
