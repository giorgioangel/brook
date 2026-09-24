// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include "ccl_kernels.cuh"
#include <algorithm>
#include <limits>

namespace brook {
namespace {
template<class Label, class Index>
__global__ void gather_mapping(const Label *labels, const Index *roots, long long *out, long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) out[i]=static_cast<long long>(labels[roots[i]]);
}
template<class Label, class Index, class Output>
Array relabel(Context &ctx, const Array &input, const Buffer<Index> &parent, brook_dtype dtype) {
  auto cc=allocate(ctx,input.shape,dtype);
  long long n=static_cast<long long>(input.size());
  cuda::ccl_relabel<Label,Index,Output><<<static_cast<unsigned>((n+255)/256),256,0,ctx.stream>>>(
    static_cast<const Label*>(input.data()),parent.data(),static_cast<Output*>(cc.data()),n);
  BROOK_CUDA(cudaGetLastError());
  return cc;
}
template<class Label, class Index>
Components run(Context &ctx, const Array &input) {
  const long long n=static_cast<long long>(input.size());
  const auto shape=input.shape;
  const unsigned grid=static_cast<unsigned>((n+255)/256);
  const Label *labels=static_cast<const Label*>(input.data());
  Buffer<Index> parent(ctx,n);
  cuda::ccl_init<Label,Index,unsigned char><<<grid,256,0,ctx.stream>>>(parent.data(),n);
  cuda::ccl_union<Label,Index,unsigned char><<<grid,256,0,ctx.stream>>>(labels,parent.data(),shape[0],shape[1],shape[2]);
  cuda::ccl_flatten<Label,Index,unsigned char><<<grid,256,0,ctx.stream>>>(parent.data(),n);
  Buffer<unsigned long long> counter(ctx,1); counter.clear(ctx);
  cuda::ccl_count_roots<Label,Index,unsigned char><<<grid,256,0,ctx.stream>>>(labels,parent.data(),n,counter.data());
  BROOK_CUDA(cudaGetLastError());
  const size_t count=counter.get(ctx,1)[0];
  Buffer<Index> roots(ctx,count);
  counter.clear(ctx);
  cuda::ccl_collect_roots<Label,Index,unsigned char><<<grid,256,0,ctx.stream>>>(labels,parent.data(),n,roots.data(),counter.data());
  BROOK_CUDA(cudaGetLastError());
  auto sorted=roots.get(ctx,count);
  std::sort(sorted.begin(),sorted.end());
  roots.set(ctx,sorted.data(),count);
  Components result;
  result.roots.assign(sorted.begin(),sorted.end());
  result.mapping.resize(count+1,0);
  if(count) {
    Buffer<long long> mapped(ctx,count);
    const unsigned blocks=static_cast<unsigned>((count+255)/256);
    gather_mapping<<<blocks,256,0,ctx.stream>>>(labels,roots.data(),mapped.data(),static_cast<long long>(count));
    cuda::ccl_mark_ranks<Label,Index,unsigned char><<<blocks,256,0,ctx.stream>>>(parent.data(),roots.data(),static_cast<long long>(count));
    BROOK_CUDA(cudaGetLastError());
    auto host=mapped.get(ctx,count);
    std::copy(host.begin(),host.end(),result.mapping.begin()+1);
  }
  if(count<=UINT8_MAX) result.labels=relabel<Label,Index,uint8_t>(ctx,input,parent,BROOK_U8);
  else if(count<=UINT16_MAX) result.labels=relabel<Label,Index,uint16_t>(ctx,input,parent,BROOK_U16);
  else if(count<=UINT32_MAX) result.labels=relabel<Label,Index,uint32_t>(ctx,input,parent,BROOK_U32);
  else result.labels=relabel<Label,Index,uint64_t>(ctx,input,parent,BROOK_U64);
  ctx.synchronize();
  return result;
}
template<class Label> Components dispatch_index(Context &ctx, const Array &input) {
  return input.size()<size_t{1}<<31 ? run<Label,int>(ctx,input) : run<Label,long long>(ctx,input);
}
}
Components connected_components(Context &ctx, const Array &input) {
  ctx.activate();
  if(!input.size()) return {allocate(ctx,input.shape,BROOK_U32),{0},{}};
  switch(input.dtype) {
    case BROOK_U8: return dispatch_index<uint8_t>(ctx,input);
    case BROOK_U16: return dispatch_index<uint16_t>(ctx,input);
    case BROOK_U32: return dispatch_index<uint32_t>(ctx,input);
    case BROOK_U64: return dispatch_index<uint64_t>(ctx,input);
    case BROOK_I8: return dispatch_index<int8_t>(ctx,input);
    case BROOK_I16: return dispatch_index<int16_t>(ctx,input);
    case BROOK_I32: return dispatch_index<int32_t>(ctx,input);
    case BROOK_I64: return dispatch_index<int64_t>(ctx,input);
    default: throw std::invalid_argument("connected components requires integer labels");
  }
}
}
