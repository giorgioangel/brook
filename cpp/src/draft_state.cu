// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "draft_state.hpp"

namespace brook {
namespace {
__global__ void clean_distance(float *distance,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) distance[i]=__int_as_float(0x7f800000);
}
void append_zeros(Context &ctx,Array &array,int dense,int reserve) {
  auto next=allocate(ctx,{int64_t(dense)+reserve,1,1},array.dtype);
  size_t prefix=size_t(dense)*dtype_size(array.dtype),extra=size_t(reserve)*dtype_size(array.dtype);
  if(prefix) BROOK_CUDA(cudaMemcpyAsync(next.data(),array.data(),prefix,cudaMemcpyDeviceToDevice,ctx.stream));
  BROOK_CUDA(cudaMemsetAsync(static_cast<char*>(next.data())+prefix,0,extra,ctx.stream));
  ctx.synchronize();array=std::move(next);
}
}
void append_draft_reserve(Context &ctx,DraftVoxelBuffers s,int dense,int foreground,int reserve) {
  if(dense<0 || foreground<0 || foreground>dense || reserve<=0 || reserve>INT32_MAX-dense)
    throw std::invalid_argument("invalid draft reserve size");
  if(s.labels.size()!=size_t(dense) || s.dbf.size()!=size_t(dense) || s.penalty.size()!=size_t(dense) ||
      s.alive.size()!=size_t(dense) || s.key.size()!=size_t(dense) || s.flags.size()!=size_t(dense) || s.distance.size()!=size_t(dense) ||
      s.dbf.dtype!=BROOK_F32 || s.penalty.dtype!=BROOK_F32)
    throw std::invalid_argument("invalid dense draft state");
  if(s.labels.dtype!=BROOK_U8 && s.labels.dtype!=BROOK_U16 && s.labels.dtype!=BROOK_U32 && s.labels.dtype!=BROOK_U64)
    throw std::invalid_argument("draft labels must be unsigned integers");
  if(s.labels.device!=ctx.device || s.dbf.device!=ctx.device || s.penalty.device!=ctx.device || &s.dbf==&s.penalty)
    throw std::invalid_argument("invalid draft voxel ownership");
  const bool graph=s.graph && s.graph->storage;
  if(graph && (s.graph->size()!=size_t(dense) || s.graph->dtype!=BROOK_U32 || s.graph->device!=ctx.device))
    throw std::invalid_argument("invalid dense draft graph");
  for(size_t i=0;i<s.worklists.size();++i) {
    auto list=s.worklists[i];if(!list || list->size()<size_t(foreground)) throw std::invalid_argument("invalid draft worklist");
    for(size_t j=0;j<i;++j) if(list==s.worklists[j]) throw std::invalid_argument("draft worklists must be distinct");
  }
  ctx.synchronize();
  // Release the clean state first, so growing never holds old and new copies of it at once.
  s.key={};s.flags={};s.distance={};
  for(auto list:s.worklists) *list={};
  append_zeros(ctx,s.labels,dense,reserve);
  append_zeros(ctx,s.dbf,dense,reserve);
  {
    Buffer<uint8_t> next(ctx,size_t(dense)+reserve);
    if(dense) BROOK_CUDA(cudaMemcpyAsync(next.data(),s.alive.data(),dense,cudaMemcpyDeviceToDevice,ctx.stream));
    BROOK_CUDA(cudaMemsetAsync(next.data()+dense,0,reserve,ctx.stream));ctx.synchronize();s.alive=std::move(next);
  }
  append_zeros(ctx,s.penalty,dense,reserve);
  if(graph) append_zeros(ctx,*s.graph,dense,reserve);
  int n=dense+reserve;
  s.key=Buffer<unsigned long long>(ctx,n);BROOK_CUDA(cudaMemsetAsync(s.key.data(),0xff,size_t(n)*8,ctx.stream));
  s.flags=Buffer<int>(ctx,n);s.flags.clear(ctx);
  s.distance=Buffer<float>(ctx,n);clean_distance<<<(n+255LL)/256,256,0,ctx.stream>>>(s.distance.data(),n);
  for(auto list:s.worklists) *list=Buffer<int>(ctx,size_t(foreground)+reserve,true);
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();
}
}
