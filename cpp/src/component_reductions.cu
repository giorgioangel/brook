// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "trace_primitives.hpp"
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_component_reductions
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cmath>

namespace cub=brook_cccl_component_reductions::cub;
namespace thrust=brook_cccl_component_reductions::thrust;
namespace brook {
namespace {
struct Sample {
  const uint8_t *mask;const float *dbf;
  __device__ ComponentSummary operator()(int i) const {
    float maximum=dbf[i];
    // std::max_element keeps an initial NaN, and ignores later NaNs.
    if(i && isnan(maximum)) maximum=-INFINITY;
    return {maximum,int(mask[i]!=0),mask[i]?i:INT32_MAX};
  }
};
struct Combine {
  __device__ ComponentSummary operator()(ComponentSummary a,ComponentSummary b) const {
    float maximum=isnan(a.maximum)?a.maximum:(isnan(b.maximum)?b.maximum:(a.maximum<b.maximum?b.maximum:a.maximum));
    return {maximum,a.valid+b.valid,min(a.first,b.first)};
  }
};
struct MaximumFlag {
  const float *dbf;float maximum;
  __device__ int operator()(int i) const { return dbf[i]==maximum; }
};
void validate(const Array &dbf) {
  if(dbf.dtype!=BROOK_F32 || dbf.size()>=size_t{1}<<31) throw std::invalid_argument("invalid component DBF");
}
}
ComponentSummary summarize_component(Context &ctx,const Array &mask,const Array &dbf) {
  ctx.activate();validate(dbf);
  if(mask.dtype!=BROOK_U8 || mask.shape!=dbf.shape) throw std::invalid_argument("invalid component mask");
  if(!mask.size()) return {0,0,-1};
  auto samples=thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
    Sample{static_cast<const uint8_t*>(mask.data()),static_cast<const float*>(dbf.data())});
  Buffer<ComponentSummary> result(ctx,1);size_t bytes=0;
  ComponentSummary initial{-INFINITY,0,INT32_MAX};
  BROOK_CUDA(cub::DeviceReduce::Reduce(nullptr,bytes,samples,result.data(),int(mask.size()),Combine{},initial,ctx.stream));
  Buffer<uint8_t> scratch(ctx,bytes);
  BROOK_CUDA(cub::DeviceReduce::Reduce(scratch.data(),bytes,samples,result.data(),int(mask.size()),Combine{},initial,ctx.stream));
  return result.get(ctx,1)[0];
}
std::vector<int> maximum_positions(Context &ctx,const Array &dbf,float maximum) {
  ctx.activate();validate(dbf);
  if(!dbf.size()) return {};
  auto sequence=thrust::counting_iterator<int>(0);
  auto matches=thrust::make_transform_iterator(sequence,MaximumFlag{static_cast<const float*>(dbf.data()),maximum});
  Buffer<int> count(ctx,1);size_t bytes=0;
  BROOK_CUDA(cub::DeviceReduce::Sum(nullptr,bytes,matches,count.data(),int(dbf.size()),ctx.stream));
  Buffer<uint8_t> scratch(ctx,bytes);
  BROOK_CUDA(cub::DeviceReduce::Sum(scratch.data(),bytes,matches,count.data(),int(dbf.size()),ctx.stream));
  int found=count.get(ctx,1)[0];if(!found) return {};
  Buffer<int> indices(ctx,found);
  BROOK_CUDA(cub::DeviceSelect::Flagged(nullptr,bytes,sequence,matches,indices.data(),count.data(),int(dbf.size()),ctx.stream));
  if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
  BROOK_CUDA(cub::DeviceSelect::Flagged(scratch.data(),bytes,sequence,matches,indices.data(),count.data(),int(dbf.size()),ctx.stream));
  return indices.get(ctx,found);
}
}
