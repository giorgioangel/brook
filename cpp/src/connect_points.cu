// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// Reproduces the behaviour of Kimimaro 5.8.1 connect_points (kimimaro/intake.py; cpp/licenses/Kimimaro.txt).
#include "point_path.hpp"
#include "trace_primitives.hpp"
#include <algorithm>
#include <cmath>

namespace brook {
namespace {
template<class T> __global__ void binary(const T *input,uint8_t *output,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) output[i]=input[i]!=T(0);
}
Array mask_input(Context &ctx,const Array &input) {
  auto output=allocate(ctx,input.shape,BROOK_U8);
  auto launch=[&](auto value) {using T=decltype(value);
    binary<<<unsigned((input.size()+255)/256),256,0,ctx.stream>>>(static_cast<const T*>(input.data()),static_cast<uint8_t*>(output.data()),int(input.size()));};
  switch(input.dtype) {
    case BROOK_U8:launch(uint8_t{});break;case BROOK_U16:launch(uint16_t{});break;
    case BROOK_U32:launch(uint32_t{});break;case BROOK_U64:launch(uint64_t{});break;
    case BROOK_I8:launch(int8_t{});break;case BROOK_I16:launch(int16_t{});break;
    case BROOK_I32:launch(int32_t{});break;case BROOK_I64:launch(int64_t{});break;
    case BROOK_F32:launch(float{});break;case BROOK_F64:launch(double{});break;
  }
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return output;
}
}
Skeleton connect_points(Context &ctx,Array labels,Point start,Point end,std::array<float,3> spacing,
                        double scale,double exponent) {
  ctx.activate();auto shape=labels.shape;size_t n=labels.size();
  if(!n || n>=size_t{1}<<31) throw std::invalid_argument("point volume must contain 1..2^31-1 voxels");
  for(int a=0;a<3;++a) {
    if(start[a]<0 || end[a]<0 || start[a]>=shape[a] || end[a]>=shape[a]) throw std::invalid_argument("point outside volume");
    if(!std::isfinite(spacing[a]) || spacing[a]<=0) throw std::invalid_argument("anisotropy must be finite and positive");
  }
  int64_t begin=flatten(start,shape),finish=flatten(end,shape);
  auto mask=mask_input(ctx,labels);labels={};
  {
    auto cc=connected_components(ctx,mask);uint64_t ids[2]={};size_t item=dtype_size(cc.labels.dtype);
    auto ptr=static_cast<const uint8_t*>(cc.labels.data());
    BROOK_CUDA(cudaMemcpyAsync(ids,ptr+begin*item,item,cudaMemcpyDeviceToHost,ctx.stream));
    BROOK_CUDA(cudaMemcpyAsync(ids+1,ptr+finish*item,item,cudaMemcpyDeviceToHost,ctx.stream));ctx.synchronize();
    if(!ids[0] || ids[0]!=ids[1]) throw std::invalid_argument("Cannot extract centerline from disconnected components.");
  }
  // kimimaro.connect_points copies its mask to Fortran order (format_labels), so the
  // EDT passes run in Fortran axis order whatever the host layout.
  auto dbf=edt(ctx,mask,spacing,true);
  auto summary=summarize_component(ctx,mask,dbf);float maximum=summary.maximum;
  auto df=euclidean_distance_field(ctx,mask,begin,spacing,0,nullptr,true,summary.valid);mask={};
  std::vector<float> boundary(n);
  dbf.copy_to_host(boundary.data(),dbf.bytes());dbf={};
  std::vector<float> penalty(n);
  df.distance.copy_to_host(penalty.data(),df.distance.bytes());df.distance={};
  float farthest=penalty[df.maximum];
  // Kimimaro's compute_pdrf computes these coefficients as NumPy float32 scalars.
  float m=1.0f/std::pow(maximum,1.01f),inverse=farthest?1.0f/farthest:0.0f;
  int squares=-1;
  if(exponent>0 && exponent<65536 && std::floor(exponent)==exponent && !(int(exponent)&(int(exponent)-1))) {
    squares=0;for(int e=int(exponent);e>1;e>>=1) ++squares;
  }
  for(size_t i=0;i<n;++i) {
    if(boundary[i]==0) boundary[i]=INFINITY;
    float value=1.0f-boundary[i]*m;
    if(squares>=0) for(int k=0;k<squares;++k) value*=value;
    else value=std::pow(value,float(exponent));
    value*=float(scale);
    if(inverse!=0) {float d=std::isinf(penalty[i])?0:penalty[i];value+=d*inverse;}
    penalty[i]=value;
  }
  auto path=point_path_heap(penalty,shape,finish,begin);
  Skeleton result;result.anisotropy=spacing;result.vertices.reserve(path.size());result.radii.reserve(path.size());
  for(size_t i=0;i<path.size();++i) {
    auto p=unravel(path[i],shape);result.vertices.push_back({float(p[0])*spacing[0],float(p[1])*spacing[1],float(p[2])*spacing[2]});
    result.radii.push_back(boundary[path[i]]);if(i) result.edges.push_back({uint32_t(i-1),uint32_t(i)});
  }
  return result;
}
}
