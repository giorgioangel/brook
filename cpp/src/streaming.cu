// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "streaming.hpp"
#include "edt_kernels.cuh"
#include "edt_bounds.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace brook {
namespace {
// Distinct argument type prevents duplicate kernel registrations across CUDA TUs.
template<class Label> struct StreamLabels {
  const Label *data;
  __device__ Label operator[](long long i) const {return data[i];}
};
template<class Label> __global__ void finish_streamed(const Label *labels,float *distance,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) distance[i]=labels[i]==0?0.0f:sqrtf(distance[i]);
}

template<class Label> void passes(Context &ctx,const Array &labels,Array &field,
    std::array<float,3> spacing,bool black_border,bool xy,size_t budget) {
  const auto s=labels.shape;
  const auto *lab=static_cast<const Label*>(labels.data());
  const StreamLabels<Label> source{lab};
  auto *dist=static_cast<float*>(field.data());
  const char *mode=std::getenv("BROOK_EDT_INDEX32");
  bool narrow=edt_index32_volume_fits(field.size()) && (!mode || std::string(mode)!="0");
  Buffer<float> table;
  if(xy) {
    std::vector<float> values(size_t(s[0])+2,0.0f);
    for(size_t i=1;i<values.size();++i) values[i]=values[i-1]+spacing[0];
    table=Buffer<float>(ctx,values.size());table.set(ctx,values.data(),values.size());
    unsigned blocks=static_cast<unsigned>((s[1]*s[2]+255)/256);
    if(narrow) cuda::edt_pass1_x<1,int><<<blocks,256,0,ctx.stream>>>(dist,source,s[0],s[1],s[2],table.data(),0,black_border);
    else cuda::edt_pass1_x<<<blocks,256,0,ctx.stream>>>(dist,source,s[0],s[1],s[2],table.data(),0,black_border);
    BROOK_CUDA(cudaGetLastError());
    // Keep the pageable source alive until its asynchronous upload completes.
    ctx.synchronize();
  }
  const int axis=xy?1:2;
  const long long pitch=s[axis]+3,lines=xy?s[0]*s[2]:s[0]*s[1];
  const long long batch=static_cast<long long>(std::max(size_t{1},std::min(size_t(lines),budget/(16*size_t(pitch)))));
  const size_t capacity=size_t(batch*pitch);
  Buffer<uint8_t> scratch(ctx,capacity*16);
  auto *v=reinterpret_cast<int*>(scratch.data());
  auto *h=reinterpret_cast<float*>(scratch.data()+capacity*4);
  auto *z=reinterpret_cast<double*>(scratch.data()+capacity*8);
  for(long long start=0;start<lines;start+=batch) {
    long long count=std::min(batch,lines-start);
    unsigned blocks=static_cast<unsigned>((count+255)/256);
    if(narrow && edt_index32_scratch_fits(pitch,count))
      cuda::edt_pass_fh<1,int><<<blocks,256,0,ctx.stream>>>(dist,source,v,h,z,s[0],s[1],s[2],axis,spacing[axis],nullptr,black_border,pitch,start,count);
    else cuda::edt_pass_fh<<<blocks,256,0,ctx.stream>>>(dist,source,v,h,z,s[0],s[1],s[2],axis,spacing[axis],nullptr,black_border,pitch,start,count);
    BROOK_CUDA(cudaGetLastError());
  }
  if(!xy) {
    finish_streamed<<<static_cast<unsigned>((field.size()+255)/256),256,0,ctx.stream>>>(lab,dist,static_cast<long long>(field.size()));
    BROOK_CUDA(cudaGetLastError());
  }
  ctx.synchronize();
}

void dispatch(Context &ctx,const Array &labels,Array &field,std::array<float,3> spacing,bool bb,bool xy,size_t budget) {
  switch(labels.dtype) {
    case BROOK_U8: return passes<uint8_t>(ctx,labels,field,spacing,bb,xy,budget);
    case BROOK_U16: return passes<uint16_t>(ctx,labels,field,spacing,bb,xy,budget);
    case BROOK_U32: return passes<uint32_t>(ctx,labels,field,spacing,bb,xy,budget);
    case BROOK_U64: return passes<uint64_t>(ctx,labels,field,spacing,bb,xy,budget);
    case BROOK_I8: return passes<int8_t>(ctx,labels,field,spacing,bb,xy,budget);
    case BROOK_I16: return passes<int16_t>(ctx,labels,field,spacing,bb,xy,budget);
    case BROOK_I32: return passes<int32_t>(ctx,labels,field,spacing,bb,xy,budget);
    case BROOK_I64: return passes<int64_t>(ctx,labels,field,spacing,bb,xy,budget);
    default: throw std::invalid_argument("streamed EDT requires integer labels");
  }
}

brook_volume region(const brook_volume &input,int axis,int64_t start,int64_t end) {
  auto v=input;v.shape[axis]=end-start;
  v.data=static_cast<const char*>(input.data)+start*input.strides[axis];return v;
}
Array upload_region(Context &ctx,const brook_volume &v) {
  const char *mode=std::getenv("BROOK_STREAM_TRANSFERS");
  if(mode && std::string(mode)=="0") return upload(ctx,v);
  const size_t item=dtype_size(v.dtype);
  if(v.strides[0]==int64_t(item) && v.strides[1]==v.shape[0]*int64_t(item) &&
     v.strides[2]>=v.shape[1]*v.strides[1]) {
    // A Y tile of an F-order host volume consists of one contiguous range per Z.
    auto out=allocate(ctx,{v.shape[0],v.shape[1],v.shape[2]},v.dtype);
    const size_t width=size_t(v.shape[0])*size_t(v.shape[1])*item;
    BROOK_CUDA(cudaMemcpy2DAsync(out.data(),width,v.data,size_t(v.strides[2]),width,
      size_t(v.shape[2]),cudaMemcpyHostToDevice,ctx.stream));
    return out;
  }
  if(v.strides[2]==int64_t(item) && v.strides[1]>=v.shape[2]*int64_t(item) &&
     v.strides[0]>=v.shape[1]*v.strides[1]) {
    // Gather bounded C-order ranges without a host transpose. upload() then
    // transfers these bytes unchanged and converts their layout on the device.
    const size_t row=size_t(v.shape[2])*item,plane=size_t(v.shape[1])*row;
    std::vector<uint8_t> packed(size_t(v.shape[0])*plane);
    const auto *src=static_cast<const uint8_t*>(v.data);
    for(int64_t x=0;x<v.shape[0];++x) {
      if(v.strides[1]==int64_t(row)) std::memcpy(packed.data()+size_t(x)*plane,src+x*v.strides[0],plane);
      else for(int64_t y=0;y<v.shape[1];++y)
        std::memcpy(packed.data()+size_t(x)*plane+size_t(y)*row,src+x*v.strides[0]+y*v.strides[1],row);
    }
    auto contiguous=v;contiguous.data=packed.data();
    contiguous.strides[0]=int64_t(plane);contiguous.strides[1]=int64_t(row);contiguous.strides[2]=int64_t(item);
    auto out=upload(ctx,contiguous);ctx.synchronize();return out;
  }
  return upload(ctx,v);
}
size_t stream_budget(size_t requested) {
  if(requested) return requested;
  if(const char *value=std::getenv("BROOK_STREAM_BUDGET_MB")) {
    // Set, even to an empty string: a positive integer number of MiB (the rule of brook._intake).
    const std::string text(value);
    const std::invalid_argument invalid("invalid BROOK_STREAM_BUDGET_MB='"+text+"': expected a positive integer number of MiB");
    if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos) throw invalid;
    unsigned long long mb=0;
    try {mb=std::stoull(text);} catch(const std::out_of_range &) {throw invalid;}
    if(!mb || mb>(SIZE_MAX>>20)) throw invalid;
    return size_t(mb)<<20;
  }
  size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
  return std::max(free/4,size_t{64}<<20);
}
}

Array upload_stream_region(Context &ctx,const brook_volume &v) {return upload_region(ctx,v);}
size_t default_stream_budget(size_t requested) {return stream_budget(requested);}

void edt_streamed(Context &ctx,const brook_volume &input,std::array<float,3> spacing,
                  bool black_border,size_t budget,float *output,size_t capacity) {
  ctx.activate();
  if(input.struct_size<sizeof(input) || input.abi_version!=BROOK_ABI_VERSION || input.memory!=BROOK_HOST)
    throw std::invalid_argument("streamed EDT requires a compatible host volume");
  const std::array<int64_t,3> shape={input.shape[0],input.shape[1],input.shape[2]};
  const size_t n=volume_size(shape),item=dtype_size(input.dtype);
  if(input.dtype==BROOK_F32 || input.dtype==BROOK_F64) throw std::invalid_argument("streamed EDT requires integer labels");
  if(n>SIZE_MAX/item || n>SIZE_MAX/sizeof(float) || capacity<n || (n && (!input.data || !output)))
    throw std::invalid_argument("streamed EDT output buffer is too small or null");
  for(float a:spacing) if(!std::isfinite(a) || a<=0) throw std::invalid_argument("anisotropy must be finite and positive");
  for(auto d:shape) if(d>INT32_MAX-3) throw std::invalid_argument("EDT axis exceeds int32 envelope capacity");
  if(!n) return;
  budget=stream_budget(budget);
  const size_t plane=size_t(shape[0])*size_t(shape[1]),row=size_t(shape[0])*size_t(shape[2]);
  // Keep the in-core scheduling policy; scratch is independently bounded.
  const int64_t dz=int64_t(std::max(size_t{1},std::min(size_t(shape[2]),budget/plane/(item+32))));
  const int64_t dy=int64_t(std::max(size_t{1},std::min(size_t(shape[1]),budget/row/(item+32))));
  const size_t scratch=std::max(size_t{1},budget/4);
  for(int64_t first=0;first<shape[2];first+=dz) {
    auto view=region(input,2,first,std::min(shape[2],first+dz));
    auto labels=upload_region(ctx,view);auto field=allocate(ctx,labels.shape,BROOK_F32);
    dispatch(ctx,labels,field,spacing,black_border,true,scratch);
    field.copy_to_host(output+size_t(first)*plane,field.bytes());
  }
  for(int64_t first=0;first<shape[1];first+=dy) {
    auto view=region(input,1,first,std::min(shape[1],first+dy));
    auto labels=upload_region(ctx,view);auto field=allocate(ctx,labels.shape,BROOK_F32);
    const size_t width=size_t(shape[0])*size_t(view.shape[1])*sizeof(float);
    BROOK_CUDA(cudaMemcpy2DAsync(field.data(),width,output+size_t(first)*size_t(shape[0]),plane*sizeof(float),
      width,size_t(shape[2]),cudaMemcpyHostToDevice,ctx.stream));
    dispatch(ctx,labels,field,spacing,black_border,false,scratch);
    BROOK_CUDA(cudaMemcpy2DAsync(output+size_t(first)*size_t(shape[0]),plane*sizeof(float),field.data(),width,
      width,size_t(shape[2]),cudaMemcpyDeviceToHost,ctx.stream));
    ctx.synchronize();
  }
}
}
