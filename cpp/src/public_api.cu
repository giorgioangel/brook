// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "public_api.hpp"
#include <algorithm>
#include <cstring>
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_public_api
#include <cub/cub.cuh>
namespace cub=brook_cccl_public_api::cub;
namespace brook {
namespace {
template<class Input,class Output> __global__ void cast_values(const Input *input,Output *output,size_t n) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n) output[i]=static_cast<Output>(input[i]);
}
template<class T> uint64_t maximum(Context &ctx,const Array &input) {
  Buffer<T> out(ctx,1);size_t bytes=0;auto source=static_cast<const T*>(input.data());
  BROOK_CUDA(cub::DeviceReduce::Max(nullptr,bytes,source,out.data(),input.size(),ctx.stream));
  Buffer<uint8_t> scratch(ctx,bytes);BROOK_CUDA(cub::DeviceReduce::Max(scratch.data(),bytes,source,out.data(),input.size(),ctx.stream));
  return out.get(ctx,1)[0];
}
template<class T> uint64_t host_maximum(const brook_volume &input) {
  uint64_t result=0;auto data=static_cast<const char*>(input.data);
  for(int64_t z=0;z<input.shape[2];++z) for(int64_t y=0;y<input.shape[1];++y) for(int64_t x=0;x<input.shape[0];++x) {
    T value;std::memcpy(&value,data+x*input.strides[0]+y*input.strides[1]+z*input.strides[2],sizeof(T));result=std::max(result,uint64_t(value));
  }
  return result;
}
}
Array cast_public_volume(Context &ctx,const Array &input,brook_dtype dtype,bool copy) {
  if(input.device!=ctx.device || input.logical_shape) throw std::invalid_argument("native volume required on the context device");
  if(dtype!=BROOK_U8 && dtype!=BROOK_F32 && dtype!=BROOK_U32) throw std::invalid_argument("unsupported public volume conversion");
  if(input.dtype==dtype && !copy) return input;
  auto output=allocate(ctx,input.shape,dtype);if(!input.size()) return output;
  if(input.dtype==dtype) BROOK_CUDA(cudaMemcpyAsync(output.data(),input.data(),input.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
  else {
    auto launch=[&](auto value) {using T=decltype(value);unsigned blocks=unsigned((input.size()+255)/256);
      if(dtype==BROOK_U8) cast_values<<<blocks,256,0,ctx.stream>>>(static_cast<const T*>(input.data()),static_cast<uint8_t*>(output.data()),input.size());
      else if(dtype==BROOK_F32) cast_values<<<blocks,256,0,ctx.stream>>>(static_cast<const T*>(input.data()),static_cast<float*>(output.data()),input.size());
      else cast_values<<<blocks,256,0,ctx.stream>>>(static_cast<const T*>(input.data()),static_cast<uint32_t*>(output.data()),input.size());};
    switch(input.dtype) {
      case BROOK_U8:launch(uint8_t{});break;case BROOK_U16:launch(uint16_t{});break;case BROOK_U32:launch(uint32_t{});break;case BROOK_U64:launch(uint64_t{});break;
      case BROOK_I8:launch(int8_t{});break;case BROOK_I16:launch(int16_t{});break;case BROOK_I32:launch(int32_t{});break;case BROOK_I64:launch(int64_t{});break;
      case BROOK_F32:launch(float{});break;case BROOK_F64:launch(double{});break;
    }
    BROOK_CUDA(cudaGetLastError());
  }
  ctx.synchronize();return output;
}
uint64_t maximum_component_label(Context &ctx,const Array &input) {
  if(input.device!=ctx.device || input.logical_shape) throw std::invalid_argument("native component volume required");
  if(!input.size()) return 0;
  switch(input.dtype) {
    case BROOK_U8:return maximum<uint8_t>(ctx,input);case BROOK_U16:return maximum<uint16_t>(ctx,input);
    case BROOK_U32:return maximum<uint32_t>(ctx,input);case BROOK_U64:return maximum<uint64_t>(ctx,input);
    default:throw std::invalid_argument("component labels must be unsigned integers");
  }
}
uint64_t maximum_component_label_host(const brook_volume &input) {
  if(input.memory!=BROOK_HOST) throw std::invalid_argument("host component volume required");
  switch(input.dtype) {
    case BROOK_U8:return host_maximum<uint8_t>(input);case BROOK_U16:return host_maximum<uint16_t>(input);
    case BROOK_U32:return host_maximum<uint32_t>(input);case BROOK_U64:return host_maximum<uint64_t>(input);
    default:throw std::invalid_argument("component labels must be unsigned integers");
  }
}
}
