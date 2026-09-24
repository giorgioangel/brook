// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include "device_check.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace brook {
bool upload_c_contiguous(Context &ctx,const brook_volume &view,Array &output);
Context::Context(int ordinal) : device(ordinal) {
  if(ordinal<0) throw std::invalid_argument("negative CUDA device ordinal");
  require_supported_device(ordinal);   // once per device: driver, compute capability, GPU code
  DeviceGuard guard(device);
  BROOK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}
Context::~Context() {
  if (stream) { DestructorDeviceGuard guard(device);border_plan.reset();cudaStreamDestroy(stream); }
}
Allocation::Allocation(size_t count, int ordinal, bool use_managed) : bytes(count), device(ordinal), managed(use_managed) {
  DeviceGuard guard(device);
  if(bytes) {
    if(managed) BROOK_CUDA(cudaMallocManaged(&data,bytes));
    else BROOK_CUDA(cudaMalloc(&data,bytes));
  }
}
Allocation::Allocation(void *external, size_t count, int ordinal, std::function<void()> on_release)
    : data(external), bytes(count), device(ordinal), release(std::move(on_release)) {}
Allocation::~Allocation() {
  if (release) { release(); return; }
  if (data) { DestructorDeviceGuard guard(device);cudaFree(data); }
}
size_t dtype_size(brook_dtype dtype) {
  switch (dtype) {
    case BROOK_U8: case BROOK_I8: return 1;
    case BROOK_U16: case BROOK_I16: return 2;
    case BROOK_U32: case BROOK_I32: case BROOK_F32: return 4;
    case BROOK_U64: case BROOK_I64: case BROOK_F64: return 8;
    default: throw std::invalid_argument("unsupported dtype");
  }
}
size_t volume_size(const std::array<int64_t, 3> &shape) {
  size_t n = 1;
  for (auto s : shape) {
    if (s < 0) throw std::invalid_argument("negative volume dimension");
    if (s && n > static_cast<size_t>(INT64_MAX) / static_cast<size_t>(s))
      throw std::invalid_argument("volume dimensions overflow int64");
    n *= static_cast<size_t>(s);
  }
  return n;
}
size_t Array::size() const { return volume_size(shape); }
size_t Array::bytes() const { return size() * dtype_size(dtype); }
std::vector<int64_t> Array::tensor_shape() const {
  return logical_shape?*logical_shape:std::vector<int64_t>(shape.begin(),shape.end());
}
std::vector<int64_t> Array::tensor_strides() const {
  if(logical_shape) return logical_strides;
  auto v=view();return {v.strides[0],v.strides[1],v.strides[2]};
}
Array tensor_view(Array array,std::vector<int64_t> shape,std::vector<int64_t> strides) {
  if(shape.size()!=strides.size() || shape.size()>3) throw std::invalid_argument("invalid tensor dimensions");
  size_t count=1;
  for(auto extent:shape) {
    if(extent<0 || (extent && count>size_t(INT64_MAX)/size_t(extent))) throw std::invalid_argument("invalid tensor shape");
    count*=size_t(extent);
  }
  if(count!=array.size()) throw std::invalid_argument("tensor view changes the element count");
  // Views currently describe contiguous storage, with arbitrary axis order.
  std::vector<size_t> axes(shape.size());
  for(size_t i=0;i<axes.size();++i) axes[i]=i;
  std::sort(axes.begin(),axes.end(),[&](auto a,auto b){return strides[a]<strides[b];});
  int64_t pitch=int64_t(dtype_size(array.dtype));
  if(count) for(auto i:axes) if(shape[i]>1) {
    if(strides[i]!=pitch) throw std::invalid_argument("tensor view must be contiguous");
    if(pitch>INT64_MAX/shape[i]) throw std::invalid_argument("tensor stride overflow");
    pitch*=shape[i];
  }
  array.logical_shape=std::move(shape);array.logical_strides=std::move(strides);return array;
}
brook_volume Array::view() const {
  brook_volume v{};
  v.struct_size = sizeof(v); v.abi_version = BROOK_ABI_VERSION;
  v.data = data(); v.dtype = dtype; v.memory = BROOK_DEVICE;
  if(logical_shape) {
    for(size_t i=0;i<3;++i) {
      v.shape[i]=i<logical_shape->size()?(*logical_shape)[i]:1;
      v.strides[i]=i<logical_strides.size()?logical_strides[i]:int64_t(bytes());
    }
    return v;
  }
  int64_t stride = static_cast<int64_t>(dtype_size(dtype));
  for (int i = 0; i < 3; ++i) { v.shape[i] = shape[i]; v.strides[i] = stride; stride *= shape[i]; }
  return v;
}
void Array::copy_to_host(void *out, size_t capacity) const {
  if (capacity < bytes() || (!out && bytes())) throw std::invalid_argument("output buffer is too small");
  DeviceGuard guard(device);
  if (bytes()) BROOK_CUDA(cudaMemcpy(out, data(), bytes(), cudaMemcpyDeviceToHost));
}
Array allocate(Context &ctx, std::array<int64_t, 3> shape, brook_dtype dtype) {
  const size_t n = volume_size(shape), item = dtype_size(dtype);
  if (n > std::numeric_limits<size_t>::max() / item) throw std::invalid_argument("array byte size overflow");
  return {std::make_shared<Allocation>(n * item, ctx.device), shape, dtype, ctx.device};
}
Array upload(Context &ctx, const brook_volume &v) {
  if (v.struct_size < sizeof(v) || v.abi_version != BROOK_ABI_VERSION)
    throw std::invalid_argument("incompatible Brook volume descriptor");
  auto out = allocate(ctx, {v.shape[0], v.shape[1], v.shape[2]}, v.dtype);
  if (!out.size()) return out;
  if (!v.data) throw std::invalid_argument("null volume data");
  size_t item = dtype_size(v.dtype);
  bool contiguous = true;
  int64_t stride = static_cast<int64_t>(item);
  for (int i = 0; i < 3; ++i) {
    if (v.shape[i] > 1 && stride != v.strides[i]) contiguous = false;
    stride *= v.shape[i];
  }
  if (v.memory == BROOK_DEVICE) {
    cudaPointerAttributes attributes{};
    BROOK_CUDA(cudaPointerGetAttributes(&attributes, v.data));
    if (attributes.device != ctx.device) throw std::invalid_argument("input is on a different CUDA device");
    if (contiguous) {
      BROOK_CUDA(cudaMemcpyAsync(out.data(), v.data, out.bytes(), cudaMemcpyDeviceToDevice, ctx.stream));
    } else {
      bool c_contiguous=true;int64_t c_stride=static_cast<int64_t>(item);
      for(int a=2;a>=0;--a) {if(v.shape[a]>1 && v.strides[a]!=c_stride)c_contiguous=false;c_stride*=v.shape[a];}
      if(!c_contiguous) throw std::invalid_argument("device volume must be C or Fortran contiguous");
      upload_c_contiguous(ctx,v,out);        // device source: transposed on the GPU, never declined
    }
  } else if (v.memory == BROOK_HOST) {
    if (contiguous) {
      BROOK_CUDA(cudaMemcpyAsync(out.data(), v.data, out.bytes(), cudaMemcpyHostToDevice, ctx.stream));
    } else {
      bool c_contiguous=true;int64_t c_stride=static_cast<int64_t>(item);
      for(int a=2;a>=0;--a) {if(v.shape[a]>1 && v.strides[a]!=c_stride)c_contiguous=false;c_stride*=v.shape[a];}
      if(c_contiguous && upload_c_contiguous(ctx,v,out)) return out;
      std::vector<unsigned char> packed(out.bytes());
      auto src = static_cast<const unsigned char *>(v.data);
      size_t pos = 0;
      for (int64_t z = 0; z < v.shape[2]; ++z)
        for (int64_t y = 0; y < v.shape[1]; ++y)
          for (int64_t x = 0; x < v.shape[0]; ++x, pos += item)
            std::memcpy(packed.data() + pos, src + x*v.strides[0] + y*v.strides[1] + z*v.strides[2], item);
      BROOK_CUDA(cudaMemcpyAsync(out.data(), packed.data(), out.bytes(), cudaMemcpyHostToDevice, ctx.stream));
      ctx.synchronize();
    }
  } else throw std::invalid_argument("unsupported memory location");
  return out;
}
}
