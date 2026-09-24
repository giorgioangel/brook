// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "dlpack.hpp"
#include <algorithm>
#include <cstdlib>
#include <type_traits>

namespace brook {
namespace {
template<class Tensor> struct Export {
  Tensor tensor{};
  Array device_owner;
  std::unique_ptr<void,decltype(&std::free)> host_owner{nullptr,std::free};
  std::vector<int64_t> shape,strides;
};
template<class Tensor> void release(Tensor *tensor) noexcept {
  delete static_cast<Export<Tensor>*>(tensor->manager_ctx);
}
template<class Tensor> Tensor *make(const Array &array,DLDevice target,bool copy,uint32_t minor) {
  if(target.device_type!=kDLCPU && target.device_type!=kDLCUDA) throw std::invalid_argument("unsupported DLPack device type");
  if(target.device_id<0 || (target.device_type==kDLCPU && target.device_id!=0)) throw std::invalid_argument("invalid DLPack device ordinal");
  bool transfer=target.device_type!=kDLCUDA || target.device_id!=array.device;
  if(transfer && !copy) throw std::invalid_argument("requested DLPack device requires a copy");
  auto out=std::make_unique<Export<Tensor>>();
  out->shape=array.tensor_shape();out->strides=array.tensor_strides();
  size_t item=dtype_size(array.dtype);
  for(auto &stride:out->strides) {
    if(stride%int64_t(item)) throw std::invalid_argument("DLPack stride is not an element multiple");
    stride/=int64_t(item);
  }
  auto &tensor=out->tensor;
  tensor.manager_ctx=out.get();tensor.deleter=release<Tensor>;
  auto &view=tensor.dl_tensor;
  view.device=target;view.ndim=int(out->shape.size());
  view.shape=out->shape.empty()?nullptr:out->shape.data();view.strides=out->strides.empty()?nullptr:out->strides.data();
  view.dtype={uint8_t(array.dtype<=BROOK_U64?kDLUInt:(array.dtype<=BROOK_I64?kDLInt:kDLFloat)),uint8_t(item*8),1};
  if(target.device_type==kDLCPU) {
    size_t bytes=array.bytes();
    if(bytes>SIZE_MAX-255) throw std::invalid_argument("host allocation size overflow");
    if(bytes) {
      out->host_owner.reset(std::aligned_alloc(256,(bytes+255)&~size_t{255}));
      if(!out->host_owner) throw std::bad_alloc();
      array.copy_to_host(out->host_owner.get(),bytes);
    }
    view.data=out->host_owner.get();
  } else if(copy) {
    int count=0;BROOK_CUDA(cudaGetDeviceCount(&count));
    if(target.device_id>=count) throw std::invalid_argument("DLPack CUDA device is out of range");
    Context context(target.device_id);ContextCall call(context);
    out->device_owner=allocate(context,array.shape,array.dtype);
    if(array.bytes()) {
      if(target.device_id==array.device) BROOK_CUDA(cudaMemcpyAsync(out->device_owner.data(),array.data(),array.bytes(),cudaMemcpyDeviceToDevice,context.stream));
      else BROOK_CUDA(cudaMemcpyPeerAsync(out->device_owner.data(),target.device_id,array.data(),array.device,array.bytes(),context.stream));
    }
    context.synchronize();
    view.data=out->device_owner.data();
  } else {
    out->device_owner=array;
    view.data=array.size()?array.storage->data:nullptr;
    view.byte_offset=array.size()?array.byte_offset:0;
  }
  if constexpr(std::is_same_v<Tensor,DLManagedTensorVersioned>) {
    tensor.version={1,std::min(minor,uint32_t(DLPACK_MINOR_VERSION))};
    tensor.flags=copy?DLPACK_FLAG_BITMASK_IS_COPIED:0;
  }
  return &out.release()->tensor;
}
}
DLManagedTensor *export_dlpack(const Array &array,DLDevice target,bool copy) {
  return make<DLManagedTensor>(array,target,copy,0);
}
DLManagedTensorVersioned *export_dlpack_versioned(const Array &array,DLDevice target,bool copy,uint32_t minor) {
  return make<DLManagedTensorVersioned>(array,target,copy,minor);
}
}
