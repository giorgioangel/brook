// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include "cooperative.cuh"
#include <string_view>

namespace brook {
struct Kernel {
  const char *name;
  const void *function;
  bool cooperative;
  template<class... Args> void launch(Context &ctx,unsigned blocks,unsigned threads,Args... values) const {
    void *args[]={static_cast<void*>(&values)...};
    if(cooperative) BROOK_CUDA(cudaLaunchCooperativeKernel(function,dim3(blocks),dim3(threads),args,0,ctx.stream));
    else BROOK_CUDA(cudaLaunchKernel(function,dim3(blocks),dim3(threads),args,0,ctx.stream));
  }
  int grid(Context &ctx,size_t n,const char *setting,int share=1) const {
    return std::max(1,cooperative_grid(ctx,function,n,setting)/share);
  }
};
struct KernelModule {
  const Kernel *entries;
  size_t count;
  const Kernel &get(std::string_view name) const {
    for(size_t i=0;i<count;++i) if(name==entries[i].name) return entries[i];
    throw std::invalid_argument("unknown CUDA kernel: "+std::string(name));
  }
};
KernelModule batched_module(brook_dtype dtype);
KernelModule lockstep_module(brook_dtype dtype);
}
