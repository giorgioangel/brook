// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "kernel_module.hpp"

namespace brook {
#define DECLARE_MODULE(name) \
  KernelModule name##_u8(); KernelModule name##_u16(); KernelModule name##_u32(); KernelModule name##_u64(); \
  KernelModule name##_module(brook_dtype dtype) { \
    switch(dtype) { \
      case BROOK_U8:return name##_u8();case BROOK_U16:return name##_u16(); \
      case BROOK_U32:return name##_u32();case BROOK_U64:return name##_u64(); \
      default:throw std::invalid_argument("invalid CUDA component label type"); \
    } \
  }
DECLARE_MODULE(batched)
DECLARE_MODULE(lockstep)
#undef DECLARE_MODULE
}
