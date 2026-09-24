// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "../common.cuh"
#include "../kernel_module.hpp"
#include <cooperative_groups.h>
namespace brook::cuda { namespace cg=cooperative_groups; }
namespace brook::cuda::batched_u64_regular {
using LABEL_T=uint64_t;
#include "batched_regular.inc"
}

namespace brook::cuda::batched_u64_coop {
using LABEL_T=uint64_t;
#include "batched_coop.inc"
}

namespace brook {
KernelModule batched_u64() {
  static const Kernel entries[]={
    {"seg_argmax",reinterpret_cast<const void*>(cuda::batched_u64_regular::seg_argmax),false},
    {"nbr_mask",reinterpret_cast<const void*>(cuda::batched_u64_regular::nbr_mask),false},
    {"freespace_seed_lab",reinterpret_cast<const void*>(cuda::batched_u64_regular::freespace_seed_lab),false},
    {"dbf_maxima_lab",reinterpret_cast<const void*>(cuda::batched_u64_regular::dbf_maxima_lab),false},
    {"pdrf_lab",reinterpret_cast<const void*>(cuda::batched_u64_regular::pdrf_lab),false},
    {"parents_lab",reinterpret_cast<const void*>(cuda::batched_u64_regular::parents_lab),false},
    {"backtrace_vol",reinterpret_cast<const void*>(cuda::batched_u64_regular::backtrace_vol),false},
    {"flood_lab",reinterpret_cast<const void*>(cuda::batched_u64_coop::flood_lab),true},
    {"flood_lab_nf",reinterpret_cast<const void*>(cuda::batched_u64_coop::flood_lab_nf),true}
  };
  return {entries,sizeof(entries)/sizeof(entries[0])};
}
}
