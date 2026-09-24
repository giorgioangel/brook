// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "../common.cuh"
#include "../kernel_module.hpp"
#include <cooperative_groups.h>
namespace brook::cuda { namespace cg=cooperative_groups; }
namespace brook::cuda::lockstep_u16_regular {
using LABEL_T=uint16_t;
#include "lockstep_regular.inc"
}

namespace brook::cuda::lockstep_u16_coop {
using LABEL_T=uint16_t;
#include "lockstep_coop.inc"
}

namespace brook {
KernelModule lockstep_u16() {
  static const Kernel entries[]={
    {"backtrace_dist_lab",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::backtrace_dist_lab),false},
    {"backtrace_parent_lab",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::backtrace_parent_lab),false},
    {"lock_offsets",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::lock_offsets),false},
    {"lock_rails",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::lock_rails),false},
    {"lock_advance",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::lock_advance),false},
    {"kd_first",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::kd_first),false},
    {"k2_setup",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_setup),false},
    {"k2_gather",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_gather),false},
    {"k2_lens",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_lens),false},
    {"k2_src_offsets",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_src_offsets),false},
    {"k2_src_copy",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_src_copy),false},
    {"k2_route",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_route),false},
    {"k2_reset",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_reset),false},
    {"k2_mark",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_mark),false},
    {"k2_contact",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_contact),false},
    {"kd_scan_offsets",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::kd_scan_offsets),false},
    {"kd_scan_init",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::kd_scan_init),false},
    {"k2_scan",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_scan),false},
    {"k2_accept",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_accept),false},
    {"kd_offsets",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::kd_offsets),false},
    {"k2_apply",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_apply),false},
    {"k2_write",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::k2_write),false},
    {"kd_advance",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::kd_advance),false},
    {"reset_touched",reinterpret_cast<const void*>(cuda::lockstep_u16_regular::reset_touched),false},
    {"dstep_lab",reinterpret_cast<const void*>(cuda::lockstep_u16_coop::dstep_lab),true},
    {"target_lab",reinterpret_cast<const void*>(cuda::lockstep_u16_coop::target_lab),true},
    {"draft_lab",reinterpret_cast<const void*>(cuda::lockstep_u16_coop::draft_lab),true},
    {"inval_lab",reinterpret_cast<const void*>(cuda::lockstep_u16_coop::inval_lab),true}
  };
  return {entries,sizeof(entries)/sizeof(entries[0])};
}
}
