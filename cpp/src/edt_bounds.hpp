// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include <cstddef>
#include <cstdint>

namespace brook {
// Positive axes and the existing axis <= INT32_MAX-3 check are preconditions.
// Reserving 255 also bounds every inactive thread in a 256-thread launch.
// For ordinary EDT voxel offsets are below volume and prefix products <= volume;
// wall positions and loop increments are <= axis, and acc_pitch is zero.
inline constexpr bool edt_index32_volume_fits(size_t volume) {
  return volume>0 && volume<=size_t(INT32_MAX)-255;
}
// Active thread scratch offsets are tid*pitch, below count*pitch. Division
// avoids overflowing the host check itself; each pitch includes both walls.
inline constexpr bool edt_index32_scratch_fits(int64_t pitch,int64_t count) {
  return pitch>0 && count>0 && pitch<=INT32_MAX && count<=INT32_MAX/pitch;
}
}
