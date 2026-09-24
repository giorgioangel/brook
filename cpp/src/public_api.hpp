// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"
namespace brook {
Array cast_public_volume(Context &ctx,const Array &input,brook_dtype dtype,bool copy);
uint64_t maximum_component_label(Context &ctx,const Array &input);
uint64_t maximum_component_label_host(const brook_volume &input);
}
