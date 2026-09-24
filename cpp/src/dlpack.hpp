// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include <brook/dlpack.h>

namespace brook {
DLManagedTensor *export_dlpack(const Array &array,DLDevice target,bool copy);
DLManagedTensorVersioned *export_dlpack_versioned(const Array &array,DLDevice target,bool copy,uint32_t minor);
}
