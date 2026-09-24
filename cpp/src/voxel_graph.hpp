// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"

namespace brook {
Array graph_for_trace(Context &ctx,const Array &graph);
Components graph_components(Context &ctx,const Array &labels,const Array &graph,bool capture=false);
// segment > 0: the z axis is a stack of independent samples of that depth (see edt); the z pass
// never crosses a sample boundary, and a black border closes every sample's own far face.
Array graph_edt(Context &ctx,const Array &labels,const Array &graph,std::array<float,3> anisotropy,
                bool black_border,bool capture=false,bool fused=false,bool compact=true,int64_t segment=0);
}
