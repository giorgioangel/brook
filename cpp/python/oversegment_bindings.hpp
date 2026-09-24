// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
pybind11::tuple oversegment_python(brook::Context*,pybind11::array,pybind11::object,pybind11::object,bool,bool,bool,int);
