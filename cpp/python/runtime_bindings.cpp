// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include "device_check.hpp"
#include "packed.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
namespace py=pybind11;
void bind_runtime(py::module_ &m) {
  m.def("device_count",[] {py::gil_scoped_release unlock;return brook::visible_device_count();});
  // The startup check every Context runs (driver, compute capability, GPU code for the device).
  m.def("check_device",[](int device) {py::gil_scoped_release unlock;brook::require_supported_device(device);},py::arg("device")=-1);
  m.def("build_info",[] {
    auto info=brook::build_info();
    auto words=[](const std::string &text) {
      py::list out;size_t start=0;
      while(start<text.size()) {
        size_t end=text.find(' ',start);if(end==std::string::npos) end=text.size();
        if(end>start) out.append(text.substr(start,end-start));
        start=end+1;
      }
      return out;
    };
    auto version=[](int value)->py::object { return value?py::object(py::str(brook::cuda_version_string(value))):py::object(py::none()); };
    return py::dict(py::arg("cuda_architectures")=info.architectures,py::arg("sass")=words(info.sass),py::arg("ptx")=words(info.ptx),
      py::arg("cuda_toolkit")=version(info.toolkit),py::arg("cuda_runtime")=version(info.runtime),py::arg("cuda_driver")=version(info.driver));
  });
  m.def("current_device",[] {
    int device=0;
    {
      py::gil_scoped_release unlock;
      if(cudaGetDevice(&device)!=cudaSuccess) {   // no usable device: report why, as the startup check does
        cudaGetLastError();brook::require_supported_device(-1);BROOK_CUDA(cudaGetDevice(&device));
      }
    }
    return device;
  });
  m.def("set_device",[](int device) {py::gil_scoped_release unlock;BROOK_CUDA(cudaSetDevice(device));});
  m.def("memory_info",[](int device) {
    size_t free=0,total=0;
    {py::gil_scoped_release unlock;if(device<0) BROOK_CUDA(cudaGetDevice(&device));brook::DeviceGuard guard(device);BROOK_CUDA(cudaMemGetInfo(&free,&total));}
    return py::make_tuple(free,total);
  },py::arg("device")=-1);
  m.def("device_properties",[](int device) {
    cudaDeviceProp properties{};
    {py::gil_scoped_release unlock;if(device<0) BROOK_CUDA(cudaGetDevice(&device));BROOK_CUDA(cudaGetDeviceProperties(&properties,device));}
    return py::dict(py::arg("id")=device,py::arg("name")=properties.name,py::arg("total_memory")=properties.totalGlobalMem,
      py::arg("major")=properties.major,py::arg("minor")=properties.minor,py::arg("multiprocessor_count")=properties.multiProcessorCount);
  },py::arg("device")=-1);
  m.def("trace_defaults",[] {
    brook::TraceParameters p;
    return py::dict(py::arg("scale")=p.scale,py::arg("const")=p.constant,py::arg("pdrf_scale")=p.pdrf_scale,
      py::arg("pdrf_exponent")=p.pdrf_exponent,py::arg("soma_detection_threshold")=p.soma_detection,
      py::arg("soma_acceptance_threshold")=p.soma_acceptance,py::arg("soma_invalidation_scale")=p.soma_scale,
      py::arg("soma_invalidation_const")=p.soma_constant,py::arg("max_paths")=py::none());
  });
  m.def("_pack_device",[](brook::Context &ctx,std::vector<int64_t> labels,std::vector<int64_t> voff,std::vector<int64_t> eoff,
      brook::Array vertices,brook::Array edges,brook::Array radii,std::array<float,3> anisotropy) {
    if(vertices.device!=ctx.device || edges.device!=ctx.device || radii.device!=ctx.device) throw py::value_error("packed buffers are on a different device");
    if(vertices.dtype!=BROOK_F32 || edges.dtype!=BROOK_U32 || radii.dtype!=BROOK_F32 ||
       vertices.tensor_shape()!=std::vector<int64_t>{int64_t(radii.size()),3} ||
       edges.tensor_shape()!=std::vector<int64_t>{int64_t(edges.size()/2),2} || radii.tensor_shape()!=std::vector<int64_t>{int64_t(radii.size())} ||
       vertices.tensor_strides()!=std::vector<int64_t>{12,4} || edges.tensor_strides()!=std::vector<int64_t>{8,4} || radii.tensor_strides()!=std::vector<int64_t>{4} ||
       voff.size()!=labels.size()+1 || eoff.size()!=labels.size()+1 || voff.front()!=0 || eoff.front()!=0 ||
       voff.back()!=int64_t(radii.size()) || eoff.back()!=int64_t(edges.size()/2) ||
       !std::is_sorted(voff.begin(),voff.end()) || !std::is_sorted(eoff.begin(),eoff.end())) throw py::value_error("invalid packed device buffers/offsets");
    return brook::PackedSkeletons{std::move(labels),std::move(voff),std::move(eoff),std::move(vertices),std::move(edges),std::move(radii),anisotropy};
  },py::arg("context"),py::arg("labels"),py::arg("v_off"),py::arg("e_off"),py::arg("vertices"),py::arg("edges"),py::arg("radii"),py::arg("anisotropy"));
}
