// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "cpu_post_bindings.hpp"
#include <pybind11/stl.h>
#include <cmath>
#include <limits>
#include <vector>
namespace py=pybind11;
namespace {
int32_t numpy_i32(double value) {
  if(!std::isfinite(value)||value<double(INT32_MIN)||value>=2147483648.0) return INT32_MIN;
  return int32_t(value);
}
std::vector<brook::CpuCrop> crop_boxes(py::object boxes,py::object resolution,py::object crop,size_t count) {
  std::vector<brook::CpuCrop> result;
  if(boxes.is_none()||py::cast<double>(crop)<=0) return result;
  auto b=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(boxes);
  auto r=py::array::ensure(resolution);
  if(!b||b.ndim()!=3||b.shape(0)!=py::ssize_t(count)||b.shape(1)!=2||b.shape(2)!=3||!r||r.size()!=3)
    throw py::value_error("expected crop boxes (N,2,3) and three resolution values");
  result.resize(count);
  if(r.dtype().kind()=='i'||r.dtype().kind()=='u') {
    if(!PyIndex_Check(crop.ptr())) throw py::type_error("floating crop cannot be added in-place to integer bounding boxes");
    auto rr=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(r);auto amount=py::cast<int64_t>(crop);
    auto integer_boxes=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(boxes);
    for(size_t i=0;i<count;++i) {
      bool zero=false,negative=false;
      for(int a=0;a<3;++a) {
        auto pad=uint64_t(amount)*uint64_t(rr.data()[a]);
        auto lo=int64_t(uint64_t(integer_boxes.data()[6*i+a])+pad),hi=int64_t(uint64_t(integer_boxes.data()[6*i+3+a])-pad);
        auto delta=int64_t(uint64_t(hi)-uint64_t(lo));zero|=delta==0;negative^=delta<0;
        // Osteoid.Bbox.create converts a CloudVolume bbox through its default
        // constructor: integer/float32 scalars choose int32, and axes reorder.
        result[i].bounds[a]=int32_t(uint32_t(std::min(lo,hi)));
        result[i].bounds[a+3]=int32_t(uint32_t(std::max(lo,hi)));
      }
      result[i].apply=!zero&&!negative;
    }
  } else {
    auto rr=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(r);double amount=py::cast<double>(crop);
    bool single=r.itemsize()<=4;
    for(size_t i=0;i<count;++i) {
      double volume=1;
      for(int a=0;a<3;++a) {
        double lo,hi,delta;
        if(single) {float pad=float(amount)*float(rr.data()[a]);float low=float(b.data()[6*i+a])+pad,high=float(b.data()[6*i+3+a])-pad;lo=low;hi=high;delta=float(high-low);}
        else {double pad=amount*rr.data()[a];lo=b.data()[6*i+a]+pad;hi=b.data()[6*i+3+a]-pad;delta=hi-lo;}
        result[i].bounds[a]=single?double(numpy_i32(std::min(lo,hi))):double(float(std::min(lo,hi)));
        result[i].bounds[a+3]=single?double(numpy_i32(std::max(lo,hi))):double(float(std::max(lo,hi)));
        volume*=delta;
      }
      result[i].apply=!(volume<=0);
    }
  }
  return result;
}
}
void bind_scheduling(py::module_ &m) {
  m.def("physical_origins",[](std::vector<std::array<int64_t,3>> starts,std::array<float,3> spacing) {
    std::vector<std::array<float,3>> result(starts.size());
    for(size_t i=0;i<starts.size();++i) for(int a=0;a<3;++a) result[i][a]=float(starts[i][a])*spacing[a];
    return result;
  },py::arg("starts"),py::arg("anisotropy"));
  m.def("chunk_shape_for",[](std::array<int64_t,3> shape,int itemsize,uint64_t free,bool fixing) {
    if(itemsize<=0) throw py::value_error("itemsize must be positive");
    // Bytes per voxel of an in-core chunk: labels, components, DBF, dense
    // alive/flags/key/(penalty,distance), and eight active int32 lists.
    int per_voxel=itemsize+4+4+1+4+8+(fixing?8:0)+4*8;
    double estimate=std::pow((0.85*double(free))/per_voxel,1.0/3.0);
    int64_t side=int64_t(estimate);side=std::max(int64_t(64),side-side%32);
    for(auto &s:shape) s=std::min(s,side);return shape;
  },py::arg("shape"),py::arg("itemsize"),py::arg("free_bytes"),py::arg("fix_branching")=true);
  m.def("cpu_fuse",[](py::iterable inputs,py::object boxes,py::object resolution,py::object crop) {
    std::vector<brook::CpuGeometry> geometries;std::vector<brook::CpuMetadata> metadata;
    for(auto item:inputs) {
      auto t=py::cast<py::tuple>(item);if(t.size()!=5) throw py::value_error("fusion inputs require vertices,edges,radii,transform,space");
      metadata.push_back(brook::python::cpu_metadata(py::reinterpret_borrow<py::object>(t[3]),py::cast<std::string>(t[4]),geometries.size()));
      geometries.push_back(brook::python::cpu_input(t[0],t[1],t[2]));
    }
    auto bounds=crop_boxes(boxes,resolution,crop,geometries.size());brook::CpuSkeleton result;std::vector<size_t> attributes;
    {py::gil_scoped_release release;result=brook::cpu_fuse(std::move(geometries),metadata,bounds,&attributes);}
    auto output=brook::python::cpu_output(result);output["attribute_inputs"]=attributes;return output;
  },py::arg("inputs"),py::arg("boxes")=py::none(),py::arg("resolution")=py::none(),py::arg("crop")=0);
  m.def("cable_length_info",[](py::handle v,py::handle e,py::handle r,py::object threshold,py::object transform,std::string space) {
    auto metadata=brook::python::cpu_metadata(transform,space,0);auto geometry=brook::python::cpu_input(v,e,r);
    bool wide=geometry.vertex64;double limit=threshold.is_none()?std::numeric_limits<double>::infinity():py::cast<double>(threshold);
    if(!wide&&(PyFloat_CheckExact(threshold.ptr())||PyLong_CheckExact(threshold.ptr()))) limit=float(limit);
    std::pair<brook::CpuSkeleton,double> result;
    {py::gil_scoped_release release;result=brook::cpu_physical_length(std::move(geometry),metadata);}
    auto out=brook::python::cpu_output(result.first);out["within"]=result.second<=limit;out["exceeds"]=result.second>limit;out["length"]=result.second;return out;
  },py::arg("vertices"),py::arg("edges"),py::arg("radii"),py::arg("threshold"),py::arg("transform")=py::none(),py::arg("space")="voxel");
}
