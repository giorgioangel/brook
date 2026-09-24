// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "oversegment_bindings.hpp"
#include "oversegment.hpp"
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <optional>
namespace py=pybind11;
namespace {
template<class T> py::array feature_array(const brook::Oversegmentation &r) {
 std::vector<py::ssize_t> shape(r.shape.begin(),r.shape.end()),strides{sizeof(T),py::ssize_t(sizeof(T)*r.shape[0]),py::ssize_t(sizeof(T)*r.shape[0]*r.shape[1])};
 py::array_t<T> out(shape,strides);auto p=out.mutable_data();for(size_t i=0;i<r.features.size();++i) p[i]=T(r.features[i]);return out;
}
}
py::tuple oversegment_python(brook::Context *ctx,py::array labels,py::object skeletons,py::object anisotropy,
                            bool progress,bool fill_holes,bool in_place,int downsample) {
 if(labels.ndim()!=3) throw py::value_error("oversegment requires a 3D volume");
 if(in_place && !labels.writeable()) throw py::value_error("in_place requires a writable label volume");
 auto np=py::module_::import("numpy");auto copied=py::module_::import("copy").attr("deepcopy")(skeletons);
 py::list list;
 if(py::hasattr(copied,"vertices")) list.append(copied);
 else if(py::isinstance<py::dict>(copied)) {for(auto item:copied.cast<py::dict>()) list.append(item.second);}
 else {for(auto item:copied) list.append(item);}
 brook::OversegmentOptions options;options.fill_holes=fill_holes;options.in_place=in_place;options.downsample=downsample;
 options.binary_labels=labels.dtype().equal(py::dtype::of<bool>());
 if(anisotropy.is_none()) anisotropy=np.attr("ones")(3,py::arg("dtype")="float32");
 py::array spacing=np.attr("asarray")(anisotropy);
 if(spacing.ndim()!=1 || spacing.size()!=3) throw py::value_error("anisotropy must have three values");
 auto doubles=py::array_t<double,py::array::c_style|py::array::forcecast>(spacing);
 for(int a=0;a<3;++a) options.spacing[a]=doubles.data()[a];
 brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;volume.memory=BROOK_HOST;volume.data=labels.data();
 static const char *names[]={"uint8","uint16","uint32","uint64","int8","int16","int32","int64","float32","float64"};
 bool valid=options.binary_labels;if(valid) volume.dtype=BROOK_U8;
 for(int i=0;i<10;++i) if(labels.dtype().equal(py::dtype(names[i]))) {volume.dtype=brook_dtype(i);valid=true;}
 if(!valid || !py::cast<bool>(labels.dtype().attr("isnative"))) throw py::type_error("unsupported label dtype");
 for(int a=0;a<3;++a) {volume.shape[a]=labels.shape(a);volume.strides[a]=labels.strides(a);}
 std::vector<brook::OversegmentSkeleton> converted_skeletons; converted_skeletons.reserve(list.size());
 for(auto item:list) {
  brook::OversegmentSkeleton skel;auto obj=py::reinterpret_borrow<py::object>(item);
  if(!options.binary_labels && !obj.attr("id").is_none()) skel.label=py::cast<int64_t>(obj.attr("id"));
  py::array input_vertices=py::array::ensure(obj.attr("vertices"));
  skel.float32_coordinates=py::cast<int>(np.attr("result_type")(input_vertices.dtype(),spacing).attr("itemsize"))<=4;
  auto vertices=py::array_t<double,py::array::c_style|py::array::forcecast>(input_vertices);
  auto edges=py::array_t<uint32_t,py::array::c_style|py::array::forcecast>(obj.attr("edges"));
  if(vertices.ndim()!=2 || vertices.shape(1)!=3 || edges.ndim()!=2 || edges.shape(1)!=2) throw py::value_error("invalid skeleton arrays");
  skel.vertices.resize(vertices.shape(0));skel.edges.resize(edges.shape(0));
  if(vertices.size()) std::memcpy(skel.vertices.data(),vertices.data(),vertices.nbytes());
  if(edges.size()) std::memcpy(skel.edges.data(),edges.data(),edges.nbytes()); converted_skeletons.push_back(std::move(skel));
 }
 brook::Oversegmentation result;
 {py::gil_scoped_release unlock;std::optional<brook::ContextCall> call;if(ctx) call.emplace(*ctx);result=brook::oversegment(ctx,volume,converted_skeletons,options);}
 uint64_t maximum=result.maximum;
 py::array features=maximum<=UINT8_MAX?feature_array<uint8_t>(result):maximum<=UINT16_MAX?feature_array<uint16_t>(result):maximum<=UINT32_MAX?feature_array<uint32_t>(result):feature_array<uint64_t>(result);
 size_t offset=0;
 for(size_t i=0;i<converted_skeletons.size();++i) {
  py::object obj=list[i];
  if(result.processed[i]) {
   bool present=false;for(auto prop:obj.attr("extra_attributes")) if(py::cast<std::string>(prop[py::str("id")])=="segments") present=true;
   if(!present) obj.attr("extra_attributes").attr("append")(py::dict(py::arg("id")="segments",py::arg("data_type")="uint64",py::arg("num_components")=1));
  }
  // The reference gathers from the refitted feature volume, so segments has
  // that dtype even though its metadata declares uint64.
  py::array segments(features.dtype(),std::vector<py::ssize_t>{py::ssize_t(converted_skeletons[i].vertices.size())});
  auto p=static_cast<char*>(segments.mutable_data());
  for(size_t j=0;j<converted_skeletons[i].vertices.size();++j) {uint64_t value=result.segments[offset++];std::memcpy(p+j*segments.itemsize(),&value,segments.itemsize());}
  obj.attr("segments")=segments;
 }
 return py::make_tuple(features,copied);
}
