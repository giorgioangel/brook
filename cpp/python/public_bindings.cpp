// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "public_api.hpp"
#include "streaming.hpp"
#include "packed.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <limits>
namespace py=pybind11;
namespace {
brook::SkeletonizeOptions options(py::dict values,std::array<double,3> spacing,bool branching) {
  brook::SkeletonizeOptions o;o.fix_branching=branching;o.component_api=true;o.trace_step_spacing=spacing;
  for(int a=0;a<3;++a) o.anisotropy[a]=float(spacing[a]);
  for(auto item:values) {
    auto key=py::cast<std::string>(item.first);
    if(key=="max_paths") {if(!item.second.is_none()) o.teasar.max_paths=py::cast<int64_t>(item.second);continue;}
    double v=py::cast<double>(item.second);
    if(key=="scale") o.teasar.scale=v;else if(key=="const") o.teasar.constant=v;
    else if(key=="pdrf_scale") o.teasar.pdrf_scale=v;else if(key=="pdrf_exponent") o.teasar.pdrf_exponent=v;
    else if(key=="soma_detection_threshold") o.teasar.soma_detection=v;else if(key=="soma_acceptance_threshold") o.teasar.soma_acceptance=v;
    else if(key=="soma_invalidation_scale") o.teasar.soma_scale=v;else if(key=="soma_invalidation_const") o.teasar.soma_constant=v;
    else throw py::type_error("unexpected TEASAR parameter: "+key);
  }
  return o;
}
brook_volume host_volume(py::array input) {
  if(input.ndim()!=3 || !py::cast<bool>(input.dtype().attr("isnative"))) throw py::value_error("a native-endian 3D component volume is required");
  brook_volume out{};out.struct_size=sizeof(out);out.abi_version=BROOK_ABI_VERSION;out.data=input.data();out.memory=BROOK_HOST;
  for(int a=0;a<3;++a) {out.shape[a]=input.shape(a);out.strides[a]=input.strides(a);}
  auto d=input.dtype();
  if(d.equal(py::dtype::of<uint8_t>())) out.dtype=BROOK_U8;else if(d.equal(py::dtype::of<uint16_t>())) out.dtype=BROOK_U16;
  else if(d.equal(py::dtype::of<uint32_t>())) out.dtype=BROOK_U32;else if(d.equal(py::dtype::of<uint64_t>())) out.dtype=BROOK_U64;
  else throw py::type_error("component labels must be unsigned integers");
  return out;
}
py::tuple metadata(const std::vector<brook::Box> &boxes,double dust) {
  py::array_t<int32_t> lo({py::ssize_t(boxes.size()),py::ssize_t(3)}),hi({py::ssize_t(boxes.size()),py::ssize_t(3)});
  py::array_t<int64_t> counts(boxes.size());std::vector<int64_t> ids;
  for(size_t i=0;i<boxes.size();++i) {
    for(int a=0;a<3;++a) {lo.mutable_data()[3*i+a]=int32_t(boxes[i].lo[a]);hi.mutable_data()[3*i+a]=int32_t(boxes[i].hi[a]);}
    counts.mutable_data()[i]=boxes[i].count;if(i && double(boxes[i].count)>dust) ids.push_back(i);
  }
  py::array_t<int64_t> selected(ids.size());std::copy(ids.begin(),ids.end(),selected.mutable_data());
  return py::make_tuple(selected,lo,hi,counts,boxes.size()-1);
}
std::optional<std::pair<brook::Array,int>> prior_check(py::object value) {
  if(value.is_none()) return std::nullopt;
  auto pair=py::cast<py::tuple>(value);if(pair.size()!=2) throw py::value_error("void_check must contain mask and count");
  brook::Array array;if(!pair[0].is_none()) array=py::cast<brook::Array>(pair[0]);
  return std::make_pair(std::move(array),py::cast<int>(pair[1]));
}
py::list points(const std::vector<brook::Point> &values) {py::list out;for(auto p:values) out.append(py::make_tuple(p[0],p[1],p[2]));return out;}
}
void bind_public_primitives(py::module_ &m) {
  py::register_exception_translator([](std::exception_ptr value) {
    try {if(value) std::rethrow_exception(value);}catch(const brook::TraceZeroDivision &error) {PyErr_SetString(PyExc_ZeroDivisionError,error.what());}
  });
  m.def("_cast_volume",[](brook::Context &ctx,const brook::Array &input,int dtype,bool copy) {
    py::gil_scoped_release unlock;brook::ContextCall call(ctx);return brook::cast_public_volume(ctx,input,static_cast<brook_dtype>(dtype),copy);
  },py::arg("context"),py::arg("input"),py::arg("dtype"),py::arg("copy")=false);
  m.def("_analyze_components",[](brook::Context &ctx,const brook::Array &input,double dust,std::optional<size_t> count) {
    std::vector<brook::Box> boxes;
    {py::gil_scoped_release unlock;brook::ContextCall call(ctx);auto maximum=brook::maximum_component_label(ctx,input);
      if(count && *count<maximum) throw std::invalid_argument("n_labels is smaller than a component label");
      if(count.value_or(maximum)>INT32_MAX) throw std::invalid_argument("component count exceeds int32 capacity");
      boxes=brook::analyze(ctx,input,count.value_or(maximum));}
    return metadata(boxes,dust);
  },py::arg("context"),py::arg("components"),py::arg("dust_threshold")=1000,py::arg("n_labels")=std::nullopt);
  m.def("_analyze_components_streamed",[](brook::Context &ctx,py::array input,double dust,std::optional<size_t> count,size_t budget) {
    auto view=host_volume(input);std::vector<brook::Box> boxes;
    {py::gil_scoped_release unlock;brook::ContextCall call(ctx);auto maximum=brook::maximum_component_label_host(view);
      if(count && *count<maximum) throw std::invalid_argument("n_labels is smaller than a component label");
      if(count.value_or(maximum)>INT32_MAX) throw std::invalid_argument("component count exceeds int32 capacity");
      boxes=brook::analyze_streamed(ctx,view,count.value_or(maximum),budget);}
    return metadata(boxes,dust);
  },py::arg("context"),py::arg("components"),py::arg("dust_threshold")=1000,py::arg("n_labels")=std::nullopt,py::arg("budget_bytes")=0);
  m.def("_component_ids",[](py::array_t<int64_t,py::array::c_style|py::array::forcecast> counts,double dust,size_t count) {
    if(counts.ndim()!=1 || size_t(counts.size())!=count+1) throw py::value_error("invalid component counts");
    std::vector<int64_t> selected;for(size_t i=1;i<=count;++i) if(double(counts.data()[i])>dust) selected.push_back(i);
    py::array_t<int64_t> out(selected.size());std::copy(selected.begin(),selected.end(),out.mutable_data());return out;
  });
  m.def("_prepare_public",[](brook::Context &ctx,brook::Array mask,brook::Array dbf,py::dict params,std::array<double,3> spacing,
       bool branching,std::vector<brook::Point> before,std::vector<brook::Point> after,std::optional<brook::Point> root,
       const brook::Array *graph,py::object prior_value) -> py::object {
    auto prior=prior_check(prior_value);auto o=options(params,spacing,branching);brook::ComponentPreparation p;std::optional<brook::Array> converted_graph;
    {py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      if(mask.shape!=dbf.shape || !mask.size()) throw std::invalid_argument("mask and DBF require matching nonempty shapes");
      if(graph && (graph->device!=ctx.device || graph->shape!=mask.shape)) throw std::invalid_argument("invalid graph volume");
      mask=brook::cast_public_volume(ctx,mask,BROOK_U8,false);dbf=brook::cast_public_volume(ctx,dbf,BROOK_F32,true);
      if(graph) {converted_graph=brook::cast_public_volume(ctx,*graph,BROOK_U32,false);graph=&*converted_graph;}
      if(prior && prior->first.storage) {if(prior->first.shape!=mask.shape) throw std::invalid_argument("invalid filled mask shape");prior->first=brook::cast_public_volume(ctx,prior->first,BROOK_U8,false);}
      p=brook::prepare_component(ctx,std::move(mask),std::move(dbf),o,std::move(before),std::move(after),root,graph,prior?&*prior:nullptr);}
    if(!p.mask.storage) return py::none();
    py::dict out;out["labels"]=p.mask;out["DBF"]=p.dbf;out["DAF"]=p.daf;out["shape"]=py::make_tuple(p.mask.shape[0],p.mask.shape[1],p.mask.shape[2]);
    out["root"]=py::make_tuple(p.root[0],p.root[1],p.root[2]);out["nfilled"]=p.filled;
    out["soma_mode"]=p.soma;out["soma_radius"]=p.soma_radius;
    out["parents"]=p.parents.storage?py::cast(brook::row_view(p.parents,0,p.parents.size(),1)):py::none();
    out["pdrf_field"]=p.penalty.storage?py::cast(p.penalty):py::none();out["rail_delta"]=py::none();
    out["manual_targets_before"]=points(p.before);out["manual_targets_after"]=points(p.after);
    return out;
  },py::arg("context"),py::arg("labels"),py::arg("DBF"),py::arg("params"),py::arg("spacing"),py::arg("fix_branching"),
    py::arg("before"),py::arg("after"),py::arg("root")=std::nullopt,py::arg("graph")=nullptr,py::arg("void_check")=py::none());
  m.def("_trace_public",[](brook::Context &ctx,brook::Array mask,brook::Array dbf,py::dict params,std::array<double,3> spacing,
       bool branching,std::vector<brook::Point> before,std::vector<brook::Point> after,std::optional<brook::Point> root,
       const brook::Array *graph,py::object prior_value) {
    auto prior=prior_check(prior_value);auto o=options(params,spacing,branching);brook::Skeleton result;brook::VoxelSkeleton voxels;bool assembled=false;std::optional<brook::Array> converted_graph;
    {py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      if(mask.shape!=dbf.shape || !mask.size()) throw std::invalid_argument("mask and DBF require matching nonempty shapes");
      if(graph && (graph->device!=ctx.device || graph->shape!=mask.shape)) throw std::invalid_argument("invalid graph volume");
      mask=brook::cast_public_volume(ctx,mask,BROOK_U8,false);dbf=brook::cast_public_volume(ctx,dbf,BROOK_F32,true);
      if(graph) {converted_graph=brook::cast_public_volume(ctx,*graph,BROOK_U32,false);graph=&*converted_graph;}
      if(prior && prior->first.storage) {if(prior->first.shape!=mask.shape) throw std::invalid_argument("invalid filled mask shape");prior->first=brook::cast_public_volume(ctx,prior->first,BROOK_U8,false);}
      result=brook::trace_component(ctx,std::move(mask),std::move(dbf),o,std::move(before),std::move(after),root,graph,prior?&*prior:nullptr,&assembled,&voxels);}
    py::object vertices=py::none(),edges=py::none();
    if(!voxels.vertices.empty()) {
      py::array_t<uint32_t> v({py::ssize_t(voxels.vertices.size()),py::ssize_t(3)});
      py::array_t<int64_t> e({py::ssize_t(voxels.edges.size()),py::ssize_t(2)});
      for(size_t i=0;i<voxels.vertices.size();++i) for(int a=0;a<3;++a) v.mutable_data()[3*i+a]=voxels.vertices[i][a];
      for(size_t i=0;i<voxels.edges.size();++i) for(int a=0;a<2;++a) e.mutable_data()[2*i+a]=voxels.edges[i][a];
      vertices=std::move(v);edges=std::move(e);
    }
    return py::make_tuple(std::move(result),assembled,vertices,edges);
  },py::arg("context"),py::arg("labels"),py::arg("DBF"),py::arg("params"),py::arg("spacing"),py::arg("fix_branching"),
    py::arg("before"),py::arg("after"),py::arg("root")=std::nullopt,py::arg("graph")=nullptr,py::arg("void_check")=py::none());
}
