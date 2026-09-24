// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "cross_section.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <cmath>
namespace py=pybind11;
void bind_cross_sections(py::module_ &m) {
  m.def("cross_sectional_area",[](py::array labels,py::list inputs,py::object anisotropy,
      int window,int step,bool fill_holes,bool multipass,bool repair_contacts,bool visualize,bool in_place,std::array<int64_t,3> origin,
      bool shape_is_crop,bool return_processed) {
    if(labels.ndim()!=3) throw py::value_error("labels must have three dimensions");
    brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
    volume.data=labels.data();volume.memory=BROOK_HOST;
    auto kind=labels.dtype().kind();int bytes=int(labels.itemsize());
    if(kind=='b') volume.dtype=BROOK_U8;
    else if(kind=='u'&&(bytes==1||bytes==2||bytes==4||bytes==8)) volume.dtype=brook_dtype(bytes==1?BROOK_U8:bytes==2?BROOK_U16:bytes==4?BROOK_U32:BROOK_U64);
    else if(kind=='i'&&(bytes==1||bytes==2||bytes==4||bytes==8)) volume.dtype=brook_dtype(bytes==1?BROOK_I8:bytes==2?BROOK_I16:bytes==4?BROOK_I32:BROOK_I64);
    else throw py::value_error("integer labels required");
    if(!py::cast<bool>(labels.dtype().attr("isnative"))) throw py::value_error("native byte order required");
    if(in_place&&kind!='b'&&!labels.writeable()&&(labels.flags()&(py::array::c_style|py::array::f_style))) throw py::value_error("in_place labels must be writable");
    for(int a=0;a<3;++a) {volume.shape[a]=labels.shape(a);volume.strides[a]=labels.strides(a);}
    std::vector<brook::CrossSectionInput> native_inputs;
    for(auto object:inputs) {
      auto d=py::cast<py::dict>(object);brook::CrossSectionInput in;
      auto vertices=py::array_t<float,py::array::c_style|py::array::forcecast>::ensure(d["vertices"]);
      auto edges=py::array_t<uint32_t,py::array::c_style|py::array::forcecast>::ensure(d["edges"]);
      if(!vertices||!edges||vertices.ndim()!=2||vertices.shape(1)!=3||edges.ndim()!=2||edges.shape(1)!=2) throw py::value_error("vertices/edges have wrong shape");
      auto original=py::array::ensure(d["vertices"]);
      in.float64_vertices=original.dtype().kind()!='f'||original.itemsize()>4;
      if(in.float64_vertices) {
        auto precise=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(d["vertices"]);
        in.precise_vertices.resize(precise.shape(0));std::memcpy(in.precise_vertices.data(),precise.data(),precise.nbytes());
      }
      auto label=d["id"];
      if(PyIndex_Check(label.ptr())) {
        auto integer=py::reinterpret_steal<py::object>(PyNumber_Index(label.ptr()));
        if(!integer) throw py::error_already_set();
        int overflow=0;auto value=PyLong_AsLongLongAndOverflow(integer.ptr(),&overflow);
        if(PyErr_Occurred()) throw py::error_already_set();
        if(!overflow) in.skeleton.label=value;
        else if(overflow>0) {
          auto value=PyLong_AsUnsignedLongLong(integer.ptr());
          if(PyErr_Occurred()) PyErr_Clear();else in.unsigned_label=value;
        }
      } else if(PyFloat_Check(label.ptr())||(py::hasattr(label,"dtype")&&py::cast<std::string>(label.attr("dtype").attr("kind"))=="f")) {
        double value=py::cast<double>(label);
        if(std::isfinite(value)&&std::trunc(value)==value) {
          if(value>=double(INT64_MIN)&&value<double(INT64_MAX)) in.skeleton.label=int64_t(value);
          else if(value>=0&&value<std::ldexp(1.0,64)) in.unsigned_label=uint64_t(value);
        }
      }
      in.skeleton.vertices.resize(vertices.shape(0));in.skeleton.edges.resize(edges.shape(0));
      std::memcpy(in.skeleton.vertices.data(),vertices.data(),vertices.nbytes());std::memcpy(in.skeleton.edges.data(),edges.data(),edges.nbytes());
      if(d.contains("physical")) in.physical=py::cast<bool>(d["physical"]);
      if(d.contains("areas")&&!d["areas"].is_none()) {
        auto a=py::array_t<float,py::array::c_style|py::array::forcecast>::ensure(d["areas"]);
        if(!a||a.ndim()!=1||a.size()!=vertices.shape(0)) throw py::value_error("areas have wrong shape");in.areas.assign(a.data(),a.data()+a.size());
      }
      if(d.contains("contacts")&&!d["contacts"].is_none()) {
        auto a=py::array_t<uint8_t,py::array::c_style|py::array::forcecast>::ensure(d["contacts"]);
        if(!a||a.ndim()!=1||a.size()!=vertices.shape(0)) throw py::value_error("contacts have wrong shape");in.contacts.assign(a.data(),a.data()+a.size());
      }
      native_inputs.push_back(std::move(in));
    }
    brook::CrossSectionOptions options;options.smoothing_window=window;options.step=step;options.origin=origin;
    if(!anisotropy.is_none()) {
      auto original=py::array::ensure(anisotropy);
      auto values=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(anisotropy);
      if(!original||!values||values.ndim()!=1||values.size()!=3) throw py::value_error("anisotropy must have three entries");
      options.float64_spacing=original.dtype().kind()!='f'||original.itemsize()>4;
      for(int a=0;a<3;++a) {options.anisotropy[a]=float(values.data()[a]);options.position_spacing[a]=values.data()[a];}
    }
    options.fill_holes=fill_holes;options.multipass=multipass;options.repair_contacts=repair_contacts;options.visualize=visualize;options.in_place=in_place&&kind!='b';
    options.shape_is_crop=shape_is_crop;
    std::vector<brook::CrossSectionResult> result;
    {py::gil_scoped_release release;result=brook::cross_sectional_area_host(volume,native_inputs,options);}
    py::list output;
    for(auto &r:result) {
      py::array_t<float> areas(r.areas.size());py::array_t<uint8_t> contacts(r.contacts.size());
      std::memcpy(areas.mutable_data(),r.areas.data(),r.areas.size()*sizeof(float));std::memcpy(contacts.mutable_data(),r.contacts.data(),r.contacts.size());
      if(visualize) {
        auto s=r.section_shape;
        py::array_t<uint32_t> sections({s[0],s[1],s[2]},{int64_t(4),4*s[0],4*s[0]*s[1]});
        std::memcpy(sections.mutable_data(),r.sections.data(),r.sections.size()*sizeof(uint32_t));
        if(return_processed) output.append(py::make_tuple(areas,contacts,sections,r.section_origin,r.processed));
        else output.append(py::make_tuple(areas,contacts,sections,r.section_origin));
      } else if(return_processed) output.append(py::make_tuple(areas,contacts,r.processed));
      else output.append(py::make_tuple(areas,contacts));
    }
    return output;
  },py::arg("labels"),py::arg("skeletons"),py::arg("anisotropy")=py::none(),
    py::arg("smoothing_window")=1,py::arg("step")=1,py::arg("fill_holes")=false,py::arg("multipass")=false,py::arg("repair_contacts")=false,
    py::arg("visualize_section_planes")=false,py::arg("in_place")=false,py::arg("origin")=std::array<int64_t,3>{0,0,0},
    py::arg("_shape_is_crop")=false,py::arg("_return_processed")=false);
}
