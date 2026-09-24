// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include <cuda_fp16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
namespace py=pybind11;
namespace {
struct Half {
  uint16_t bits=0;
  Half()=default;
  Half(double value):bits(static_cast<__half_raw>(__double2half(value)).x) {}
  operator double() const {return __half2float(__half(__half_raw{bits}));}
};
template<class T> constexpr bool floating=std::is_floating_point_v<T> || std::is_same_v<T,Half>;
[[noreturn]] void overflow() {PyErr_SetString(PyExc_OverflowError,"object label outside input dtype range");throw py::error_already_set();}
template<class T> std::vector<T> integer_ids(py::iterable ids) {
  std::vector<T> out;
  for(auto id:ids) {
    if(!PyNumber_Check(id.ptr())) throw py::type_error("object labels must be numbers");
    auto value=py::reinterpret_steal<py::object>(PyNumber_Long(id.ptr()));if(!value) throw py::error_already_set();
    if constexpr(std::is_unsigned_v<T>) {
      uint64_t v=PyLong_AsUnsignedLongLong(value.ptr());if(PyErr_Occurred()) throw py::error_already_set();
      if(v>std::numeric_limits<T>::max()) overflow();out.push_back(T(v));
    } else {
      int64_t v=PyLong_AsLongLong(value.ptr());if(PyErr_Occurred()) throw py::error_already_set();
      if(v<std::numeric_limits<T>::min() || v>std::numeric_limits<T>::max()) overflow();out.push_back(T(v));
    }
  }
  return out;
}
struct Query {int kind=0;uint64_t integer=0;double real=0;};
Query query(py::handle value,bool floating) {
  Query q;if(!PyNumber_Check(value.ptr())) return q;
  bool strong=py::hasattr(value,"dtype");
  if(floating || !PyIndex_Check(value.ptr())) {q.kind=strong?3:4;q.real=py::cast<double>(value);return q;}
  auto integer=py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));if(!integer) throw py::error_already_set();
  int overflow=0;auto v=PyLong_AsLongLongAndOverflow(integer.ptr(),&overflow);if(PyErr_Occurred()) throw py::error_already_set();
  if(!overflow) {q.kind=1;q.integer=uint64_t(v);}
  else if(overflow>0) {q.integer=PyLong_AsUnsignedLongLong(integer.ptr());if(PyErr_Occurred()) PyErr_Clear();else q.kind=2;}
  return q;
}
template<class T> bool matches(T v,const Query &q) {
  if(q.kind==3) return double(v)==q.real;
  if(q.kind==4) {
    if constexpr(floating<T>) return v==T(q.real);
    else return double(v)==q.real;
  }
  if(q.kind==1) {
    if constexpr(std::is_signed_v<T>) return int64_t(v)==int64_t(q.integer);
    else return int64_t(q.integer)>=0 && uint64_t(v)==q.integer;
  }
  if(q.kind==2) return v>=0 && uint64_t(v)==q.integer;
  return false;
}
template<class T> struct Selection {
  std::vector<Query> queries;std::vector<T> allowed;
  bool filtered=false,multiple=false;
  explicit Selection(py::object ids):filtered(!ids.is_none()) {
    if(!filtered) return;
    py::list items(ids);multiple=items.size()!=1;
    if(multiple) {
      if constexpr(floating<T>) throw py::type_error("No matching signature found for floating labels and multiple object_ids");
      else {allowed=integer_ids<T>(items);std::sort(allowed.begin(),allowed.end());}
    } else for(auto id:items) queries.push_back(query(id,floating<T>));
  }
  bool operator()(T value) const {
    return !filtered || (multiple?std::binary_search(allowed.begin(),allowed.end(),value):matches(value,queries.front()));
  }
};
template<class T> bool nonzero(py::array image,py::object ids) {
  Selection<T> keep(ids);const char *data=static_cast<const char*>(image.data());
  std::array<py::ssize_t,3> shape={image.shape(0),image.shape(1),image.shape(2)},strides={image.strides(0),image.strides(1),image.strides(2)};
  bool fortran=image.flags()&py::array::f_style;
  py::gil_scoped_release unlock;
  auto found=[&](py::ssize_t x,py::ssize_t y,py::ssize_t z) {T value;std::memcpy(&value,data+x*strides[0]+y*strides[1]+z*strides[2],sizeof(T));return value!=T(0) && keep(value);};
  if(fortran) {for(py::ssize_t z=0;z<shape[2];++z) for(py::ssize_t y=0;y<shape[1];++y) for(py::ssize_t x=0;x<shape[0];++x) if(found(x,y,z)) return true;}
  else {for(py::ssize_t x=0;x<shape[0];++x) for(py::ssize_t y=0;y<shape[1];++y) for(py::ssize_t z=0;z<shape[2];++z) if(found(x,y,z)) return true;}
  return false;
}
template<class T> py::tuple process(py::array image,py::object ids,bool in_place,bool convert) {
  Selection<T> keep(ids);
  bool mutate=in_place && keep.filtered && keep.multiple;
  if(mutate && !image.writeable()) throw py::value_error("in_place requires writable labels");
  // fastremap makes a contiguous temporary for noncontiguous input even when
  // in_place is requested; public intake additionally requires own-data.
  mutate=mutate && (image.flags()&(py::array::c_style|py::array::f_style));
  std::vector<py::ssize_t> shape(image.shape(),image.shape()+3),strides(3);
  bool fortran=image.flags()&py::array::f_style;
  py::ssize_t step=convert?8:sizeof(T);
  for(int j=0;j<3;++j) {int a=fortran?j:2-j;strides[a]=step;step*=shape[a];}
  py::array result=(!convert && mutate)?image:py::array(convert?py::dtype::of<int64_t>():image.dtype(),shape,strides);
  const char *input=static_cast<const char*>(image.data());char *output=static_cast<char*>(result.mutable_data());
  std::array<py::ssize_t,3> input_strides={image.strides(0),image.strides(1),image.strides(2)};
  bool uniform=true,first=true,integer_uniform=true,integer_first=true;T initial=0;int64_t initial_integer=0;
  {
    py::gil_scoped_release unlock;
    auto visit=[&](py::ssize_t x,py::ssize_t y,py::ssize_t z) {
      auto offset=x*input_strides[0]+y*input_strides[1]+z*input_strides[2];T value;std::memcpy(&value,input+offset,sizeof(T));
      if(!keep(value)) value=0;
      if(first) {initial=value;first=false;}if(value!=initial) uniform=false;
      if(mutate) std::memcpy(const_cast<char*>(input)+offset,&value,sizeof(T));
      auto target=output+x*strides[0]+y*strides[1]+z*strides[2];
      if(convert) {
        int64_t integer;
        if constexpr(floating<T>) {
          // CUDA floating-to-int64 conversion clamps positive overflow to
          // INT64_MAX; NumPy's host astype instead yields INT64_MIN there.
          double real=double(value);
          integer=(std::isnan(real) || real<double(INT64_MIN))?INT64_MIN:
                  real>=-double(INT64_MIN)?INT64_MAX:int64_t(real);
        } else integer=int64_t(value);
        if(integer_first) {initial_integer=integer;integer_first=false;}
        if(integer!=initial_integer) integer_uniform=false;
        std::memcpy(target,&integer,sizeof(integer));
      } else std::memcpy(target,&value,sizeof(T));
    };
    if(fortran) for(py::ssize_t z=0;z<shape[2];++z) for(py::ssize_t y=0;y<shape[1];++y) for(py::ssize_t x=0;x<shape[0];++x) visit(x,y,z);
    else for(py::ssize_t x=0;x<shape[0];++x) for(py::ssize_t y=0;y<shape[1];++y) for(py::ssize_t z=0;z<shape[2];++z) visit(x,y,z);
  }
  return py::make_tuple(result,uniform,convert && !uniform && integer_uniform && initial_integer!=0);
}
py::tuple dispatch(py::array image,py::object ids,bool in_place,bool convert) {
  if(image.ndim()!=3) throw py::value_error("native label normalization requires three dimensions");
  if(!py::cast<bool>(image.dtype().attr("isnative"))) throw py::value_error("non-native label byte order is unsupported");
  auto dtype=image.dtype();
#define TYPE(T) if(dtype.equal(py::dtype::of<T>())) return process<T>(image,ids,in_place,convert)
  if(py::cast<std::string>(dtype.attr("kind"))=="f" && image.itemsize()==2) return process<Half>(image,ids,in_place,convert);
  TYPE(uint8_t);TYPE(uint16_t);TYPE(uint32_t);TYPE(uint64_t);TYPE(int8_t);TYPE(int16_t);TYPE(int32_t);TYPE(int64_t);TYPE(float);TYPE(double);
#undef TYPE
  throw py::type_error("label dtype is unsupported");
}
}
void bind_intake(py::module_ &m) {
  m.def("labels_nonzero",[](py::array image,py::object ids) {
    if(image.ndim()!=3) throw py::value_error("native label summary requires three dimensions");
    auto dtype=image.dtype();
    if(!py::cast<bool>(dtype.attr("isnative"))) throw py::value_error("non-native label byte order is unsupported");
#define NONZERO(T) if(dtype.equal(py::dtype::of<T>())) return nonzero<T>(image,ids)
    if(py::cast<std::string>(dtype.attr("kind"))=="f" && image.itemsize()==2) return nonzero<Half>(image,ids);
    NONZERO(uint8_t);NONZERO(uint16_t);NONZERO(uint32_t);NONZERO(uint64_t);NONZERO(int8_t);NONZERO(int16_t);NONZERO(int32_t);NONZERO(int64_t);NONZERO(float);NONZERO(double);
#undef NONZERO
    throw py::type_error("label dtype is unsupported");
  },py::arg("labels"),py::arg("object_ids")=py::none());
  m.def("normalize_object_ids",[](py::dtype dtype,py::iterable ids) {
    std::vector<int64_t> out;
    auto convert=[&](auto value) {for(auto v:integer_ids<decltype(value)>(ids)) out.push_back(int64_t(v));};
#define IDS(T) if(dtype.equal(py::dtype::of<T>())) {convert(T{});return out;}
    IDS(uint8_t);IDS(uint16_t);IDS(uint32_t);IDS(uint64_t);IDS(int8_t);IDS(int16_t);IDS(int32_t);IDS(int64_t);
#undef IDS
    throw py::type_error("No matching signature found for object_ids");
  },py::arg("dtype"),py::arg("object_ids"));
  m.def("normalize_labels",[](py::array image,py::object ids,bool in_place,bool metadata) {
    auto result=dispatch(image,ids,in_place,true);return metadata?result:py::make_tuple(result[0],result[1]);
  },py::arg("labels"),py::arg("object_ids")=py::none(),py::arg("in_place")=false,py::arg("_return_metadata")=false);
  m.def("mask_labels",[](py::array image,py::object ids,bool in_place)->py::object {return dispatch(image,ids,in_place,false)[0];},
        py::arg("labels"),py::arg("object_ids"),py::arg("in_place")=false);
}
