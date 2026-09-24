// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "brook.h"
#include "dlpack.h"
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <array>
#include <cmath>

namespace brook::api {
inline void cross_sectional_area(const brook_volume &labels,
    const std::vector<brook_cross_section_input> &inputs,const brook_cross_section_options &options) {
  auto status=brook_cross_sectional_area(&labels,inputs.data(),inputs.size(),&options);
  if(status!=BROOK_SUCCESS) throw std::runtime_error(brook_last_error());
}
inline void check(brook_status status) {
  if(status!=BROOK_SUCCESS) throw std::runtime_error(brook_last_error());
}
inline brook_skeletonize_options skeletonize_defaults() {
  brook_skeletonize_options value;brook_skeletonize_options_init(&value);return value;
}
struct DLPackDeleter {
  void operator()(DLManagedTensor *p) const noexcept { if(p && p->deleter) p->deleter(p); }
};
using DLPackTensor=std::unique_ptr<DLManagedTensor,DLPackDeleter>;
class Array {
  struct Destroy { void operator()(brook_array *p) const noexcept { brook_array_destroy(p); } };
  std::unique_ptr<brook_array,Destroy> handle_;
 public:
  explicit Array(brook_array *handle):handle_(handle) {}
  Array(Array &&)=default;
  Array &operator=(Array &&)=default;
  brook_volume view() const {
    brook_volume out{};check(brook_array_view(handle_.get(),&out));return out;
  }
  void copy_to_host(void *destination,size_t bytes) const { check(brook_array_copy_to_host(handle_.get(),destination,bytes)); }
  // A zero-copy DLPack export; the tensor keeps the storage alive after the Array is destroyed.
  DLPackTensor dlpack() const {
    DLManagedTensor *out=nullptr;check(brook_array_to_dlpack(handle_.get(),&out));return DLPackTensor(out);
  }
};
struct ComponentsView {
  brook_volume labels{};
  const int64_t *mapping=nullptr,*roots=nullptr;
  size_t count=0;
};
class Components {
  struct Destroy { void operator()(brook_components *p) const noexcept { brook_components_destroy(p); } };
  std::unique_ptr<brook_components,Destroy> handle_;
 public:
  explicit Components(brook_components *handle):handle_(handle) {}
  Components(Components &&)=default;
  Components &operator=(Components &&)=default;
  ComponentsView view() const {
    ComponentsView out;check(brook_components_view(handle_.get(),&out.labels,&out.mapping,&out.roots,&out.count));return out;
  }
  Array labels() const {
    brook_array *out=nullptr;check(brook_components_labels(handle_.get(),&out));return Array(out);
  }
};
class HostComponents {
  struct Destroy {void operator()(brook_host_components *p) const noexcept {brook_host_components_destroy(p);}};
  std::unique_ptr<brook_host_components,Destroy> handle_;
 public:
  explicit HostComponents(brook_host_components *p):handle_(p) {}
  HostComponents(HostComponents &&)=default;
  HostComponents &operator=(HostComponents &&)=default;
  ComponentsView view() const {
    ComponentsView out;
    check(brook_host_components_view(handle_.get(),&out.labels,&out.mapping,&out.roots,&out.count));return out;
  }
};
// Owns a brook_skeletons result: host or device views of the packed skeletons.
class Skeletons {
  struct Destroy { void operator()(brook_skeletons *p) const noexcept { brook_skeletons_destroy(p); } };
  std::unique_ptr<brook_skeletons,Destroy> handle_;
 public:
  explicit Skeletons(brook_skeletons *handle):handle_(handle) {}
  Skeletons(Skeletons &&)=default;
  Skeletons &operator=(Skeletons &&)=default;
  brook_skeletons *get() const noexcept { return handle_.get(); }
  brook_skeletons_view view() const {
    brook_skeletons_view out;check(brook_skeletons_get_view(handle_.get(),&out));return out;
  }
  std::pair<const uint64_t*,size_t> sources() const {
    const uint64_t *data=nullptr;size_t count=0;check(brook_skeletons_get_sources(handle_.get(),&data,&count));return {data,count};
  }
  brook_skeletons_device_view device_view() const {
    brook_skeletons_device_view out;check(brook_skeletons_get_device_view(handle_.get(),&out));return out;
  }
  Array array(brook_skeleton_buffer buffer) const {
    brook_array *out=nullptr;check(brook_skeletons_get_array(handle_.get(),buffer,&out));return Array(out);
  }
};
struct Oversegmentation {
  std::vector<uint64_t> features,segments;
  std::vector<uint8_t> processed;
};
// Kimimaro-compatible oversegment; a null context works unless flags request FILL_HOLES.
inline Oversegmentation oversegment(const brook_volume &labels,const brook_skeletons_view &skeletons,
    std::array<double,3> spacing={1,1,1},int downsample=0,uint32_t flags=BROOK_OVERSEGMENT_FLOAT32_COORDINATES,
    brook_context *context=nullptr) {
  size_t n=1;for(auto extent:labels.shape) {
    if(extent<0 || (extent && n>SIZE_MAX/size_t(extent))) throw std::invalid_argument("invalid volume shape");
    n*=size_t(extent);
  }
  Oversegmentation result;result.features.resize(n);result.segments.resize(skeletons.vertex_count);result.processed.resize(skeletons.skeleton_count);
  check(brook_oversegment(context,&labels,&skeletons,spacing.data(),downsample,flags,
    result.features.data(),result.features.size(),result.segments.data(),result.segments.size(),result.processed.data(),result.processed.size()));
  return result;
}
// Kimimaro-compatible postprocessing and joining, computed on the CPU; no CUDA context required.
inline Skeletons postprocess_cpu(const brook_skeletons_view &input,double dust=1500,double tick=3500) {
  brook_skeletons *out=nullptr;check(brook_postprocess_cpu(&input,dust,tick,&out));return Skeletons(out);
}
inline Skeletons join_cpu(const brook_skeletons_view &input,double radius=INFINITY,bool restricted=false) {
  brook_skeletons *out=nullptr;check(brook_join_cpu(&input,radius,restricted,&out));return Skeletons(out);
}
// A brook_context on one CUDA device; its calls block. Errors throw std::runtime_error.
class Context {
  struct Destroy { void operator()(brook_context *p) const noexcept { brook_context_destroy(p); } };
  std::unique_ptr<brook_context,Destroy> handle_;
 public:
  explicit Context(int device=0) {
    brook_context *p=nullptr;check(brook_context_create(device,&p));handle_.reset(p);
  }
  Context(Context &&)=default;
  Context &operator=(Context &&)=default;
  brook_context *get() const noexcept { return handle_.get(); }
  void synchronize() const { check(brook_context_synchronize(handle_.get())); }
  Array upload(const brook_volume &volume) {
    brook_array *out=nullptr;check(brook_array_upload(handle_.get(),&volume,&out));return Array(out);
  }
  HostComponents connected_components_streamed(const brook_volume &volume,size_t budget_bytes=0) {
    brook_host_components *out=nullptr;
    check(brook_connected_components_streamed(handle_.get(),&volume,budget_bytes,&out));return HostComponents(out);
  }
  Components connected_components(const brook_volume &volume,const brook_volume *graph=nullptr) {
    brook_components *out=nullptr;
    check(graph?brook_graph_components(handle_.get(),&volume,graph,&out):brook_connected_components(handle_.get(),&volume,&out));
    return Components(out);
  }
  void edt_streamed(const brook_volume &volume,const float anisotropy[3],float *output,size_t output_bytes,
                    bool black_border=false,size_t budget_bytes=0) {
    check(brook_edt_streamed(handle_.get(),&volume,anisotropy,black_border,budget_bytes,output,output_bytes));
  }
  Array edt(const brook_volume &volume,const float anisotropy[3],bool black_border=false,const brook_volume *graph=nullptr) {
    brook_array *out=nullptr;
    check(graph?brook_graph_edt(handle_.get(),&volume,graph,anisotropy,black_border,&out):brook_edt(handle_.get(),&volume,anisotropy,black_border,&out));
    return Array(out);
  }
  Skeletons skeletonize(const brook_volume &volume,const brook_skeletonize_options &options=skeletonize_defaults()) {
    brook_skeletons *out=nullptr;check(brook_skeletonize(handle_.get(),&volume,&options,&out));return Skeletons(out);
  }
  Skeletons skeletonize_streamed(const brook_volume &volume,const brook_skeletonize_options &options=skeletonize_defaults(),size_t budget_bytes=0) {
    brook_skeletons *out=nullptr;check(brook_skeletonize_streamed(handle_.get(),&volume,&options,budget_bytes,&out));return Skeletons(out);
  }
  // Kimimaro-compatible connect_points: one path from end to start, in physical coordinates.
  Skeletons connect_points(const brook_volume &volume,std::array<int64_t,3> start,std::array<int64_t,3> end,
                           std::array<float,3> anisotropy={1,1,1},double scale=100000,double exponent=4) {
    brook_skeletons *out=nullptr;
    check(brook_connect_points(handle_.get(),&volume,start.data(),end.data(),anisotropy.data(),scale,exponent,&out));
    return Skeletons(out);
  }
  Oversegmentation oversegment(const brook_volume &labels,const brook_skeletons_view &skeletons,
      std::array<double,3> spacing={1,1,1},int downsample=0,uint32_t flags=BROOK_OVERSEGMENT_FLOAT32_COORDINATES) {
    return api::oversegment(labels,skeletons,spacing,downsample,flags,handle_.get());
  }
  Skeletons merge(const std::vector<const Skeletons*> &fragments,const std::vector<std::array<float,3>> &origins) {
    if(fragments.size()!=origins.size()) throw std::invalid_argument("fragment/origin count mismatch");
    std::vector<const brook_skeletons*> handles;for(auto p:fragments) handles.push_back(p?p->get():nullptr);
    brook_skeletons *out=nullptr;check(brook_merge_skeletons(handle_.get(),handles.data(),origins.empty()?nullptr:origins.front().data(),handles.size(),&out));return Skeletons(out);
  }
  Skeletons postprocess_canonical(const Skeletons &input,double dust_threshold=1500,double tick_threshold=3500) {
    brook_skeletons *out=nullptr;check(brook_postprocess_canonical(handle_.get(),input.get(),dust_threshold,tick_threshold,&out));return Skeletons(out);
  }
  std::vector<std::array<int64_t,3>> nearest_label_voxels(const brook_volume &volume,const std::vector<brook_label_query> &queries) {
    if(queries.size()>INT32_MAX) throw std::invalid_argument("too many nearest-label queries");
    std::vector<std::array<int64_t,3>> result(queries.size());
    check(brook_nearest_label_voxels(handle_.get(),&volume,queries.data(),queries.size(),
      result.empty()?nullptr:result.front().data(),result.size()*3));return result;
  }
};
inline std::vector<std::array<int64_t,3>> nearest_label_voxels_host(const brook_volume &volume,
    const std::vector<brook_label_query> &queries) {
  if(queries.size()>INT32_MAX) throw std::invalid_argument("too many nearest-label queries");
  std::vector<std::array<int64_t,3>> result(queries.size());
  check(brook_nearest_label_voxels(nullptr,&volume,queries.data(),queries.size(),
    result.empty()?nullptr:result.front().data(),result.size()*3));return result;
}
}
