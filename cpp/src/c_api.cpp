// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include "skeleton.hpp"
#include "voxel_graph.hpp"
#include "dlpack.hpp"
#include "packed.hpp"
#include "post.hpp"
#include "nearest.hpp"
#include "point_path.hpp"
#include "streaming.hpp"
#include "oversegment.hpp"
#include "cross_section.hpp"
#include "cpu_post.hpp"
#include <algorithm>

struct brook_context { brook::Context value; explicit brook_context(int d) : value(d) {} };
struct brook_array { brook::Array value; };
struct brook_components { brook::Components value; };
struct brook_host_components { brook::HostComponents value; };
struct brook_skeletons {
  std::vector<int64_t> labels,vertex_offsets{0},edge_offsets{0};
  mutable std::vector<float> vertices,radii;
  mutable std::vector<uint32_t> edges;
  std::array<float,3> anisotropy;
  std::optional<std::vector<uint64_t>> sources;
  std::optional<brook::PackedSkeletons> device;
  mutable std::once_flag host_snapshot;
};
namespace {
thread_local std::string error;
template<class F> brook_status protect(F &&fn) noexcept {
  try { error.clear(); fn(); return BROOK_SUCCESS; }
  catch (const std::invalid_argument &e) { error=e.what(); return BROOK_INVALID_ARGUMENT; }
  catch (const std::bad_alloc &e) { error=e.what(); return BROOK_OUT_OF_MEMORY; }
  catch (const brook::CudaError &e) {
    error=e.what(); return e.code == cudaErrorMemoryAllocation ? BROOK_OUT_OF_MEMORY : BROOK_CUDA_ERROR;
  }
  catch (const std::exception &e) { error=e.what(); return BROOK_INTERNAL_ERROR; }
  catch (...) { error="unknown exception"; return BROOK_INTERNAL_ERROR; }
}
void require(bool condition) { if (!condition) throw std::invalid_argument("null Brook argument"); }
brook::PackedSkeletons on_device(brook::Context &ctx,const brook_skeletons &p) {
  if(p.device) return *p.device;
  return brook::upload_packed(ctx,p.labels,p.vertex_offsets,p.edge_offsets,p.vertices.data(),p.edges.data(),p.radii.data(),p.anisotropy);
}
brook_skeletons *device_result(brook::PackedSkeletons value) {
  auto out=std::make_unique<brook_skeletons>();out->labels=value.labels;out->vertex_offsets=value.vertex_offsets;out->edge_offsets=value.edge_offsets;
  out->anisotropy=value.anisotropy;out->device=std::move(value);return out.release();
}
std::vector<brook::Skeleton> cpu_inputs(const brook_skeletons_view &v) {
  require(v.vertex_count<=SIZE_MAX/3 && v.edge_count<=SIZE_MAX/2);
  require(!v.skeleton_count || (v.labels && v.vertex_offsets && v.edge_offsets));
  require(!v.vertex_count || (v.vertices && v.radii));require(!v.edge_count || v.edges);
  if(!v.skeleton_count) {require(!v.vertex_count && !v.edge_count);return {};}
  require(v.vertex_offsets[0]==0 && v.edge_offsets[0]==0);
  require(v.vertex_offsets[v.skeleton_count]>=0 && uint64_t(v.vertex_offsets[v.skeleton_count])==v.vertex_count);
  require(v.edge_offsets[v.skeleton_count]>=0 && uint64_t(v.edge_offsets[v.skeleton_count])==v.edge_count);
  std::vector<brook::Skeleton> out;out.reserve(v.skeleton_count);
  for(size_t i=0;i<v.skeleton_count;++i) {
    auto a=v.vertex_offsets[i],b=v.vertex_offsets[i+1],c=v.edge_offsets[i],d=v.edge_offsets[i+1];
    require(a>=0 && b>=a && uint64_t(b)<=v.vertex_count && c>=0 && d>=c && uint64_t(d)<=v.edge_count);
    brook::Skeleton s;s.label=v.labels[i];std::copy(v.anisotropy,v.anisotropy+3,s.anisotropy.begin());
    for(auto j=a;j<b;++j){s.vertices.push_back({v.vertices[3*j],v.vertices[3*j+1],v.vertices[3*j+2]});s.radii.push_back(v.radii[j]);}
    for(auto j=c;j<d;++j)s.edges.push_back({v.edges[2*j],v.edges[2*j+1]});out.push_back(std::move(s));
  }return out;
}
void append_cpu(brook_skeletons &out,brook::CpuSkeleton result,uint64_t offset=0) {
  auto &s=result.skeleton;out.labels.push_back(s.label);
  for(auto v:s.vertices)out.vertices.insert(out.vertices.end(),v.begin(),v.end());
  for(auto e:s.edges)out.edges.insert(out.edges.end(),e.begin(),e.end());
  out.radii.insert(out.radii.end(),s.radii.begin(),s.radii.end());
  out.vertex_offsets.push_back(out.radii.size());out.edge_offsets.push_back(out.edges.size()/2);
  for(auto i:result.sources)out.sources->push_back(i+offset);
}

}
extern "C" {
uint32_t brook_abi_version(void) { return BROOK_ABI_VERSION; }
const char *brook_version(void) { return BROOK_VERSION; }
const char *brook_last_error(void) { return error.c_str(); }
void brook_cross_section_options_init(brook_cross_section_options *p) {
  if(!p) return;*p={};p->struct_size=sizeof(*p);p->abi_version=BROOK_ABI_VERSION;
  p->anisotropy[0]=p->anisotropy[1]=p->anisotropy[2]=1;p->smoothing_window=p->step=1;
}
brook_status brook_cross_sectional_area(const brook_volume *labels,const brook_cross_section_input *inputs,
    size_t count,const brook_cross_section_options *options) {
  return protect([&] {
    require(labels&&options&&(!count||inputs));
    if(options->struct_size<sizeof(*options)||options->abi_version!=BROOK_ABI_VERSION) throw std::invalid_argument("invalid cross-section options version");
    brook::CrossSectionOptions o;o.anisotropy={options->anisotropy[0],options->anisotropy[1],options->anisotropy[2]};
    o.smoothing_window=options->smoothing_window;o.step=options->step;o.fill_holes=options->fill_holes;
    o.multipass=options->multipass;o.repair_contacts=options->repair_contacts;o.in_place=options->in_place;
    std::vector<brook::CrossSectionInput> request;request.reserve(count);
    for(size_t i=0;i<count;++i) {
      auto &in=inputs[i];require((!in.vertex_count||(in.vertices&&in.areas&&in.contacts))&&(!in.edge_count||in.edges));
      if(in.vertex_count>UINT32_MAX) throw std::invalid_argument("too many vertices");
      brook::CrossSectionInput q;q.skeleton.label=in.label;q.physical=in.physical;
      q.skeleton.vertices.resize(in.vertex_count);q.skeleton.edges.resize(in.edge_count);
      for(size_t j=0;j<in.vertex_count;++j) q.skeleton.vertices[j]={in.vertices[3*j],in.vertices[3*j+1],in.vertices[3*j+2]};
      for(size_t j=0;j<in.edge_count;++j) q.skeleton.edges[j]={in.edges[2*j],in.edges[2*j+1]};
      if(in.initial_areas) q.areas.assign(in.initial_areas,in.initial_areas+in.vertex_count);
      if(in.initial_contacts) q.contacts.assign(in.initial_contacts,in.initial_contacts+in.vertex_count);
      request.push_back(std::move(q));
    }
    auto result=brook::cross_sectional_area_host(*labels,request,o);
    for(size_t i=0;i<count;++i) {std::copy(result[i].areas.begin(),result[i].areas.end(),inputs[i].areas);std::copy(result[i].contacts.begin(),result[i].contacts.end(),inputs[i].contacts);}
  });
}
brook_status brook_context_create(int device, brook_context **out) {
  return protect([&] { require(out); *out=nullptr; *out=new brook_context(device); });
}
void brook_context_destroy(brook_context *ctx) { delete ctx; }
brook_status brook_context_synchronize(brook_context *ctx) {
  return protect([&] { require(ctx);brook::ContextCall call(ctx->value);ctx->value.synchronize(); });
}
void brook_array_destroy(brook_array *a) { delete a; }
brook_status brook_array_upload(brook_context *ctx,const brook_volume *input,brook_array **out) {
  return protect([&] {
    require(ctx && input && out);*out=nullptr;brook::ContextCall call(ctx->value);
    auto result=brook::upload(ctx->value,*input);ctx->value.synchronize();*out=new brook_array{std::move(result)};
  });
}
brook_status brook_array_view(const brook_array *a, brook_volume *out) {
  return protect([&] { require(a && out); *out=a->value.view(); });
}
brook_status brook_array_copy_to_host(const brook_array *a, void *out, size_t bytes) {
  return protect([&] { require(a); a->value.copy_to_host(out,bytes); });
}
brook_status brook_array_to_dlpack(const brook_array *a,DLManagedTensor **out) {
  return protect([&] { require(a && out);*out=nullptr;*out=brook::export_dlpack(a->value,{kDLCUDA,a->value.device},false); });
}
brook_status brook_array_to_dlpack_versioned(const brook_array *a,uint32_t max_minor,DLManagedTensorVersioned **out) {
  return protect([&] { require(a && out);*out=nullptr;*out=brook::export_dlpack_versioned(a->value,{kDLCUDA,a->value.device},false,max_minor); });
}
brook_status brook_connected_components(brook_context *ctx, const brook_volume *input, brook_components **out) {
  return protect([&] {
    require(ctx && input && out);*out=nullptr;brook::ContextCall call(ctx->value);
    auto labels=brook::upload(ctx->value,*input);
    auto result=brook::connected_components(ctx->value,labels);
    ctx->value.synchronize(); *out=new brook_components{std::move(result)};
  });
}
brook_status brook_nearest_label_voxels(brook_context *ctx,const brook_volume *input,
    const brook_label_query *queries,size_t count,int64_t *coordinates,size_t capacity) {
  return protect([&] {
    require(input && (!count || (queries && coordinates)));
    if(count>INT32_MAX || capacity<count*3) throw std::invalid_argument("invalid nearest-label query/output count");
    std::vector<brook_label_query> request;if(count) request.assign(queries,queries+count);
    std::vector<std::array<int64_t,3>> result;
    if(ctx) { brook::ContextCall call(ctx->value);result=brook::nearest_label_voxels(ctx->value,*input,request); }
    else result=brook::nearest_label_voxels_host(*input,request);
    for(size_t i=0;i<count;++i) for(int axis=0;axis<3;++axis) coordinates[3*i+axis]=result[i][axis];
  });
}
void brook_components_destroy(brook_components *p) { delete p; }
brook_status brook_connected_components_streamed(brook_context *ctx,const brook_volume *input,
                                                 size_t budget,brook_host_components **out) {
  return protect([&] {
    require(ctx && input && out);*out=nullptr;brook::ContextCall call(ctx->value);
    auto result=brook::connected_components_streamed(ctx->value,*input,budget);
    *out=new brook_host_components{std::move(result)};
  });
}
void brook_host_components_destroy(brook_host_components *p) {delete p;}
brook_status brook_host_components_view(const brook_host_components *p,brook_volume *labels,
    const int64_t **mapping,const int64_t **roots,size_t *count) {
  return protect([&] {
    require(p && labels && mapping && roots && count);*labels=p->value.view();
    *mapping=p->value.mapping.data();*roots=p->value.roots.data();*count=p->value.roots.size();
  });
}
brook_status brook_components_view(const brook_components *p, brook_volume *labels,
                                  const int64_t **mapping, const int64_t **roots, size_t *count) {
  return protect([&] {
    require(p && labels && mapping && roots && count); *labels=p->value.labels.view();
    *mapping=p->value.mapping.data(); *roots=p->value.roots.data(); *count=p->value.roots.size();
  });
}
brook_status brook_components_copy_to_host(const brook_components *p, void *out, size_t bytes) {
  return protect([&] { require(p); p->value.labels.copy_to_host(out,bytes); });
}
brook_status brook_components_labels(const brook_components *p,brook_array **out) {
  return protect([&] { require(p && out);*out=nullptr;*out=new brook_array{p->value.labels}; });
}
brook_status brook_edt(brook_context *ctx, const brook_volume *input, const float aniso[3], int bb, brook_array **out) {
  return protect([&] {
    require(ctx && input && aniso && out);*out=nullptr;brook::ContextCall call(ctx->value);
    auto labels=brook::upload(ctx->value,*input);
    auto result=brook::edt(ctx->value,labels,{aniso[0],aniso[1],aniso[2]},bb != 0);
    ctx->value.synchronize(); *out=new brook_array{std::move(result)};
  });
}
brook_status brook_edt_streamed(brook_context *ctx,const brook_volume *input,const float aniso[3],
    int bb,size_t budget,float *output,size_t output_bytes) {
  return protect([&] {
    require(ctx && input && aniso);brook::ContextCall call(ctx->value);
    brook::edt_streamed(ctx->value,*input,{aniso[0],aniso[1],aniso[2]},bb!=0,budget,output,output_bytes/sizeof(float));
  });
}
brook_status brook_graph_components(brook_context *ctx,const brook_volume *input,const brook_volume *graph,brook_components **out) {
  return protect([&] {
    require(ctx && input && graph && out);*out=nullptr;brook::ContextCall call(ctx->value);
    auto labels=brook::upload(ctx->value,*input),vg=brook::upload(ctx->value,*graph);
    auto result=brook::graph_components(ctx->value,labels,vg);
    ctx->value.synchronize();*out=new brook_components{std::move(result)};
  });
}
brook_status brook_graph_edt(brook_context *ctx,const brook_volume *input,const brook_volume *graph,
                            const float aniso[3],int bb,brook_array **out) {
  return protect([&] {
    require(ctx && input && graph && aniso && out);*out=nullptr;brook::ContextCall call(ctx->value);
    auto labels=brook::upload(ctx->value,*input),vg=brook::upload(ctx->value,*graph);
    auto result=brook::graph_edt(ctx->value,labels,vg,{aniso[0],aniso[1],aniso[2]},bb!=0);
    ctx->value.synchronize();*out=new brook_array{std::move(result)};
  });
}
void brook_skeletonize_options_init(brook_skeletonize_options *p) {
  if(!p) return;
  *p={};p->struct_size=sizeof(*p);p->abi_version=BROOK_ABI_VERSION;
  p->teasar.scale=1.5;p->teasar.constant=300;p->teasar.pdrf_scale=100000;p->teasar.pdrf_exponent=4;
  p->teasar.soma_detection_threshold=750;p->teasar.soma_acceptance_threshold=3500;
  p->teasar.soma_invalidation_scale=2;p->teasar.soma_invalidation_const=300;
  p->anisotropy[0]=p->anisotropy[1]=p->anisotropy[2]=1;
  p->dust_threshold=1000;p->flags=BROOK_FIX_BRANCHING|BROOK_FIX_BORDERS;
}
static brook_status skeletonize_impl(brook_context *ctx,const brook_volume *input,const brook_skeletonize_options *p,brook_skeletons **out,bool streamed,size_t budget) {
  return protect([&] {
    require(ctx && input && p && out);*out=nullptr;
    brook::ContextCall call(ctx->value);
    if(p->struct_size<sizeof(*p) || p->abi_version!=BROOK_ABI_VERSION)
      throw std::invalid_argument("incompatible skeletonization options");
    if(p->flags & ~(BROOK_FIX_BRANCHING|BROOK_FIX_BORDERS|BROOK_FILTER_OBJECTS|BROOK_DEVICE_OUTPUT|BROOK_FILL_HOLES|BROOK_FIX_AVOCADOS))
      throw std::invalid_argument("unknown skeletonization flags");
    brook::SkeletonizeOptions options;
    auto &t=options.teasar;
    t.scale=p->teasar.scale;t.constant=p->teasar.constant;t.pdrf_scale=p->teasar.pdrf_scale;t.pdrf_exponent=p->teasar.pdrf_exponent;
    t.soma_detection=p->teasar.soma_detection_threshold;t.soma_acceptance=p->teasar.soma_acceptance_threshold;
    t.soma_scale=p->teasar.soma_invalidation_scale;t.soma_constant=p->teasar.soma_invalidation_const;
    if(p->teasar.has_max_paths) t.max_paths=p->teasar.max_paths;
    std::copy(p->anisotropy,p->anisotropy+3,options.anisotropy.begin());options.dust=p->dust_threshold;
    options.fix_branching=(p->flags&BROOK_FIX_BRANCHING)!=0;options.fix_borders=(p->flags&BROOK_FIX_BORDERS)!=0;
    options.fill_holes=(p->flags&BROOK_FILL_HOLES)!=0;
    options.fix_avocados=(p->flags&BROOK_FIX_AVOCADOS)!=0;options.avocado_detection=p->teasar.soma_detection_threshold;
    if(p->flags&BROOK_FILTER_OBJECTS) {
      require(p->object_ids || !p->object_ids_count);options.object_ids=std::vector<int64_t>{};
      if(p->object_ids_count) options.object_ids->assign(p->object_ids,p->object_ids+p->object_ids_count);
    }
    auto points=[](const int64_t *data,size_t count) {
      require(data || !count);std::vector<brook::Point> value;value.reserve(count);
      for(size_t i=0;i<count;++i) value.push_back({data[3*i],data[3*i+1],data[3*i+2]});return value;
    };
    options.targets_before=points(p->targets_before,p->targets_before_count);
    options.targets_after=points(p->targets_after,p->targets_after_count);
    if(streamed && p->voxel_graph) throw std::invalid_argument("streamed skeletonization does not support voxel_graph");
    if(p->voxel_graph) options.voxel_graph=brook::upload(ctx->value,*p->voxel_graph);
    auto packed=std::make_unique<brook_skeletons>();packed->anisotropy=options.anisotropy;
    if(p->flags&BROOK_DEVICE_OUTPUT) {
      if(streamed) packed->device=brook::pack_skeletons(ctx->value,brook::skeletonize_streamed(ctx->value,*input,options,budget),options.anisotropy);
      else packed->device=brook::skeletonize_packed(ctx->value,brook::upload(ctx->value,*input),options);
      packed->labels=packed->device->labels;packed->vertex_offsets=packed->device->vertex_offsets;packed->edge_offsets=packed->device->edge_offsets;
      *out=packed.release();return;
    }
    auto result=streamed?brook::skeletonize_streamed(ctx->value,*input,options,budget):brook::skeletonize(ctx->value,brook::upload(ctx->value,*input),options);
    for(const auto &s:result) {
      packed->labels.push_back(s.label);
      for(auto v:s.vertices) packed->vertices.insert(packed->vertices.end(),v.begin(),v.end());
      for(auto e:s.edges) packed->edges.insert(packed->edges.end(),e.begin(),e.end());
      packed->radii.insert(packed->radii.end(),s.radii.begin(),s.radii.end());
      packed->vertex_offsets.push_back(packed->radii.size());packed->edge_offsets.push_back(packed->edges.size()/2);
    }
    *out=packed.release();
  });
}
brook_status brook_skeletonize(brook_context *ctx,const brook_volume *input,const brook_skeletonize_options *p,brook_skeletons **out) {
  return skeletonize_impl(ctx,input,p,out,false,0);
}
brook_status brook_skeletonize_streamed(brook_context *ctx,const brook_volume *input,const brook_skeletonize_options *p,
                                      size_t budget,brook_skeletons **out) {
  return skeletonize_impl(ctx,input,p,out,true,budget);
}
brook_status brook_connect_points(brook_context *ctx,const brook_volume *input,const int64_t start[3],
                                  const int64_t end[3],const float spacing[3],double scale,double exponent,
                                  brook_skeletons **out) {
  return protect([&] {
    require(ctx && input && start && end && spacing && out);*out=nullptr;brook::ContextCall call(ctx->value);
    auto sk=brook::connect_points(ctx->value,brook::upload(ctx->value,*input),{start[0],start[1],start[2]},
      {end[0],end[1],end[2]},{spacing[0],spacing[1],spacing[2]},scale,exponent);
    auto result=std::make_unique<brook_skeletons>();result->anisotropy=sk.anisotropy;
    result->labels.push_back(0);result->radii=std::move(sk.radii);
    result->vertices.reserve(3*sk.vertices.size());result->edges.reserve(2*sk.edges.size());
    for(auto v:sk.vertices) result->vertices.insert(result->vertices.end(),v.begin(),v.end());
    for(auto e:sk.edges) result->edges.insert(result->edges.end(),e.begin(),e.end());
    result->vertex_offsets.push_back(sk.vertices.size());result->edge_offsets.push_back(sk.edges.size());
    *out=result.release();
  });
}
brook_status brook_postprocess_cpu(const brook_skeletons_view *input,double dust,double tick,brook_skeletons **out) {
  return protect([&]{require(input && out);*out=nullptr;auto ss=cpu_inputs(*input);auto result=std::make_unique<brook_skeletons>();
    std::copy(input->anisotropy,input->anisotropy+3,result->anisotropy.begin());result->sources.emplace();
    for(size_t i=0;i<ss.size();++i)append_cpu(*result,brook::cpu_postprocess(std::move(ss[i]),dust,tick),input->vertex_offsets[i]);
    *out=result.release();});
}
brook_status brook_join_cpu(const brook_skeletons_view *input,double radius,int restricted,brook_skeletons **out) {
  return protect([&]{require(input && out);*out=nullptr;auto ss=cpu_inputs(*input);auto result=std::make_unique<brook_skeletons>();
    std::copy(input->anisotropy,input->anisotropy+3,result->anisotropy.begin());result->sources.emplace();
    append_cpu(*result,brook::cpu_join(std::move(ss),radius,restricted!=0));*out=result.release();});
}
brook_status brook_skeletons_get_sources(const brook_skeletons *p,const uint64_t **sources,size_t *count) {
  return protect([&]{require(p && sources && count);*sources=nullptr;*count=0;
    if(!p->sources)throw std::invalid_argument("result has no source indices");
    *sources=p->sources->data();*count=p->sources->size();});
}
brook_status brook_skeletons_get_view(const brook_skeletons *p,brook_skeletons_view *out) {
  return protect([&] {
    require(p && out);*out={};
    std::call_once(p->host_snapshot,[&] {
      if(!p->device) return;
      p->vertices.resize(3*p->vertex_offsets.back());p->radii.resize(p->vertex_offsets.back());p->edges.resize(2*p->edge_offsets.back());
      p->device->vertices.copy_to_host(p->vertices.data(),p->vertices.size()*sizeof(float));
      p->device->radii.copy_to_host(p->radii.data(),p->radii.size()*sizeof(float));
      p->device->edges.copy_to_host(p->edges.data(),p->edges.size()*sizeof(uint32_t));
    });
    out->skeleton_count=p->labels.size();out->vertex_count=p->radii.size();out->edge_count=p->edges.size()/2;
    out->labels=p->labels.data();out->vertex_offsets=p->vertex_offsets.data();out->edge_offsets=p->edge_offsets.data();
    out->vertices=p->vertices.data();out->edges=p->edges.data();out->radii=p->radii.data();
    std::copy(p->anisotropy.begin(),p->anisotropy.end(),out->anisotropy);
  });
}
brook_status brook_skeletons_get_device_view(const brook_skeletons *p,brook_skeletons_device_view *out) {
  return protect([&] {
    require(p && out);*out={};
    if(!p->device) throw std::invalid_argument("skeleton result was not requested with BROOK_DEVICE_OUTPUT");
    out->skeleton_count=p->labels.size();out->vertex_count=p->vertex_offsets.back();out->edge_count=p->edge_offsets.back();
    out->labels=p->labels.data();out->vertex_offsets=p->vertex_offsets.data();out->edge_offsets=p->edge_offsets.data();
    out->vertices=p->device->vertices.view();out->edges=p->device->edges.view();out->radii=p->device->radii.view();
    std::copy(p->anisotropy.begin(),p->anisotropy.end(),out->anisotropy);
  });
}
brook_status brook_skeletons_get_array(const brook_skeletons *p,brook_skeleton_buffer buffer,brook_array **out) {
  return protect([&] {
    require(p && out);*out=nullptr;
    if(!p->device) throw std::invalid_argument("skeleton result was not requested with BROOK_DEVICE_OUTPUT");
    const brook::Array *value=nullptr;
    switch(buffer) {
      case BROOK_VERTICES:value=&p->device->vertices;break;
      case BROOK_EDGES:value=&p->device->edges;break;
      case BROOK_RADII:value=&p->device->radii;break;
      default:throw std::invalid_argument("invalid skeleton buffer");
    }
    *out=new brook_array{*value};
  });
}
brook_status brook_oversegment(brook_context *ctx,const brook_volume *labels,
    const brook_skeletons_view *input,const double spacing[3],int downsample,uint32_t flags,
    uint64_t *features,size_t feature_capacity,uint64_t *segments,size_t segment_capacity,
    uint8_t *processed,size_t processed_capacity) {
 return protect([&] {
  require(labels && input && spacing);
  size_t n=brook::volume_size({labels->shape[0],labels->shape[1],labels->shape[2]});
  require((!n || features) && (!input->vertex_count || (segments && input->vertices)) &&
          (!input->skeleton_count || (input->labels && input->vertex_offsets && input->edge_offsets)) && (!input->edge_count || input->edges));
  if(feature_capacity<n || segment_capacity<input->vertex_count || (processed && processed_capacity<input->skeleton_count)) throw std::invalid_argument("oversegment output capacity too small");
  if(input->skeleton_count && (input->vertex_offsets[0] || input->edge_offsets[0] ||
      input->vertex_offsets[input->skeleton_count]!=int64_t(input->vertex_count) || input->edge_offsets[input->skeleton_count]!=int64_t(input->edge_count))) throw std::invalid_argument("invalid packed offsets");
  brook::OversegmentOptions options;options.downsample=downsample;
  for(int a=0;a<3;++a) options.spacing[a]=spacing[a];
  options.binary_labels=flags&BROOK_OVERSEGMENT_BINARY;options.fill_holes=flags&BROOK_OVERSEGMENT_FILL_HOLES;
  options.in_place=flags&BROOK_OVERSEGMENT_IN_PLACE;options.float32_coordinates=flags&BROOK_OVERSEGMENT_FLOAT32_COORDINATES;
  std::vector<brook::OversegmentSkeleton> skeletons(input->skeleton_count);
  for(size_t i=0;i<input->skeleton_count;++i) {
   auto v0=input->vertex_offsets[i],v1=input->vertex_offsets[i+1],e0=input->edge_offsets[i],e1=input->edge_offsets[i+1];
   if(v0<0 || v1<v0 || size_t(v1)>input->vertex_count || e0<0 || e1<e0 || size_t(e1)>input->edge_count) throw std::invalid_argument("invalid packed offsets");
   auto &s=skeletons[i];s.label=input->labels[i];s.float32_coordinates=options.float32_coordinates;s.vertices.resize(v1-v0);s.edges.resize(e1-e0);
   for(int64_t j=v0;j<v1;++j) for(int a=0;a<3;++a) s.vertices[j-v0][a]=input->vertices[3*j+a];
   for(int64_t j=e0;j<e1;++j) for(int a=0;a<2;++a) s.edges[j-e0][a]=input->edges[2*j+a];
  }
  brook::Oversegmentation result;
  if(ctx) {brook::ContextCall call(ctx->value);result=brook::oversegment(&ctx->value,*labels,skeletons,options);}
  else result=brook::oversegment(nullptr,*labels,skeletons,options);
  std::copy(result.features.begin(),result.features.end(),features);std::copy(result.segments.begin(),result.segments.end(),segments);
  if(processed) std::copy(result.processed.begin(),result.processed.end(),processed);
 });
}
void brook_skeletons_destroy(brook_skeletons *p) { delete p; }
brook_status brook_merge_skeletons(brook_context *ctx,const brook_skeletons *const *inputs,const float *origins,size_t count,brook_skeletons **out) {
  return protect([&] {
    require(ctx && out && (inputs || !count));*out=nullptr;brook::ContextCall call(ctx->value);
    std::vector<brook::PackedSkeletons> fragments;std::vector<std::array<float,3>> offsets;
    fragments.reserve(count);offsets.reserve(count);
    for(size_t i=0;i<count;++i) { require(inputs[i]);fragments.push_back(on_device(ctx->value,*inputs[i]));
      offsets.push_back(origins?std::array<float,3>{origins[3*i],origins[3*i+1],origins[3*i+2]}:std::array<float,3>{0,0,0}); }
    *out=device_result(brook::merge_fragments(ctx->value,fragments,offsets));
  });
}
brook_status brook_postprocess_canonical(brook_context *ctx,const brook_skeletons *input,double dust,double ticks,brook_skeletons **out) {
  return protect([&] {
    require(ctx && input && out);*out=nullptr;brook::ContextCall call(ctx->value);
    *out=device_result(brook::postprocess(ctx->value,on_device(ctx->value,*input),dust,ticks));
  });
}
}
