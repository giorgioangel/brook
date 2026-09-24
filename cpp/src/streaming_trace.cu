// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "streaming.hpp"
#include "graph.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace brook {
namespace {
struct Pinned {
  void *data=nullptr;
  explicit Pinned(size_t bytes) {if(bytes) BROOK_CUDA(cudaHostAlloc(&data,bytes,cudaHostAllocDefault));}
  ~Pinned() {if(data) cudaFreeHost(data);}
  Pinned(const Pinned&)=delete;
};
struct FreeFloat {void operator()(float *p) const noexcept {std::free(p);}};

std::vector<Box> streamed_boxes(Context &ctx,const HostComponents &cc,size_t budget) {
  return analyze_streamed(ctx,cc.view(),cc.roots.size(),budget);
}

std::vector<std::vector<Point>> host_targets(const HostComponents &cc,const std::vector<Point> &points) {
  std::vector<std::vector<Point>> result(cc.mapping.size());
  size_t item=dtype_size(cc.dtype);
  for(auto original:points) {
    auto p=original;
    for(int a=0;a<3;++a) {
      if(p[a]<0) p[a]+=cc.shape[a];
      if(p[a]<0 || p[a]>=cc.shape[a]) throw std::invalid_argument("extra target is outside the input volume");
    }
    uint64_t label=0;std::memcpy(&label,cc.labels.get()+size_t(flatten(p,cc.shape))*item,item);
    // Lookup uses NumPy's negative indexing; the supplied coordinate itself is
    // retained for subsequent translation, as in Kimimaro's points_to_labels.
    result.at(size_t(label)).push_back(original);
  }
  return result;
}
void translate(std::vector<Point> &points,Point origin) {
  for(auto &p:points) for(int a=0;a<3;++a) p[a]-=origin[a];
}
template<class Label> __global__ void mask_crop(const Label *labels,float *distance,uint8_t *mask,
                                               Label label,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) {bool live=labels[i]==label;mask[i]=live;if(!live) distance[i]=0.0f;}
}
std::pair<Array,Array> host_crop(Context &ctx,const HostComponents &cc,const float *distance,
    int64_t label,const Box &box,Pinned &label_stage,Pinned &distance_stage) {
  std::array<int64_t,3> s;
  for(int a=0;a<3;++a) s[a]=box.hi[a]-box.lo[a]+1;
  size_t item=dtype_size(cc.dtype),row=size_t(s[0]);
  auto *staged_labels=static_cast<uint8_t*>(label_stage.data);
  auto *staged_distance=static_cast<float*>(distance_stage.data);
  size_t at=0;
  for(int64_t z=box.lo[2];z<=box.hi[2];++z) for(int64_t y=box.lo[1];y<=box.hi[1];++y,at+=row) {
    size_t source=size_t(box.lo[0]+cc.shape[0]*(y+cc.shape[1]*z));
    std::memcpy(staged_labels+at*item,cc.labels.get()+source*item,row*item);
    std::memcpy(staged_distance+at,distance+source,row*sizeof(float));
  }
  auto labels=allocate(ctx,s,cc.dtype),field=allocate(ctx,s,BROOK_F32),mask=allocate(ctx,s,BROOK_U8);
  BROOK_CUDA(cudaMemcpyAsync(labels.data(),staged_labels,labels.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  BROOK_CUDA(cudaMemcpyAsync(field.data(),staged_distance,field.bytes(),cudaMemcpyHostToDevice,ctx.stream));
  unsigned blocks=static_cast<unsigned>((mask.size()+255)/256);long long n=static_cast<long long>(mask.size());
  auto *d=static_cast<float*>(field.data());auto *m=static_cast<uint8_t*>(mask.data());
  switch(cc.dtype) {
    case BROOK_U8:mask_crop<<<blocks,256,0,ctx.stream>>>(static_cast<const uint8_t*>(labels.data()),d,m,uint8_t(label),n);break;
    case BROOK_U16:mask_crop<<<blocks,256,0,ctx.stream>>>(static_cast<const uint16_t*>(labels.data()),d,m,uint16_t(label),n);break;
    case BROOK_U32:mask_crop<<<blocks,256,0,ctx.stream>>>(static_cast<const uint32_t*>(labels.data()),d,m,uint32_t(label),n);break;
    case BROOK_U64:mask_crop<<<blocks,256,0,ctx.stream>>>(static_cast<const uint64_t*>(labels.data()),d,m,uint64_t(label),n);break;
    default:throw std::invalid_argument("invalid component crop type");
  }
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return {std::move(mask),std::move(field)};
}
}

std::vector<Box> analyze_streamed(Context &ctx,const brook_volume &input,size_t count,size_t budget) {
  const std::array<int64_t,3> s={input.shape[0],input.shape[1],input.shape[2]};
  if(!volume_size(s)) return std::vector<Box>(count+1);
  budget=default_stream_budget(budget);
  const size_t plane=size_t(s[0])*size_t(s[1]);
  const int64_t depth=int64_t(std::max(size_t{1},std::min(size_t(s[2]),budget/plane/(dtype_size(input.dtype)+32))));
  std::vector<Box> result(count+1);
  for(int64_t first=0;first<s[2];first+=depth) {
    auto view=input;view.shape[2]=std::min(depth,s[2]-first);
    view.data=static_cast<const char*>(input.data)+first*input.strides[2];
    auto boxes=analyze(ctx,upload_stream_region(ctx,view),count);
    result[0].count+=boxes[0].count;
    for(size_t label=1;label<boxes.size();++label) if(boxes[label].count) {
      auto &out=result[label];const auto &in=boxes[label];out.count+=in.count;
      for(int a=0;a<3;++a) {
        int64_t offset=a==2?first:0;
        out.lo[a]=std::min(out.lo[a],in.lo[a]+offset);out.hi[a]=std::max(out.hi[a],in.hi[a]+offset);
      }
    }
  }
  return result;
}
std::vector<Skeleton> skeletonize_streamed(Context &ctx,const brook_volume &input,
                                         const SkeletonizeOptions &options,size_t budget) {
  ctx.activate();ctx.stats.clear();ctx.stats["graphs"]=graphs_enabled();
  if(input.struct_size<sizeof(input) || input.abi_version!=BROOK_ABI_VERSION || input.memory!=BROOK_HOST)
    throw std::invalid_argument("streamed skeletonization requires a compatible host volume");
  const std::array<int64_t,3> shape={input.shape[0],input.shape[1],input.shape[2]};
  const size_t n=volume_size(shape);
  if(!n || double(n)<=options.dust) return {};
  if(options.voxel_graph.storage || options.fill_holes || options.fix_avocados)
    throw std::invalid_argument("streamed skeletonization does not support voxel_graph, fill_holes or fix_avocados");
  budget=default_stream_budget(budget);
  auto cc=connected_components_streamed(ctx,input,budget,options.object_ids?&*options.object_ids:nullptr);
  const size_t count=cc.roots.size();if(!count) return {};
  auto boxes=streamed_boxes(ctx,cc,budget);
  bool black_border=options.black_border_override.value_or(count==1 && boxes[1].count==int64_t(n));
  if(n>SIZE_MAX/sizeof(float)) throw std::invalid_argument("host distance size overflow");
  std::unique_ptr<float,FreeFloat> distance(static_cast<float*>(std::malloc(n*sizeof(float))));
  if(!distance) throw std::bad_alloc();
  edt_streamed(ctx,cc.view(),options.anisotropy,black_border,budget,distance.get(),n);
  auto borders=options.fix_borders?border_targets_host(ctx,cc.view(),count,options.anisotropy):std::vector<std::vector<Point>>(count+1);
  auto extra_before=host_targets(cc,options.targets_before),extra_after=host_targets(cc,options.targets_after);
  const double exponent=options.teasar.pdrf_exponent;
  if(std::isfinite(exponent) && exponent==std::floor(exponent) && std::fmod(exponent,2.0)==1.0)
    throw std::invalid_argument("odd pdrf_exponent makes the background penalty negative");
  size_t maximum=0;
  for(size_t id=1;id<=count;++id) if(boxes[id].count && double(boxes[id].count)>options.dust) {
    std::array<int64_t,3> s;for(int a=0;a<3;++a) s[a]=boxes[id].hi[a]-boxes[id].lo[a]+1;
    size_t size=volume_size(s);
    if(size>=size_t{1}<<31) throw std::invalid_argument("component bounding box exceeds int32 worklist capacity");
    if(size>1) maximum=std::max(maximum,size);
  }
  if(!maximum) return {};
  const size_t item=dtype_size(cc.dtype);
  if(maximum>SIZE_MAX/item || maximum>SIZE_MAX/sizeof(float)) throw std::invalid_argument("pinned crop size overflow");
  Pinned labels_stage(maximum*item),distance_stage(maximum*sizeof(float));
  std::vector<Skeleton> result;std::unordered_map<int64_t,size_t> index;
  for(size_t id=1;id<=count;++id) {
    const auto &box=boxes[id];if(!box.count || double(box.count)<=options.dust) continue;
    std::array<int64_t,3> s;for(int a=0;a<3;++a) s[a]=box.hi[a]-box.lo[a]+1;
    if(volume_size(s)<=1) continue;
    auto inputs=host_crop(ctx,cc,distance.get(),int64_t(id),box,labels_stage,distance_stage);
    auto before=borders[id],after=extra_after[id];std::optional<Point> root;
    if(!before.empty()) {root=before.back();before.pop_back();for(int a=0;a<3;++a) (*root)[a]-=box.lo[a];}
    before.insert(before.end(),extra_before[id].begin(),extra_before[id].end());
    translate(before,box.lo);translate(after,box.lo);
    auto sk=trace_component(ctx,std::move(inputs.first),std::move(inputs.second),options,std::move(before),std::move(after),root);
    ++ctx.stats["streamed_traced_components"];
    for(auto &v:sk.vertices) for(int a=0;a<3;++a) v[a]=(v[a]+float(box.lo[a]))*options.anisotropy[a];
    if(sk.edges.empty()) continue;
    sk.label=cc.mapping[id];
    auto entry=index.find(sk.label);
    if(entry==index.end()) {index[sk.label]=result.size();result.push_back(std::move(sk));}
    else {
      auto &out=result[entry->second];size_t base=out.vertices.size();
      if(base+sk.vertices.size()>UINT32_MAX) throw std::invalid_argument("merged skeleton exceeds uint32 vertex capacity");
      out.vertices.insert(out.vertices.end(),sk.vertices.begin(),sk.vertices.end());
      out.radii.insert(out.radii.end(),sk.radii.begin(),sk.radii.end());
      for(auto edge:sk.edges) out.edges.push_back({uint32_t(base+edge[0]),uint32_t(base+edge[1])});
    }
  }
  for(auto &s:result) consolidate(s);
  return result;
}
}
