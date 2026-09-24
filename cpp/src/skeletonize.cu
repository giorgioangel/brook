// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "skeleton.hpp"
#include "graph.hpp"
#include "lockstep.hpp"
#include "soma.hpp"
#include "voxel_graph.hpp"
#include "packed.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

namespace brook {
int64_t component_at(Context &ctx,const Array &cc,Point p) {
  for(int axis=0;axis<3;++axis) {
    if(p[axis]<0) p[axis]+=cc.shape[axis];
    if(p[axis]<0 || p[axis]>=cc.shape[axis]) throw std::invalid_argument("extra target is outside the input volume");
  }
  uint64_t value=0;
  const char *source=static_cast<const char*>(cc.data())+flatten(p,cc.shape)*dtype_size(cc.dtype);
  BROOK_CUDA(cudaMemcpyAsync(&value,source,dtype_size(cc.dtype),cudaMemcpyDeviceToHost,ctx.stream));
  ctx.synchronize();return static_cast<int64_t>(value);
}
namespace {
template<class Label> __global__ void mask_ids(const Label *input,Label *output,long long n,const long long *ids,int count) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n) return;
  long long value=static_cast<long long>(input[i]);int lo=0,hi=count;
  while(lo<hi) { int mid=lo+(hi-lo)/2;if(ids[mid]<value) lo=mid+1;else hi=mid; }
  output[i]=(lo<count && ids[lo]==value)?input[i]:Label(0);
}
template<class Label> void filter(Context &ctx,const Array &input,Array &output,const Buffer<long long> &ids) {
  mask_ids<<<static_cast<unsigned>((input.size()+255)/256),256,0,ctx.stream>>>(static_cast<const Label*>(input.data()),
    static_cast<Label*>(output.data()),input.size(),ids.data(),ids.size());
  BROOK_CUDA(cudaGetLastError());
}
Array filter(Context &ctx,const Array &input,std::vector<int64_t> ids) {
  std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
  std::vector<long long> values(ids.begin(),ids.end());Buffer<long long> d(ctx,values.size());d.set(ctx,values.data(),values.size());
  auto output=allocate(ctx,input.shape,input.dtype);
  switch(input.dtype) {
    case BROOK_U8:filter<uint8_t>(ctx,input,output,d);break;
    case BROOK_U16:filter<uint16_t>(ctx,input,output,d);break;
    case BROOK_U32:filter<uint32_t>(ctx,input,output,d);break;
    case BROOK_U64:filter<uint64_t>(ctx,input,output,d);break;
    case BROOK_I8:filter<int8_t>(ctx,input,output,d);break;
    case BROOK_I16:filter<int16_t>(ctx,input,output,d);break;
    case BROOK_I32:filter<int32_t>(ctx,input,output,d);break;
    case BROOK_I64:filter<int64_t>(ctx,input,output,d);break;
    default:throw std::invalid_argument("integer labels required");
  }
  ctx.synchronize();return output;
}
void translate(std::vector<Point> &points,Point origin) {
  for(auto &point:points) for(int axis=0;axis<3;++axis) point[axis]-=origin[axis];
}
// The points a component traces: each is kept as the caller wrote it, so the coordinates the
// trace flattens are the caller's own even when a lookup wrapped a negative one.
std::vector<std::vector<Point>> assign_points(Context &ctx,const Array &cc,size_t count,const std::vector<Point> &points) {
  std::vector<std::vector<Point>> result(count+1);
  for(const auto &p:points) result.at(size_t(component_at(ctx,cc,p))).push_back(p);
  return result;
}
}

Array filter_labels(Context &ctx,const Array &input,const std::vector<int64_t> &ids) {return filter(ctx,input,ids);}

PreparedSample prepare_sample(Context &ctx,Array labels,const SkeletonizeOptions &options,bool lockstep_wanted) {
  PreparedSample p;
  using clock=std::chrono::steady_clock;auto t=clock::now();
  auto lap=[&](const char *key){ auto now=clock::now();ctx.stats[key]+=std::chrono::duration_cast<std::chrono::microseconds>(now-t).count();t=now; };
  if(!labels.size() || double(labels.size())<=options.dust) return p;
  if(options.object_ids) labels=filter(ctx,labels,*options.object_ids);
  const Array *graph=options.voxel_graph.storage?&options.voxel_graph:nullptr;
  auto &cc=p.cc;cc=graph?graph_components(ctx,labels,*graph):connected_components(ctx,labels);labels={};lap("prep_ccl_us");
  if(graph) { p.trace_graph=graph_for_trace(ctx,*graph);graph=&p.trace_graph; }
  size_t &count=p.count;count=cc.roots.size();
  if(!count) return p;
  auto &boxes=p.boxes;boxes=analyze(ctx,cc.labels,count);lap("prep_analyze_us");
  bool black_border=options.black_border_override.value_or(graph?cc.input_uniform:(count==1 && boxes[1].count==static_cast<int64_t>(cc.labels.size())));
  // Holes are filled per component (per graph component with a voxel graph, whose bits stay as given: a
  // filled voxel joins only where the graph permits), as in Kimimaro.
  // black_border reflects the original volume, before any holes are filled.
  if(options.fill_holes) {
    auto filled=fill_all_holes(ctx,std::move(cc.labels),boxes);
    cc.labels=std::move(filled.first);ctx.stats["hole_filled_voxels"]=filled.second;
    if(filled.second) { boxes=analyze(ctx,cc.labels,count);refresh_roots(ctx,cc.labels,cc.roots); }
  }
  auto &dbf=p.dbf;dbf=graph?graph_edt(ctx,cc.labels,*graph,options.anisotropy,black_border):edt(ctx,cc.labels,options.anisotropy,black_border);lap("prep_edt_us");
  if(options.fix_avocados) {
    fix_avocados(ctx,cc,dbf,options.avocado_detection,options.anisotropy,black_border,0,graph);
    count=cc.roots.size();if(!count) return p;
    boxes=analyze(ctx,cc.labels,count);
  }
  auto &borders=p.borders;borders.assign(count+1,{});
  if(options.fix_borders) borders=border_targets(ctx,cc.labels,count,options.anisotropy);lap("prep_borders_us");
  p.extra_before=assign_points(ctx,cc.labels,count,options.targets_before);
  p.extra_after=assign_points(ctx,cc.labels,count,options.targets_after);lap("prep_targets_us");
  double exponent=options.teasar.pdrf_exponent;
  if(std::isfinite(exponent) && exponent==std::floor(exponent) && std::fmod(exponent,2.0)==1.0)
    throw std::invalid_argument("odd pdrf_exponent makes the background penalty negative");
  p.empty=false;
  const char *lockstep_setting=std::getenv("BROOK_LOCKSTEP");
  bool use_lockstep=!lockstep_setting || (std::string(lockstep_setting)!="0" && std::string(lockstep_setting)!="off" &&
    std::string(lockstep_setting)!="false" && std::string(lockstep_setting)!="no");
  // Voxel graphs and avocado-corrected volumes trace in lockstep too: the tracing graph gates the
  // batched preparation, the private soma preparations and every lockstep kernel as it gates the
  // per-component tracer, and the batched PDRF treats painted distance-zero foreground as +inf.
  if(lockstep_wanted && use_lockstep && cc.labels.size()<(size_t{1}<<31)) {
    auto preparation=prepare_batched(ctx,cc,dbf,boxes,borders,options,graph);lap("prep_fields_us");
    p.lockstep=std::make_shared<SomaPreparation>(prepare_somata(ctx,cc,dbf,std::move(preparation),boxes,borders,p.extra_before,p.extra_after,options,graph));lap("prep_somata_us");
  }
  return p;
}
std::vector<Skeleton> finish_sample(Context &ctx,PreparedSample &sample,LockstepResult &locked,const SkeletonizeOptions &options,
    PackedSkeletons *device_result) {
  auto &cc=sample.cc;auto &dbf=sample.dbf;const auto &boxes=sample.boxes;const size_t count=sample.count;
  const auto &borders=sample.borders;const auto &extra_before=sample.extra_before;const auto &extra_after=sample.extra_after;
  const Array *graph=sample.trace_graph.storage?&sample.trace_graph:nullptr;Array &trace_graph=sample.trace_graph;
  std::vector<Skeleton> results;
  std::unordered_map<int64_t,size_t> index;
  std::vector<SkeletonPart> device_parts;
  std::vector<std::pair<bool,size_t>> part_order;
  for(size_t id=1;id<=count;++id) {
    const Box &box=boxes[id];
    if(!box.count || double(box.count)<=options.dust) continue;
    std::array<int64_t,3> shape;
    for(int k=0;k<3;++k) shape[k]=box.hi[k]-box.lo[k]+1;
    size_t voxels=volume_size(shape);
    if(voxels<=1) continue;
    if(voxels>=size_t{1}<<31) throw std::invalid_argument("component bounding box exceeds int32 worklist capacity");
    if(device_result && locked.device && id<locked.handled.size() && locked.handled[id]) {
      const auto &p=*locked.device;
      int64_t nv=p.vertex_offsets[id+1]-p.vertex_offsets[id],ne=p.edge_offsets[id+1]-p.edge_offsets[id];
      if(ne) {
        part_order.push_back({true,device_parts.size()});
        device_parts.push_back({cc.mapping[id],row_view(p.vertices,p.vertex_offsets[id],nv,3),
          row_view(p.edges,p.edge_offsets[id],ne,2),row_view(p.radii,p.vertex_offsets[id],nv,1)});
      }
      continue;
    }
    Skeleton sk;
    if(id<locked.handled.size() && locked.handled[id]) {
      sk=std::move(*locked.components[id]);
    } else {
      Array graph_crop;
      auto inputs=crop_component(ctx,cc.labels,dbf,id,box,graph,&graph_crop);
      auto before=borders[id],after=extra_after[id];
      std::optional<Point> root;
      if(!before.empty()) { root=before.back();before.pop_back();for(int k=0;k<3;++k) (*root)[k]-=box.lo[k]; }
      before.insert(before.end(),extra_before[id].begin(),extra_before[id].end());
      translate(before,box.lo);translate(after,box.lo);
      sk=trace_component(ctx,std::move(inputs.first),std::move(inputs.second),options,std::move(before),std::move(after),root,
        graph?&graph_crop:nullptr);
      for(auto &vertex:sk.vertices) for(int k=0;k<3;++k) vertex[k]=(vertex[k]+float(box.lo[k]))*options.anisotropy[k];
    }
    if(sk.edges.empty()) continue;
    sk.label=cc.mapping[id];
    if(device_result) { part_order.push_back({false,results.size()});results.push_back(std::move(sk));continue; }
    auto entry=index.find(sk.label);
    if(entry==index.end()) { index[sk.label]=results.size();results.push_back(std::move(sk)); }
    else {
      auto &out=results[entry->second];size_t base=out.vertices.size();
      if(base+sk.vertices.size()>UINT32_MAX) throw std::invalid_argument("merged skeleton exceeds uint32 vertex capacity");
      out.vertices.insert(out.vertices.end(),sk.vertices.begin(),sk.vertices.end());
      out.radii.insert(out.radii.end(),sk.radii.begin(),sk.radii.end());
      for(auto edge:sk.edges) out.edges.push_back({uint32_t(base+edge[0]),uint32_t(base+edge[1])});
    }
  }
  if(device_result) {
    cc.labels={};dbf={};trace_graph={};locked.device.reset();sample.lockstep.reset();
    auto fallback=pack_skeletons(ctx,results,options.anisotropy);
    auto fallback_parts=skeleton_parts(fallback);
    std::vector<SkeletonPart> parts;parts.reserve(part_order.size());
    for(auto entry:part_order) parts.push_back(entry.first?std::move(device_parts[entry.second]):std::move(fallback_parts[entry.second]));
    *device_result=merge_parts(ctx,parts,options.anisotropy,false,true);
    return {};
  }
  for(auto &skeleton:results) consolidate(skeleton);
  return results;
}
std::vector<Skeleton> skeletonize(Context &ctx,Array labels,const SkeletonizeOptions &options,PackedSkeletons *device_result) {
  ctx.activate();
  ctx.stats.clear();ctx.stats["graphs"]=graphs_enabled();
  if(device_result) *device_result=empty_packed(ctx,options.anisotropy);
  auto sample=prepare_sample(ctx,std::move(labels),options,true);
  if(sample.empty) return {};
  LockstepResult locked;
  if(sample.lockstep) {
    auto &a=*sample.lockstep;
    locked=trace_lockstep(ctx,std::move(a.components),std::move(a.dbf),std::move(a.fields),a.layout,a.boxes,sample.borders,
      sample.extra_before,sample.extra_after,options,device_result!=nullptr);
    sample.lockstep.reset();
  }
  return finish_sample(ctx,sample,locked,options,device_result);
}
PackedSkeletons skeletonize_packed(Context &ctx,Array labels,const SkeletonizeOptions &options) {
  PackedSkeletons result;skeletonize(ctx,std::move(labels),options,&result);return result;
}
}
