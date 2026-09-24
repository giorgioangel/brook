// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "batch.hpp"
#include "graph.hpp"
#include "lockstep.hpp"
#include "soma.hpp"
#include "voxel_graph.hpp"
#include "common.cuh"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <optional>
#include <tuple>

namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
// Component ids of one sample copied into the batch's flat label volume, shifted into the sample's
// global id range; background stays 0.
template<class In,class Out> __global__ void renumber(Out *out,const In *in,unsigned long long base,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;
  In v=in[i];out[i]=v?Out(static_cast<unsigned long long>(v)+base):Out(0);
}
// Parent pointers are flat indices of the sample's own arena space: shifted by the sample's base.
__global__ void shift_parents(int *out,const int *in,int offset,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;
  int v=in[i];out[i]=v<0?v:v+offset;
}
// Seams of a stacked voxel graph. The graph CCL unites a voxel only through the bits of the HIGHER
// voxel of an edge, so of all the bits that could join two consecutive samples only the nine
// dz = -1 bits of a sample's first slice matter: they are cleared. Nothing else reads them: the
// graph EDT samples only the +x/+y/+z face bits (and runs its z pass per sample), and every flood,
// backtrace and invalidation is label-gated before it consults the graph. In a sample's own frame
// those bits point outside the volume, where the bounds check skips them: the cleared stack and
// the sample see the same edges.
__global__ void mask_seams(uint32_t *graph,long long per_slice,long long depth,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;
  if((i/per_slice)%depth!=0) return;
  unsigned mask=0;for(int k=0;k<26;++k) if(cuda::BROOK_NZ[k]<0) mask|=1u<<cuda::BROOK_BIT[k];
  graph[i]&=~mask;
}
// Every edge permitted: the words of a sample without a voxel graph inside a graph batch (a gate
// that always passes is the ungated step).
__global__ void fill_words(uint32_t *out,uint32_t value,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n) out[i]=value;
}
constexpr uint32_t all_edges=(1u<<26)-1;
template<class Out> void renumber_into(Context &ctx,Array &out,int64_t offset,const Array &in,uint64_t base) {
  Out *dst=static_cast<Out*>(out.data())+offset;const long long n=static_cast<long long>(in.size());
  if(!n) return;
  switch(in.dtype) {
    case BROOK_U8:renumber<uint8_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint8_t*>(in.data()),base,n);break;
    case BROOK_U16:renumber<uint16_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint16_t*>(in.data()),base,n);break;
    case BROOK_U32:renumber<uint32_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint32_t*>(in.data()),base,n);break;
    case BROOK_U64:renumber<uint64_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint64_t*>(in.data()),base,n);break;
    default:throw std::invalid_argument("invalid batch label dtype");
  }
  BROOK_CUDA(cudaGetLastError());
}
void copy_into(Context &ctx,Array &out,int64_t offset,const Array &in) {
  if(!in.bytes()) return;
  BROOK_CUDA(cudaMemcpyAsync(static_cast<char*>(out.data())+offset*int64_t(dtype_size(out.dtype)),in.data(),in.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
}

// Nonzero label range of a sample (the per-sample id offsets of a stacked batch).
template<class In> __global__ void label_range(const In *in,long long n,long long *lo,long long *hi) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;
  long long v=static_cast<long long>(in[i]);if(!v) return;
  atomicMin(lo,v);atomicMax(hi,v);
}
std::pair<long long,long long> label_range_of(Context &ctx,const Array &a) {
  std::vector<long long> init={INT64_MAX,INT64_MIN};Buffer<long long> r(ctx,2);r.set(ctx,init.data(),2);
  const long long n=static_cast<long long>(a.size());
  if(n) switch(a.dtype) {
    case BROOK_U8:label_range<uint8_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const uint8_t*>(a.data()),n,r.data(),r.data()+1);break;
    case BROOK_U16:label_range<uint16_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const uint16_t*>(a.data()),n,r.data(),r.data()+1);break;
    case BROOK_U32:label_range<uint32_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const uint32_t*>(a.data()),n,r.data(),r.data()+1);break;
    case BROOK_U64:label_range<uint64_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const uint64_t*>(a.data()),n,r.data(),r.data()+1);break;
    case BROOK_I8:label_range<int8_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const int8_t*>(a.data()),n,r.data(),r.data()+1);break;
    case BROOK_I16:label_range<int16_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const int16_t*>(a.data()),n,r.data(),r.data()+1);break;
    case BROOK_I32:label_range<int32_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const int32_t*>(a.data()),n,r.data(),r.data()+1);break;
    case BROOK_I64:label_range<int64_t><<<blocks(n),256,0,ctx.stream>>>(static_cast<const int64_t*>(a.data()),n,r.data(),r.data()+1);break;
    default:throw std::invalid_argument("batch samples must be integer labels");
  }
  BROOK_CUDA(cudaGetLastError());auto out=r.get(ctx,2);
  return {out[0],out[1]};
}
template<class In,class Out> __global__ void renumber_any_kernel(Out *out,const In *in,unsigned long long base,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;
  long long v=static_cast<long long>(in[i]);out[i]=v?Out(static_cast<unsigned long long>(v)+base):Out(0);
}
template<class Out> void renumber_any_out(Context &ctx,Out *dst,const Array &in,uint64_t base) {
  const long long n=static_cast<long long>(in.size());if(!n) return;
  switch(in.dtype) {
    case BROOK_U8:renumber_any_kernel<uint8_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint8_t*>(in.data()),base,n);break;
    case BROOK_U16:renumber_any_kernel<uint16_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint16_t*>(in.data()),base,n);break;
    case BROOK_U32:renumber_any_kernel<uint32_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint32_t*>(in.data()),base,n);break;
    case BROOK_U64:renumber_any_kernel<uint64_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const uint64_t*>(in.data()),base,n);break;
    case BROOK_I8:renumber_any_kernel<int8_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const int8_t*>(in.data()),base,n);break;
    case BROOK_I16:renumber_any_kernel<int16_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const int16_t*>(in.data()),base,n);break;
    case BROOK_I32:renumber_any_kernel<int32_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const int32_t*>(in.data()),base,n);break;
    case BROOK_I64:renumber_any_kernel<int64_t,Out><<<blocks(n),256,0,ctx.stream>>>(dst,static_cast<const int64_t*>(in.data()),base,n);break;
    default:throw std::invalid_argument("batch samples must be integer labels");
  }
  BROOK_CUDA(cudaGetLastError());
}
// Any integer sample (signed dtypes included) into its slab of the stack, in the stack's dtype.
void renumber_any(Context &ctx,Array &out,int64_t offset,const Array &in,uint64_t base) {
  switch(out.dtype) {
    case BROOK_U8:renumber_any_out<uint8_t>(ctx,static_cast<uint8_t*>(out.data())+offset,in,base);break;
    case BROOK_U16:renumber_any_out<uint16_t>(ctx,static_cast<uint16_t*>(out.data())+offset,in,base);break;
    case BROOK_U32:renumber_any_out<uint32_t>(ctx,static_cast<uint32_t*>(out.data())+offset,in,base);break;
    default:renumber_any_out<uint64_t>(ctx,static_cast<uint64_t*>(out.data())+offset,in,base);break;
  }
}
// One sample's manual target points in the stack's frame, in two forms. `located` is what the
// component lookup reads: the bounds check and the wrap of a negative coordinate are the
// single-volume ones applied to the sample's own shape, so a point outside the sample is rejected
// exactly as it would be in a call on that sample alone, and the shift then puts the point in its
// own slab, where it can only meet that sample's components. `kept` is the point the trace
// receives, the caller's own coordinates shifted by the same slab offset: a single call keeps
// them unwrapped too, and the component crop subtracts its box from them either way.
void stack_points(std::vector<Point> &located,std::vector<Point> &kept,const std::vector<Point> &points,
    const std::array<int64_t,3> &shape,int64_t offset) {
  for(const auto &point:points) {
    Point inside=point;
    for(int axis=0;axis<3;++axis) {
      if(inside[axis]<0) inside[axis]+=shape[axis];
      if(inside[axis]<0 || inside[axis]>=shape[axis]) throw std::invalid_argument("extra target is outside the input volume");
    }
    inside[2]+=offset;located.push_back(inside);
    Point shifted=point;shifted[2]+=offset;kept.push_back(shifted);
  }
}
// The batch's points bucketed by the stacked component each one falls on, as assign_points buckets
// a single volume's: bucket 0 collects the points on background, which nothing traces.
std::vector<std::vector<Point>> bucket_points(Context &ctx,const Array &labels,size_t count,
    const std::vector<Point> &located,const std::vector<Point> &kept) {
  std::vector<std::vector<Point>> result(count+1);
  for(size_t i=0;i<located.size();++i) result.at(size_t(component_at(ctx,labels,located[i]))).push_back(kept[i]);
  return result;
}
// The manual targets of the samples [first, last) of a batch, or an empty list when the batch has
// none (the engine reads an empty list as "no sample has targets").
std::vector<std::vector<Point>> slice_targets(const std::vector<std::vector<Point>> &targets,size_t first,size_t last) {
  if(targets.empty()) return {};
  return std::vector<std::vector<Point>>(targets.begin()+first,targets.begin()+last);
}
bool lockstep_enabled() {
  const char *v=std::getenv("BROOK_LOCKSTEP");
  return !v || (std::string(v)!="0" && std::string(v)!="off" && std::string(v)!="false" && std::string(v)!="no");
}
// Samples of one shape can share the whole preamble: stacked along z with disjoint id ranges they
// are one volume whose connected components never merge (different ids), whose holes are filled
// box by box (a component's box lies inside its sample), whose EDT runs its z lines per sample
// (segmented), whose avocado correction runs sample by sample (segmented too), whose DAF/PDRF
// floods stay inside each label, and whose border targets are found on each sample's own faces.
// One trace covers the stack. Voxel graphs stack with the samples (uint32 graphs of the samples'
// shape, one per sample: see mask_seams), and hole filling runs on the stacked graph components as it
// does per sample; a graph with avocado correction takes the generic path, because Kimimaro visits
// avocado candidates in the order of each volume's own graph ids (Components::keys).
bool uniform_eligible(const std::vector<Array> &samples,const SkeletonizeOptions &o,const std::vector<Array> &graphs) {
  if(samples.size()<2 || !lockstep_enabled()) return false;
  if(o.voxel_graph.storage || o.object_ids || o.black_border_override) return false;
  if(!graphs.empty() && o.fix_avocados) return false;
  const auto shape=samples.front().shape;
  for(const auto &s:samples) if(s.shape!=shape || !s.size()) return false;
  if(!graphs.empty()) {
    if(graphs.size()!=samples.size()) return false;
    for(const auto &g:graphs) if(!g.storage || g.dtype!=BROOK_U32 || g.shape!=shape) return false;
  }
  return true;
}
// nullopt: a sample would have been a single full-volume component (black border; with a voxel
// graph, a uniform input), which the stack cannot represent; the caller takes the generic path.
std::optional<std::vector<PackedSkeletons>> uniform_batch(Context &ctx,std::vector<Array> &samples,const SkeletonizeOptions &options,
    const std::vector<std::vector<Point>> &targets_before,const std::vector<std::vector<Point>> &targets_after,std::vector<Array> &graphs) {
  const auto shape=samples.front().shape;const int64_t n=int64_t(volume_size(shape));const size_t K=samples.size();
  const auto aniso=options.anisotropy;const bool graph=!graphs.empty();
  std::vector<PackedSkeletons> results;for(size_t s=0;s<K;++s) results.push_back(empty_packed(ctx,aniso));
  std::vector<long long> base(K),lowest(K),highest(K);long long next=0;
  for(size_t s=0;s<K;++s) {
    auto [lo,hi]=label_range_of(ctx,samples[s]);lowest[s]=lo;highest[s]=hi;
    if(hi>=lo && lo<0) throw std::invalid_argument("batch samples must not contain negative labels");
    base[s]=next;if(hi>=lo) next+=hi;
  }
  const brook_dtype dtype=next<=UINT8_MAX?BROOK_U8:next<=UINT16_MAX?BROOK_U16:next<=UINT32_MAX?BROOK_U32:BROOK_U64;
  std::array<int64_t,3> stacked={shape[0],shape[1],shape[2]*int64_t(K)};
  using clock=std::chrono::steady_clock;auto t=clock::now();
  auto lap=[&](const char *key){ auto now=clock::now();ctx.stats[key]+=std::chrono::duration_cast<std::chrono::microseconds>(now-t).count();t=now; };
  Array stack=allocate(ctx,stacked,dtype);
  for(size_t s=0;s<K;++s) renumber_any(ctx,stack,int64_t(s)*n,samples[s],uint64_t(base[s]));
  Array stacked_graph;
  if(graph) {
    stacked_graph=allocate(ctx,stacked,BROOK_U32);
    for(size_t s=0;s<K;++s) copy_into(ctx,stacked_graph,int64_t(s)*n,graphs[s]);
    mask_seams<<<blocks(stacked_graph.size()),256,0,ctx.stream>>>(static_cast<uint32_t*>(stacked_graph.data()),shape[0]*shape[1],shape[2],int64_t(stacked_graph.size()));
    BROOK_CUDA(cudaGetLastError());
  }
  ctx.synchronize();lap("uniform_stack_us");
  auto cc=graph?graph_components(ctx,stack,stacked_graph):connected_components(ctx,stack);stack={};lap("uniform_ccl_us");
  size_t count=cc.roots.size();
  if(!count) { for(auto &s:samples) s={};for(auto &g:graphs) g={};return results; }
  auto boxes=analyze(ctx,cc.labels,count);lap("uniform_analyze_us");
  std::vector<int> owner(count+1,0),per_sample(K,0);std::vector<int64_t> foreground(K,0);
  for(size_t id=1;id<=count;++id) {
    if(!boxes[id].count) continue;                            // an id without voxels owns nothing
    owner[id]=int(boxes[id].lo[2]/shape[2]);++per_sample[owner[id]];foreground[owner[id]]+=boxes[id].count;
  }
  if(graph) { for(size_t s=0;s<K;++s) if(lowest[s]<=highest[s] && lowest[s]==highest[s] && foreground[s]==n) return std::nullopt; }
  else for(size_t id=1;id<=count;++id) if(per_sample[owner[id]]==1 && boxes[id].count==n) return std::nullopt;
  for(auto &s:samples) s={};
  for(auto &g:graphs) g={};
  // The stages of prepare_sample() in its order: black_border was decided on the unfilled volumes.
  if(options.fill_holes) {
    auto filled=fill_all_holes(ctx,std::move(cc.labels),boxes);
    cc.labels=std::move(filled.first);ctx.stats["hole_filled_voxels"]=filled.second;
    if(filled.second) { boxes=analyze(ctx,cc.labels,count);refresh_roots(ctx,cc.labels,cc.roots); }
    lap("uniform_holes_us");
  }
  auto dbf=graph?graph_edt(ctx,cc.labels,stacked_graph,aniso,false,false,false,true,shape[2]):edt(ctx,cc.labels,aniso,false,3,shape[2]);lap("uniform_edt_us");
  if(options.fix_avocados) {
    // Sample by sample on the stack; the ids are renumbered, still contiguous per sample.
    fix_avocados(ctx,cc,dbf,options.avocado_detection,aniso,false,shape[2]);
    count=cc.roots.size();if(!count) return results;
    boxes=analyze(ctx,cc.labels,count);owner.assign(count+1,0);
    for(size_t id=1;id<=count;++id) owner[id]=int(boxes[id].lo[2]/shape[2]);
    lap("uniform_avocados_us");
  }
  std::vector<std::vector<Point>> borders(count+1),before(count+1),after(count+1);
  // Manual targets are bucketed by component exactly as a single volume's are, on the stack's
  // components: a sample's points can only land on its own components or on background (bucket 0,
  // which nothing traces). A sample smaller than the dust threshold never reaches this point in a
  // single call, so its targets are not looked at here either.
  if((!targets_before.empty() || !targets_after.empty()) && double(n)>options.dust) {
    std::vector<Point> located_before,located_after,kept_before,kept_after;
    for(size_t s=0;s<K;++s) {
      if(!targets_before.empty()) stack_points(located_before,kept_before,targets_before[s],shape,int64_t(s)*shape[2]);
      if(!targets_after.empty()) stack_points(located_after,kept_after,targets_after[s],shape,int64_t(s)*shape[2]);
    }
    before=bucket_points(ctx,cc.labels,count,located_before,kept_before);
    after=bucket_points(ctx,cc.labels,count,located_after,kept_after);
    lap("uniform_targets_us");
  }
  if(options.fix_borders) {
    // the face-stack plan of a group of samples: bounded by a share of free memory
    size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
    const double plan_bytes=3.0*56.0*double(shape[0]*shape[1]+shape[0]*shape[2]+shape[1]*shape[2]);
    // groups sized so the plan stays within the border cache limit (total / 8) and is reused
    const size_t group=std::max<size_t>(1,std::min<size_t>(K,size_t(std::min(0.25*double(free),double(total)/8.0)/std::max(plan_bytes,1.0))));
    for(size_t first=0;first<K;first+=group) {
      const size_t last=std::min(K,first+group);
      Array view=cc.labels;view.byte_offset+=first*size_t(n)*dtype_size(cc.labels.dtype);
      view.shape={shape[0],shape[1],shape[2]*int64_t(last-first)};view.logical_shape.reset();view.logical_strides.clear();
      auto found=border_targets_stacked(ctx,view,int(last-first),count,aniso);
      for(size_t id=1;id<=count;++id) for(auto pt:found[id]) { pt[2]+=int64_t(first)*shape[2];borders[id].push_back(pt); }
    }
  }
  lap("uniform_borders_us");
  double exponent=options.teasar.pdrf_exponent;
  if(std::isfinite(exponent) && exponent==std::floor(exponent) && std::fmod(exponent,2.0)==1.0)
    throw std::invalid_argument("odd pdrf_exponent makes the background penalty negative");
  LockstepResult locked;
  const Array *trace_graph=graph?&stacked_graph:nullptr;
  {
    auto preparation=prepare_batched(ctx,cc,dbf,boxes,borders,options,trace_graph);
    auto a=prepare_somata(ctx,cc,dbf,std::move(preparation),boxes,borders,before,after,options,trace_graph);lap("uniform_fields_us");
    a.layout.label_z.assign(count+1,0);
    for(size_t id=1;id<=count;++id) a.layout.label_z[id]=int64_t(owner[id])*shape[2];
    locked=trace_lockstep(ctx,std::move(a.components),std::move(a.dbf),std::move(a.fields),a.layout,a.boxes,borders,before,after,options,true);lap("uniform_trace_us");
  }
  // Per sample, in component id order as finish_sample keeps it: device parts of the lockstep
  // assembly, host-traced fallbacks for the labels it did not handle. Every vertex is its voxel
  // coordinate in the sample's own frame times the anisotropy, rounded once, as in a separate call:
  // the lockstep assembly subtracts the sample's first slice (layout.label_z) before scaling, and so
  // does the fallback below.
  std::vector<std::vector<Skeleton>> host(K);
  std::vector<std::vector<std::tuple<bool,size_t,int64_t>>> order(K);   // (device?, index, label)
  std::vector<std::vector<SkeletonPart>> device_parts(K);
  for(size_t id=1;id<=count;++id) {
    const Box &box=boxes[id];
    if(!box.count || double(box.count)<=options.dust) continue;
    std::array<int64_t,3> extent;for(int k=0;k<3;++k) extent[k]=box.hi[k]-box.lo[k]+1;
    const size_t voxels=volume_size(extent);
    if(voxels<=1) continue;
    if(voxels>=size_t{1}<<31) throw std::invalid_argument("component bounding box exceeds int32 worklist capacity");
    const int s=owner[id];
    const int64_t label=cc.mapping[id]-base[s];
    const bool handled=id<locked.handled.size() && locked.handled[id];
    if(handled && locked.device) {
      const auto &p=*locked.device;
      int64_t nv=p.vertex_offsets[id+1]-p.vertex_offsets[id],ne=p.edge_offsets[id+1]-p.edge_offsets[id];
      if(ne) {
        order[s].push_back({true,device_parts[s].size(),label});
        device_parts[s].push_back({label,row_view(p.vertices,p.vertex_offsets[id],nv,3),row_view(p.edges,p.edge_offsets[id],ne,2),
          row_view(p.radii,p.vertex_offsets[id],nv,1),{0,0,0}});
      }
      continue;
    }
    Skeleton sk;
    if(handled) sk=std::move(*locked.components[id]);
    else {
      Array graph_crop;
      auto inputs=crop_component(ctx,cc.labels,dbf,id,box,trace_graph,trace_graph?&graph_crop:nullptr);
      auto manual=borders[id],extra=after[id];std::optional<Point> root;
      if(!manual.empty()) { root=manual.back();manual.pop_back();for(int k=0;k<3;++k) (*root)[k]-=box.lo[k]; }
      manual.insert(manual.end(),before[id].begin(),before[id].end());
      for(auto &point:manual) for(int k=0;k<3;++k) point[k]-=box.lo[k];
      for(auto &point:extra) for(int k=0;k<3;++k) point[k]-=box.lo[k];
      sk=trace_component(ctx,std::move(inputs.first),std::move(inputs.second),options,std::move(manual),std::move(extra),root,trace_graph?&graph_crop:nullptr);
      const Point lo={box.lo[0],box.lo[1],box.lo[2]-int64_t(s)*shape[2]};   // the box in the sample's frame
      for(auto &vertex:sk.vertices) for(int k=0;k<3;++k) vertex[k]=(vertex[k]+float(lo[k]))*aniso[k];
    }
    if(sk.edges.empty()) continue;
    sk.label=label;order[s].push_back({false,host[s].size(),label});host[s].push_back(std::move(sk));
  }
  cc.labels={};dbf={};stacked_graph={};locked.device.reset();
  for(size_t s=0;s<K;++s) {
    if(order[s].empty()) continue;
    auto fallback=pack_skeletons(ctx,host[s],aniso);auto fallback_parts=skeleton_parts(fallback);
    std::vector<SkeletonPart> parts;parts.reserve(order[s].size());
    for(auto &[on_device,index,label]:order[s]) parts.push_back(on_device?std::move(device_parts[s][index]):std::move(fallback_parts[index]));
    results[s]=merge_parts(ctx,parts,aniso,false,true);
  }
  lap("uniform_finish_us");
  return results;
}
// The prepared samples of a sub-batch as one lockstep input: one flat allocation per field, one
// arena per sample (plus the samples' private soma arenas), one global label table. Nothing of one
// sample is adjacent to, or addressable from, another sample: arenas are disjoint index ranges.
struct Merged {
  Components cc;Array dbf;BatchedPreparation fields;ArenaLayout layout;std::vector<Box> boxes;
  std::vector<std::vector<Point>> borders,before,after;
  std::vector<int64_t> label_base,label_count;   // sample j: global id = label_base[j] + local id, local ids 1..label_count[j]-1
};
Merged concatenate(Context &ctx,std::vector<PreparedSample> &samples,bool fix) {
  Merged m;uint64_t K=1;int64_t N=0;bool graph=false;
  for(auto &p:samples) { auto &a=*p.lockstep;K+=a.fields.active.size()-1;N+=a.layout.offsets.back();graph=graph || a.fields.graph.storage; }
  brook_dtype dtype=K<=UINT8_MAX?BROOK_U8:K<=UINT16_MAX?BROOK_U16:K<=UINT32_MAX?BROOK_U32:BROOK_U64;
  std::array<int64_t,3> shape={N,1,1};
  m.cc.labels=allocate(ctx,shape,dtype);m.dbf=allocate(ctx,shape,BROOK_F32);m.fields.daf=allocate(ctx,shape,BROOK_F32);
  if(fix) m.fields.pdrf=allocate(ctx,shape,BROOK_F32);else m.fields.parents=allocate(ctx,shape,BROOK_I32);
  // Graph and graph-less samples trace together: the latter's words permit every edge.
  if(graph) m.fields.graph=allocate(ctx,shape,BROOK_U32);
  m.cc.mapping={0};m.fields.active={0};m.fields.soma={0};m.fields.dbf_max={0};m.fields.max_daf={0};m.fields.root={-1};m.fields.target={-1};m.fields.filled={0};
  m.layout.offsets={0};m.layout.label_arena={0};m.boxes={Box{}};m.borders={{}};m.before={{}};m.after={{}};
  int64_t voxel_base=0;uint64_t label_base=0;int arena_base=0;
  for(auto &p:samples) {
    auto &a=*p.lockstep;const size_t k=a.fields.active.size();const int64_t n=a.layout.offsets.back();
    m.label_base.push_back(int64_t(label_base));m.label_count.push_back(int64_t(k));
    switch(dtype) {
      case BROOK_U8:renumber_into<uint8_t>(ctx,m.cc.labels,voxel_base,a.components.labels,label_base);break;
      case BROOK_U16:renumber_into<uint16_t>(ctx,m.cc.labels,voxel_base,a.components.labels,label_base);break;
      case BROOK_U32:renumber_into<uint32_t>(ctx,m.cc.labels,voxel_base,a.components.labels,label_base);break;
      default:renumber_into<uint64_t>(ctx,m.cc.labels,voxel_base,a.components.labels,label_base);break;
    }
    copy_into(ctx,m.dbf,voxel_base,a.dbf);copy_into(ctx,m.fields.daf,voxel_base,a.fields.daf);
    if(graph) {
      if(a.fields.graph.storage) {
        if(a.fields.graph.size()!=size_t(n) || a.fields.graph.dtype!=BROOK_U32) throw std::logic_error("prepared sample graph does not cover its arenas");
        copy_into(ctx,m.fields.graph,voxel_base,a.fields.graph);
      } else if(n) {
        fill_words<<<blocks(n),256,0,ctx.stream>>>(static_cast<uint32_t*>(m.fields.graph.data())+voxel_base,all_edges,n);
        BROOK_CUDA(cudaGetLastError());
      }
    }
    if(fix) copy_into(ctx,m.fields.pdrf,voxel_base,a.fields.pdrf);
    else if(n) {
      shift_parents<<<blocks(n),256,0,ctx.stream>>>(static_cast<int*>(m.fields.parents.data())+voxel_base,static_cast<const int*>(a.fields.parents.data()),int(voxel_base),n);
      BROOK_CUDA(cudaGetLastError());
    }
    for(size_t id=1;id<k;++id) {
      m.cc.mapping.push_back(a.components.mapping[id]);
      m.fields.active.push_back(a.fields.active[id]);m.fields.soma.push_back(a.fields.soma[id]);
      m.fields.dbf_max.push_back(a.fields.dbf_max[id]);m.fields.max_daf.push_back(a.fields.max_daf[id]);m.fields.filled.push_back(a.fields.filled[id]);
      m.fields.root.push_back(a.fields.root[id]<0?-1:a.fields.root[id]+voxel_base);
      m.fields.target.push_back(a.fields.target[id]<0?-1:a.fields.target[id]+voxel_base);
      m.layout.label_arena.push_back(a.layout.label_arena[id]+arena_base);
      m.boxes.push_back(a.boxes[id]);m.borders.push_back(p.borders[id]);m.before.push_back(p.extra_before[id]);m.after.push_back(p.extra_after[id]);
    }
    for(size_t id=0;id+1<k && id<a.components.roots.size();++id) m.cc.roots.push_back(a.components.roots[id]+voxel_base);
    for(size_t ar=0;ar<a.layout.dimensions.size();++ar) {
      m.layout.dimensions.push_back(a.layout.dimensions[ar]);m.layout.origins.push_back(a.layout.origins[ar]);
      m.layout.offsets.push_back(a.layout.offsets[ar+1]+voxel_base);m.layout.base.push_back(arena_base);
    }
    voxel_base+=n;label_base+=k-1;arena_base+=int(a.layout.dimensions.size());
    a.components.labels={};a.dbf={};a.fields.daf={};a.fields.pdrf={};a.fields.parents={};a.fields.graph={};   // copied: release the originals
  }
  ctx.synchronize();
  return m;
}
// The batch's lockstep result restricted to one sample: the same tables, indexed by local id.
LockstepResult view_of(LockstepResult &locked,int64_t label_base,int64_t k) {
  LockstepResult v;v.handled.assign(k,0);v.components.resize(k);
  v.iterations=locked.iterations;v.paths=locked.paths;v.ejected=locked.ejected;
  for(int64_t i=1;i<k;++i) {
    if(size_t(label_base+i)<locked.handled.size()) v.handled[i]=locked.handled[label_base+i];
    if(size_t(label_base+i)<locked.components.size() && locked.components[label_base+i]) v.components[i]=std::move(locked.components[label_base+i]);
  }
  if(locked.device) {
    ComponentAssembly c;c.vertices=locked.device->vertices;c.edges=locked.device->edges;c.radii=locked.device->radii;
    const auto &vo=locked.device->vertex_offsets,&eo=locked.device->edge_offsets;
    c.vertex_offsets.assign(vo.begin()+label_base,vo.begin()+label_base+k+1);
    c.edge_offsets.assign(eo.begin()+label_base,eo.begin()+label_base+k+1);
    v.device=std::move(c);
  }
  return v;
}
// Packed results of the samples merged into one packed array set, in sample order. The merge is
// keyed by a running skeleton index so equal labels of different samples never coalesce.
BatchedSkeletons combine(Context &ctx,std::vector<PackedSkeletons> results,std::array<float,3> anisotropy) {
  BatchedSkeletons out;std::vector<SkeletonPart> parts;
  for(auto &packed:results) {
    for(size_t i=0;i<packed.labels.size();++i) {
      auto nv=packed.vertex_offsets[i+1]-packed.vertex_offsets[i],ne=packed.edge_offsets[i+1]-packed.edge_offsets[i];
      if(!nv) continue;
      parts.push_back({int64_t(parts.size()),row_view(packed.vertices,packed.vertex_offsets[i],nv,3),
        row_view(packed.edges,packed.edge_offsets[i],ne,2),row_view(packed.radii,packed.vertex_offsets[i],nv,1),{0,0,0}});
      out.labels.push_back(packed.labels[i]);out.sample.push_back(int64_t(out.sample_offsets.size())-1);
    }
    out.sample_offsets.push_back(int64_t(out.labels.size()));
  }
  out.packed=merge_parts(ctx,parts,anisotropy,false,true);
  if(out.packed.labels.size()!=out.labels.size()) throw std::runtime_error("batch merge lost skeletons");
  return out;
}
}
BatchedSkeletons skeletonize_batch(Context &ctx,std::vector<Array> samples,const SkeletonizeOptions &options,
    const std::vector<std::vector<Point>> &targets_before,const std::vector<std::vector<Point>> &targets_after,std::vector<Array> graphs) {
  ctx.activate();ctx.stats.clear();ctx.stats["graphs"]=graphs_enabled();
  // A batch's manual targets and voxel graphs belong to one sample each: the whole-volume options
  // would apply one list of points, or one graph, to every sample, which is never what a batch means.
  if(!options.targets_before.empty() || !options.targets_after.empty())
    throw std::invalid_argument("batch manual targets are per sample: pass them as skeletonize_batch arguments");
  if((!targets_before.empty() && targets_before.size()!=samples.size()) ||
     (!targets_after.empty() && targets_after.size()!=samples.size()))
    throw std::invalid_argument("batch manual targets need one list of points per sample");
  if(!graphs.empty() && graphs.size()!=samples.size()) throw std::invalid_argument("one voxel graph per batch sample is required");
  if(options.voxel_graph.storage) throw std::invalid_argument("batch voxel graphs are given per sample");
  std::vector<PackedSkeletons> results;results.reserve(samples.size());
  for(size_t i=0;i<samples.size();++i) results.push_back(empty_packed(ctx,options.anisotropy));
  size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
  const char *budget_setting=std::getenv("BROOK_BATCH_BUDGET");   // bytes; tests use it to force sub-batching
  const double budget=budget_setting?std::atof(budget_setting):0.5*double(free);
  const double bytes_per_voxel=graphs.empty()?96:104;              // fields, lockstep working set, flat copies (+ the graph)
  size_t i=0;int sub_batches=0;
  const char *uniform_setting=std::getenv("BROOK_BATCH_UNIFORM");
  if(uniform_eligible(samples,options,graphs) && (!uniform_setting || std::string(uniform_setting)!="0")) {
    const double per_sample=double(samples.front().size())*bytes_per_voxel;   // borders are grouped separately
    const size_t chunk=std::max<size_t>(1,std::min<size_t>(samples.size(),size_t(budget/std::max(per_sample,1.0))));
    const int64_t cap_voxels=(int64_t{1}<<31)/std::max<int64_t>(1,int64_t(samples.front().size()));
    const size_t step=std::max<size_t>(1,std::min<size_t>(chunk,size_t(std::max<int64_t>(1,cap_voxels))));
    using clock=std::chrono::steady_clock;auto t0=clock::now();
    bool all_uniform=true;size_t resume=0;
    for(size_t first=0;first<samples.size() && all_uniform;first+=step) {
      const size_t last=std::min(samples.size(),first+step);
      std::vector<Array> chunk_samples(samples.begin()+first,samples.begin()+last),chunk_graphs;
      if(!graphs.empty()) chunk_graphs.assign(graphs.begin()+first,graphs.begin()+last);
      auto packed=uniform_batch(ctx,chunk_samples,options,slice_targets(targets_before,first,last),slice_targets(targets_after,first,last),chunk_graphs);
      if(!packed) { all_uniform=false;resume=first;break; }
      for(size_t j=0;j<packed->size();++j) results[first+j]=std::move((*packed)[j]);
      for(size_t j=first;j<last;++j) { samples[j]={};if(!graphs.empty()) graphs[j]={}; }
      ++sub_batches;
    }
    if(all_uniform) {
      ctx.stats["batch_uniform"]=1;ctx.stats["batch_sub_batches"]=sub_batches;
      ctx.stats["batch_uniform_us"]=std::chrono::duration_cast<std::chrono::microseconds>(clock::now()-t0).count();
      return combine(ctx,std::move(results),options.anisotropy);
    }
    // a sample was a single full-volume component: the generic path handles the rest
    i=resume;
  }
  using clock=std::chrono::steady_clock;auto elapsed=[](clock::time_point t){return std::chrono::duration_cast<std::chrono::microseconds>(clock::now()-t).count();};
  int64_t prepare_us=0,concat_us=0,trace_us=0,finish_us=0;
  while(i<samples.size()) {
    std::vector<PreparedSample> prepared;std::vector<size_t> members;double used=0;int64_t voxels=0;
    for(;i<samples.size();++i) {
      const double estimate=double(samples[i].size())*bytes_per_voxel;
      if(!prepared.empty() && (used+estimate>budget || voxels+int64_t(samples[i].size())>=(int64_t{1}<<31))) break;
      auto t=clock::now();
      SkeletonizeOptions sample_options=options;                       // this sample's own targets and graph
      if(!targets_before.empty()) sample_options.targets_before=targets_before[i];
      if(!targets_after.empty()) sample_options.targets_after=targets_after[i];
      if(!graphs.empty()) { sample_options.voxel_graph=std::move(graphs[i]);graphs[i]={}; }
      auto p=prepare_sample(ctx,std::move(samples[i]),sample_options,true);sample_options.voxel_graph={};prepare_us+=elapsed(t);
      if(p.empty) continue;
      if(!p.lockstep) {                                    // no lockstep inputs: this sample traces alone
        LockstepResult none;finish_sample(ctx,p,none,options,&results[i]);continue;
      }
      used+=estimate;voxels+=int64_t(p.lockstep->layout.offsets.back());
      prepared.push_back(std::move(p));members.push_back(i);
    }
    if(prepared.empty()) continue;
    ++sub_batches;
    LockstepResult locked;std::vector<int64_t> label_base,label_count;
    {
      auto t=clock::now();auto merged=concatenate(ctx,prepared,options.fix_branching);concat_us+=elapsed(t);
      label_base=merged.label_base;label_count=merged.label_count;
      t=clock::now();locked=trace_lockstep(ctx,std::move(merged.cc),std::move(merged.dbf),std::move(merged.fields),merged.layout,merged.boxes,
        merged.borders,merged.before,merged.after,options,true);trace_us+=elapsed(t);
    }
    auto t=clock::now();
    for(size_t j=0;j<prepared.size();++j) {
      auto view=view_of(locked,label_base[j],label_count[j]);
      prepared[j].lockstep.reset();
      finish_sample(ctx,prepared[j],view,options,&results[members[j]]);
    }
    finish_us+=elapsed(t);
  }
  ctx.stats["batch_sub_batches"]=sub_batches;
  ctx.stats["batch_prepare_us"]=prepare_us;ctx.stats["batch_concat_us"]=concat_us;ctx.stats["batch_trace_us"]=trace_us;ctx.stats["batch_finish_us"]=finish_us;
  return combine(ctx,std::move(results),options.anisotropy);
}
PackedSkeletons packed_slice(const BatchedSkeletons &batch,int64_t first,int64_t last) {
  const auto &p=batch.packed;int64_t count=int64_t(p.labels.size());
  if(first<0 || last<first || last>count) throw std::invalid_argument("batch slice out of range");
  PackedSkeletons out;out.anisotropy=p.anisotropy;
  int64_t v0=p.vertex_offsets[first],v1=p.vertex_offsets[last],e0=p.edge_offsets[first],e1=p.edge_offsets[last];
  out.labels.assign(batch.labels.begin()+first,batch.labels.begin()+last);
  out.vertex_offsets.clear();out.edge_offsets.clear();
  for(int64_t i=first;i<=last;++i) { out.vertex_offsets.push_back(p.vertex_offsets[i]-v0);out.edge_offsets.push_back(p.edge_offsets[i]-e0); }
  out.vertices=row_view(p.vertices,v0,v1-v0,3);out.edges=row_view(p.edges,e0,e1-e0,2);out.radii=row_view(p.radii,v0,v1-v0,1);
  return out;
}
}
