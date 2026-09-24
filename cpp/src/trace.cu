// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "skeleton.hpp"
#include "trace_primitives.hpp"
#include "lockstep_order.hpp"
#include "voxel_graph.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <type_traits>

namespace brook {
namespace {
template<class T> std::vector<T> host(const Array &a) {
  std::vector<T> result(a.size());a.copy_to_host(result.data(),result.size()*sizeof(T));return result;
}
__global__ void normalize(float *dbf,float *daf,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { if(dbf[i]==0) dbf[i]=__int_as_float(0x7f800000);if(isinf(daf[i])) daf[i]=0; }
}
__global__ void rail_scatter(float *field,const long long *path,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) field[path[i]]=0;
}
__global__ void next_target(const uint8_t *mask,const int *order,int count,int *cursor,long long *out) {
  // Every thread keeps the scan position in a register. A shared position that thread 0 advances
  // can be loaded together with `best` in one 64-bit load, which racecheck reports as a hazard.
  __shared__ int best;
  int start=*cursor;
  if(threadIdx.x==0) { best=INT32_MAX;*out=-1; }
  __syncthreads();
  while(start<count) {
    int pos=start+threadIdx.x;
    if(pos<count && mask[order[pos]]) atomicMin(&best,pos);
    __syncthreads();
    if(best!=INT32_MAX) break;
    start+=blockDim.x;
    __syncthreads();
  }
  if(threadIdx.x==0) { *cursor=best==INT32_MAX?count:best;if(best!=INT32_MAX) *out=order[best]; }
}
__global__ void gather_radius(const float *dbf,const long long *loc,float *radius,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) radius[i]=dbf[loc[i]];
}
void mark_rails(Context &ctx,Array &field,const std::vector<int64_t> &path,const long long *device_path=nullptr) {
  if(path.empty()) return;
  std::vector<long long> p;Buffer<long long> seeds;
  if(!device_path) {
    p.assign(path.begin(),path.end());seeds=Buffer<long long>(ctx,p.size());seeds.set(ctx,p.data(),p.size());
    device_path=seeds.data();
  }
  rail_scatter<<<static_cast<unsigned>((path.size()+255)/256),256,0,ctx.stream>>>(static_cast<float*>(field.data()),device_path,path.size());
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();
}
Point soma_root_points(std::vector<Point> points) {
  if(points.empty()) throw std::invalid_argument("no soma maximum");
  Point sum={0,0,0};
  for(auto p:points) for(int k=0;k<3;++k) sum[k]+=p[k];
  std::sort(points.begin(),points.end());
  std::array<float,3> center;
  for(int k=0;k<3;++k) center[k]=static_cast<float>(double(sum[k])/double(points.size()));
  double best=INFINITY;Point root=points.front();
  for(auto p:points) {
    double cost=0;for(int k=0;k<3;++k) { double delta=double(p[k])-center[k];cost+=delta*delta; }
    if(cost<best) { best=cost;root=p; }
  }
  return root;
}
Point soma_root(const std::vector<float> &dbf,float maximum,const std::array<int64_t,3> &shape) {
  std::vector<Point> points;
  for(size_t i=0;i<dbf.size();++i) if(dbf[i]==maximum) points.push_back(unravel(i,shape));
  return soma_root_points(std::move(points));
}
}

ComponentPreparation prepare_component(Context &ctx,Array mask,Array dbf,const SkeletonizeOptions &options,
                         std::vector<Point> before,std::vector<Point> after,std::optional<Point> root,const Array *voxel_graph,
                         const std::pair<Array,int> *void_check) {
  ctx.activate();
  const auto &params=options.teasar;
  const auto *step_spacing=options.trace_step_spacing?&*options.trace_step_spacing:nullptr;
  const auto shape=mask.shape;
  const int n=static_cast<int>(mask.size());
  ComponentPreparation result;
  const char *reductions=std::getenv("BROOK_COMPONENT_REDUCTIONS");
  bool device_reductions=!reductions || std::string(reductions)!="0";
  std::vector<uint8_t> labels;std::vector<float> distances;
  ComponentSummary summary{};
  if(device_reductions) { summary=summarize_component(ctx,mask,dbf);++ctx.stats["component_reductions"]; }
  else {
    labels=host<uint8_t>(mask);distances=host<float>(dbf);
    summary.valid=std::count_if(labels.begin(),labels.end(),[](auto x){return x!=0;});
    summary.first=int(std::find_if(labels.begin(),labels.end(),[](auto v){return v!=0;})-labels.begin());
    if(!distances.empty()) summary.maximum=*std::max_element(distances.begin(),distances.end());
  }
  int64_t valid=summary.valid;
  if(!valid && !(options.component_api && root)) return result;
  float maximum=summary.maximum;
  bool soma=false;
  if(maximum>params.soma_detection) {
    auto filled=void_check && (void_check->second==0 || void_check->first.storage)
      ?*void_check:fill_voids(ctx,mask);
    result.filled=filled.second;
    if(filled.second) {
      mask=std::move(filled.first);valid+=filled.second;
      // Kimimaro recomputes this field with the voxel graph (filled voxels keep their bits).
      dbf=voxel_graph?graph_edt(ctx,mask,*voxel_graph,options.anisotropy,valid==n):edt(ctx,mask,options.anisotropy,valid==n);
      if(device_reductions) { summary=summarize_component(ctx,mask,dbf);maximum=summary.maximum; }
      else {
        labels=host<uint8_t>(mask);distances=host<float>(dbf);
        maximum=*std::max_element(distances.begin(),distances.end());
        summary.first=int(std::find_if(labels.begin(),labels.end(),[](auto v){return v!=0;})-labels.begin());
      }
    }
    soma=maximum>params.soma_acceptance;
  }
  double soma_radius=0;
  if(soma) {
    if(root) before.insert(before.begin(),*root);
    if(device_reductions) {
      auto positions=maximum_positions(ctx,dbf,maximum);std::vector<Point> points;points.reserve(positions.size());
      for(int position:positions) points.push_back(unravel(position,shape));
      root=soma_root_points(std::move(points));
    } else root=soma_root(distances,maximum,shape);
    soma_radius=double(maximum)*params.soma_scale+params.soma_constant;
  } else if(!root) {
    auto first=summary.first;
    auto arbitrary=euclidean_distance_field(ctx,mask,first,options.anisotropy,0,voxel_graph,false,0,step_spacing);
    root=unravel(arbitrary.maximum,shape);
  }
  int64_t root_flat=flatten(*root,shape);
  auto daf=euclidean_distance_field(ctx,mask,root_flat,options.anisotropy,soma?maximum:0,voxel_graph,false,0,step_spacing);
  Point target=unravel(daf.maximum,shape);
  float farthest=0;
  BROOK_CUDA(cudaMemcpyAsync(&farthest,static_cast<const float*>(daf.distance.data())+daf.maximum,sizeof(float),cudaMemcpyDeviceToHost,ctx.stream));
  ctx.synchronize();double max_daf=farthest;
  const char *fusion=std::getenv("BROOK_NORMALIZE_PDRF");
  Array penalty;
  if(options.component_api && maximum==0) throw TraceZeroDivision();
  if(fusion && std::string(fusion)=="1")
    penalty=normalize_pdrf(ctx,dbf,daf.distance,maximum,max_daf,params.pdrf_scale,params.pdrf_exponent);
  else {
    normalize<<<static_cast<unsigned>((n+255)/256),256,0,ctx.stream>>>(static_cast<float*>(dbf.data()),static_cast<float*>(daf.distance.data()),n);
    BROOK_CUDA(cudaGetLastError());
    penalty=pdrf(ctx,dbf,daf.distance,maximum,max_daf,params.pdrf_scale,params.pdrf_exponent);
  }
  Array parents;
  if(!options.fix_branching) {
    const char *bands=std::getenv("BROOK_PARENT_BANDS");
    bool banded=!voxel_graph && (!bands || std::string(bands)!="0");
    if(banded) {
      size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
      // Four int work arrays, distance, int64 parents, and lane snapshots.
      double needed=28.0*mask.size()+12.0*std::min(mask.size(),size_t{1}<<18);
      banded=needed<=0.85*double(free);
    }
    auto tree=banded?parental_field_banded(ctx,mask,penalty,root_flat):parental_field(ctx,penalty,root_flat,false,voxel_graph);
    if(banded) ++ctx.stats["banded_component_parents"];
    parents=std::move(tree.second);penalty={};
  }
  if(!soma && before.empty()) before.push_back(target);
  result.mask=std::move(mask);result.dbf=std::move(dbf);result.daf=std::move(daf.distance);
  result.penalty=std::move(penalty);result.parents=std::move(parents);
  result.root=*root;result.target=target;result.before=std::move(before);result.after=std::move(after);
  result.valid=valid;result.soma=soma;result.soma_radius=soma_radius;result.maximum=maximum;
  return result;
}

Skeleton trace_component(Context &ctx,Array mask,Array dbf,const SkeletonizeOptions &options,
                         std::vector<Point> before,std::vector<Point> after,std::optional<Point> given_root,const Array *voxel_graph,
                         const std::pair<Array,int> *void_check,bool *assembled,VoxelSkeleton *voxel_result) {
  if(assembled) *assembled=false;
  auto preparation=prepare_component(ctx,std::move(mask),std::move(dbf),options,std::move(before),std::move(after),given_root,voxel_graph,void_check);
  Skeleton result;result.anisotropy=options.anisotropy;
  if(!preparation.mask.storage) return result;
  mask=std::move(preparation.mask);dbf=std::move(preparation.dbf);
  Array penalty=std::move(preparation.penalty),parents=std::move(preparation.parents);
  auto shape=mask.shape;int n=int(mask.size());
  const auto &params=options.teasar;
  Point root=preparation.root;int64_t root_flat=flatten(root,shape),valid=preparation.valid;
  bool soma=preparation.soma;double soma_radius=preparation.soma_radius;
  before=std::move(preparation.before);after=std::move(preparation.after);
  Buffer<int> device_order;
  size_t order_count=size_t(valid);
  std::vector<uint8_t> labels;std::vector<float> host_daf;std::vector<int> order;
  const char *order_mode=std::getenv("BROOK_COMPONENT_ORDER_GPU");
  if(!order_mode || std::string(order_mode)!="0") {
    device_order=build_component_target_order(ctx,mask,preparation.daf,order_count);
  } else {
    labels=host<uint8_t>(mask);host_daf=host<float>(preparation.daf);
    for(int i=0;i<n;++i) if(labels[i]) order.push_back(i);
    std::sort(order.begin(),order.end(),[&](int a,int b) {
      float da=std::isinf(host_daf[a])?0:host_daf[a],db=std::isinf(host_daf[b])?0:host_daf[b];
      return da==db?a>b:da>db;
    });
    order_count=order.size();device_order=Buffer<int>(ctx,order_count);
    device_order.set(ctx,order.data(),order_count);
  }
  Buffer<int> cursor(ctx,1);cursor.clear(ctx);
  Buffer<long long> picked(ctx,1);
  std::vector<int>().swap(order);std::vector<uint8_t>().swap(labels);std::vector<float>().swap(host_daf);
  preparation.daf={};
  TraceWorkspace ws(ctx,n,options.fix_branching);
  if(soma) valid-=invalidate(ctx,mask,dbf,float(params.soma_scale),float(params.soma_constant),options.anisotropy,{root_flat},ws,voxel_graph);
  int64_t max_paths=params.max_paths.value_or(valid);
  if(static_cast<int64_t>(before.size()+after.size())>=max_paths) return result;
  if(options.fix_branching) mark_rails(ctx,penalty,{root_flat});
  std::vector<std::vector<int64_t>> paths;
  while((valid>0 || !before.empty() || !after.empty()) && static_cast<int64_t>(paths.size())<max_paths) {
    int64_t flat=-1;bool ordered=false;
    if(!before.empty()) { flat=flatten(before.back(),shape);before.pop_back(); }
    else if(valid==0) { flat=flatten(after.back(),shape);after.pop_back(); }
    else {
      next_target<<<1,256,0,ctx.stream>>>(static_cast<const uint8_t*>(mask.data()),device_order.data(),order_count,cursor.data(),picked.data());
      BROOK_CUDA(cudaGetLastError());flat=picked.get(ctx,1)[0];ordered=true;
    }
    if(flat<0) break;
    auto path=options.fix_branching?railroad(ctx,penalty,flat,ws,voxel_graph):backtrace_workspace(ctx,parents,flat,root_flat,ws);
    // A target of the DAF order that no path reaches (a voxel graph can cut a part of the component
    // off, and so can a cavity the avocado stage painted without recomputing the field, whose zero
    // distance is an infinite penalty): nothing is invalidated or railed and next_target returns it
    // again, so every later iteration repeats this one until max_paths, as in Kimimaro. Stopping
    // keeps the same paths.
    if(ordered && path.empty()) { ++ctx.stats["component_unreachable_stops"];break; }
    if(soma && !path.empty()) {
      // Railroad paths are target-first: a rail end within the radius is trimmed, so the root
      // takes its place (root-first paths keep it as path.front()).
      std::vector<int64_t> filtered={path.front()};bool far=true;
      for(auto index:path) {
        auto p=unravel(index,shape);double squared=0;
        for(int k=0;k<3;++k) {
          double delta=double(options.anisotropy[k])*(double(p[k])-double(float(root[k])));
          squared+=delta*delta;
        }
        far=std::sqrt(squared)>soma_radius;
        if(far) filtered.push_back(index);
      }
      if(options.fix_branching && !far) filtered.push_back(root_flat);
      path=std::move(filtered);
    }
    const long long *device_path=nullptr;
    if(ws.reuse_path && !path.empty()) {ws.stage_path(ctx,path);device_path=ws.path_seeds.data();}
    if(valid>0 && !path.empty()) valid-=invalidate_device_path(ctx,mask,dbf,float(params.scale),float(params.constant),options.anisotropy,path,ws,voxel_graph,device_path);
    if(options.fix_branching) mark_rails(ctx,penalty,path,device_path);
    paths.push_back(std::move(path));
  }
  if(assembled) *assembled=std::any_of(paths.begin(),paths.end(),[](const auto &path){return !path.empty();});
  auto assemble=[&](auto &assembled_result) {
  for(const auto &path:paths) {
    if(path.size()<2) continue;
    auto base=assembled_result.vertices.size();
    if(base+path.size()>UINT32_MAX) throw std::invalid_argument("skeleton exceeds uint32 vertex capacity");
    for(size_t j=0;j<path.size();++j) {
      auto p=unravel(path[j],shape);
      using Coordinate=typename std::decay_t<decltype(assembled_result.vertices)>::value_type::value_type;
      assembled_result.vertices.push_back({Coordinate(p[0]),Coordinate(p[1]),Coordinate(p[2])});assembled_result.radii.push_back(0);
      if(j) assembled_result.edges.push_back({uint32_t(base+j-1),uint32_t(base+j)});
    }
  }
  consolidate(assembled_result);
  if(!assembled_result.vertices.empty()) {
    std::vector<long long> locations;
    for(auto v:assembled_result.vertices) locations.push_back(flatten({int64_t(v[0]),int64_t(v[1]),int64_t(v[2])},shape));
    Buffer<long long> loc(ctx,locations.size());loc.set(ctx,locations.data(),locations.size());
    Buffer<float> radii(ctx,locations.size());
    gather_radius<<<static_cast<unsigned>((locations.size()+255)/256),256,0,ctx.stream>>>(
      static_cast<const float*>(dbf.data()),loc.data(),radii.data(),locations.size());
    BROOK_CUDA(cudaGetLastError());assembled_result.radii=radii.get(ctx,locations.size());
  }
  };
  if(voxel_result) {assemble(*voxel_result);result.radii=voxel_result->radii;}
  else assemble(result);
  return result;
}
}
