// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "soma.hpp"
#include <algorithm>
#include <cstdlib>

namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
Array clone(Context &ctx,const Array &a) {
  auto result=allocate(ctx,a.shape,a.dtype);
  if(a.bytes()) BROOK_CUDA(cudaMemcpyAsync(result.data(),a.data(),a.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
  ctx.synchronize();return result;
}
template<class Label> __global__ void ownership(const Label *original,const Label *current,const uint8_t *filled,
    long long vx,long long vy,long long sx,long long sy,long long sz,long long ox,long long oy,long long oz,int label,int *taken) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=sx*sy*sz || !filled[i]) return;
  long long z=i/(sx*sy),r=i-z*sx*sy,y=r/sx,x=r-y*sx;
  long long v=(x+ox)+vx*((y+oy)+vy*(z+oz));
  if(original[v]!=Label(label) && current[v]) atomicAdd(taken+current[v],1);
}
template<class Label> __global__ void overlay(Label *labels,float *dbf,float *daf,float *penalty,int *parent,
    const uint8_t *mask,const float *local_dbf,const float *local_daf,const float *local_penalty,const long long *local_parent,
    long long vx,long long vy,long long sx,long long sy,long long sz,long long ox,long long oy,long long oz,int label,bool relabel) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=sx*sy*sz || !mask[i]) return;
  long long z=i/(sx*sy),r=i-z*sx*sy,y=r/sx,x=r-y*sx;
  long long v=(x+ox)+vx*((y+oy)+vy*(z+oz));
  if(relabel) { labels[v]=Label(label);dbf[v]=local_dbf[i]; }
  daf[v]=local_daf[i];
  if(penalty) penalty[v]=local_penalty[i];
  else {
    long long p=local_parent[i];
    if(p<0) parent[v]=-1;
    else { long long pz=p/(sx*sy),pr=p-pz*sx*sy,py=pr/sx,px=pr-py*sx;parent[v]=int((px+ox)+vx*((py+oy)+vy*(pz+oz))); }
  }
}
template<class Label> void query(Context &ctx,const Array &original,const Array &current,const ComponentPreparation &p,const Box &box,int label,Buffer<int> &taken) {
  auto s=p.mask.shape;auto o=box.lo;
  ownership<<<blocks(p.mask.size()),256,0,ctx.stream>>>(static_cast<const Label*>(original.data()),static_cast<const Label*>(current.data()),
    static_cast<const uint8_t*>(p.mask.data()),original.shape[0],original.shape[1],s[0],s[1],s[2],o[0],o[1],o[2],label,taken.data());
  BROOK_CUDA(cudaGetLastError());
}
template<class Label> void apply(Context &ctx,SomaPreparation &state,const ComponentPreparation &p,const Box &box,int label) {
  auto s=p.mask.shape;auto o=box.lo;auto &volume=state.components.labels;
  overlay<<<blocks(p.mask.size()),256,0,ctx.stream>>>(static_cast<Label*>(volume.data()),static_cast<float*>(state.dbf.data()),
    static_cast<float*>(state.fields.daf.data()),static_cast<float*>(state.fields.pdrf.data()),static_cast<int*>(state.fields.parents.data()),
    static_cast<const uint8_t*>(p.mask.data()),static_cast<const float*>(p.dbf.data()),static_cast<const float*>(p.daf.data()),
    static_cast<const float*>(p.penalty.data()),static_cast<const long long*>(p.parents.data()),volume.shape[0],volume.shape[1],s[0],s[1],s[2],
    o[0],o[1],o[2],label,p.filled>0);
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();
}
bool enabled(const char *name) { const char *v=std::getenv(name);return !v || (std::string(v)!="0" && std::string(v)!="off"); }
struct PrivateArena { int label;Point origin;ComponentPreparation preparation;Array graph; };
template<class Label> __global__ void remove_arena_labels(Label *labels,const uint8_t *private_label,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && private_label[labels[i]]) labels[i]=0;
}
template<class Label> __global__ void append_mask(Label *output,const uint8_t *mask,int label,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) output[i]=mask[i]?Label(label):Label(0);
}
__global__ void append_parents(int *output,const long long *input,int offset,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) output[i]=input[i]<0?-1:int(input[i])+offset;
}
template<class Label> void labels_append(Context &ctx,Array &output,int64_t offset,const PrivateArena &arena) {
  const auto &mask=arena.preparation.mask;
  append_mask<<<blocks(mask.size()),256,0,ctx.stream>>>(static_cast<Label*>(output.data())+offset,static_cast<const uint8_t*>(mask.data()),arena.label,int(mask.size()));
  BROOK_CUDA(cudaGetLastError());
}
void append_arenas(Context &ctx,SomaPreparation &state,std::vector<PrivateArena> &arenas,bool fix) {
  auto &layout=state.layout;int64_t nbase=layout.offsets.back();
  std::vector<uint8_t> private_label(state.fields.active.size(),0);
  for(auto &a:arenas) {
    const auto &p=a.preparation;
    int slot=int(layout.dimensions.size());int64_t offset=layout.offsets.back();
    layout.label_arena[a.label]=slot;layout.dimensions.push_back(p.mask.shape);layout.origins.push_back(a.origin);
    layout.offsets.push_back(offset+int64_t(p.mask.size()));private_label[a.label]=1;
    state.fields.active[a.label]=1;state.fields.soma[a.label]=p.soma;state.fields.dbf_max[a.label]=p.maximum;
    state.fields.root[a.label]=offset+flatten(p.root,p.mask.shape);state.fields.target[a.label]=offset+flatten(p.target,p.mask.shape);
    state.boxes[a.label].count=p.valid;
  }
  std::array<int64_t,3> shape={layout.offsets.back(),1,1};
  auto new_labels=allocate(ctx,shape,state.components.labels.dtype);
  BROOK_CUDA(cudaMemcpyAsync(new_labels.data(),state.components.labels.data(),state.components.labels.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
  Buffer<uint8_t> flags(ctx,private_label.size());flags.set(ctx,private_label.data(),private_label.size());
  for(size_t index=0;index<arenas.size();++index) {
    auto &a=arenas[index];int64_t offset=layout.offsets[layout.label_arena[a.label]];
    switch(new_labels.dtype) {
      case BROOK_U8:labels_append<uint8_t>(ctx,new_labels,offset,a);break;
      case BROOK_U16:labels_append<uint16_t>(ctx,new_labels,offset,a);break;
      case BROOK_U32:labels_append<uint32_t>(ctx,new_labels,offset,a);break;
      case BROOK_U64:labels_append<uint64_t>(ctx,new_labels,offset,a);break;
      default:throw std::invalid_argument("invalid arena label dtype");
    }
  }
  switch(new_labels.dtype) {
    case BROOK_U8:remove_arena_labels<<<blocks(nbase),256,0,ctx.stream>>>(static_cast<uint8_t*>(new_labels.data()),flags.data(),int(nbase));break;
    case BROOK_U16:remove_arena_labels<<<blocks(nbase),256,0,ctx.stream>>>(static_cast<uint16_t*>(new_labels.data()),flags.data(),int(nbase));break;
    case BROOK_U32:remove_arena_labels<<<blocks(nbase),256,0,ctx.stream>>>(static_cast<uint32_t*>(new_labels.data()),flags.data(),int(nbase));break;
    case BROOK_U64:remove_arena_labels<<<blocks(nbase),256,0,ctx.stream>>>(static_cast<uint64_t*>(new_labels.data()),flags.data(),int(nbase));break;
    default:throw std::invalid_argument("invalid arena label dtype");
  }
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();state.components.labels=std::move(new_labels);
  for(auto &a:arenas) a.preparation.mask={};
  auto append_float=[&](Array &base,Array ComponentPreparation::*member) {
    auto result=allocate(ctx,shape,BROOK_F32);
    BROOK_CUDA(cudaMemcpyAsync(result.data(),base.data(),base.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
    for(auto &a:arenas) {
      auto &input=a.preparation.*member;int64_t offset=layout.offsets[layout.label_arena[a.label]];
      BROOK_CUDA(cudaMemcpyAsync(static_cast<float*>(result.data())+offset,input.data(),input.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
    }
    ctx.synchronize();base=std::move(result);
    for(auto &a:arenas) a.preparation.*member={};
  };
  append_float(state.dbf,&ComponentPreparation::dbf);append_float(state.fields.daf,&ComponentPreparation::daf);
  if(fix) append_float(state.fields.pdrf,&ComponentPreparation::penalty);
  else {
    auto result=allocate(ctx,shape,BROOK_I32);
    BROOK_CUDA(cudaMemcpyAsync(result.data(),state.fields.parents.data(),state.fields.parents.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
    for(auto &a:arenas) {
      auto &p=a.preparation.parents;int offset=int(layout.offsets[layout.label_arena[a.label]]);
      append_parents<<<blocks(p.size()),256,0,ctx.stream>>>(static_cast<int*>(result.data())+offset,static_cast<const long long*>(p.data()),offset,int(p.size()));
      BROOK_CUDA(cudaGetLastError());
    }
    ctx.synchronize();state.fields.parents=std::move(result);
    for(auto &a:arenas) a.preparation.parents={};
  }
  if(state.fields.graph.storage) {
    // The graph of a private arena is the crop its preparation ran on: its voxels' own words.
    auto result=allocate(ctx,shape,BROOK_U32);
    BROOK_CUDA(cudaMemcpyAsync(result.data(),state.fields.graph.data(),state.fields.graph.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
    for(auto &a:arenas) {
      int64_t offset=layout.offsets[layout.label_arena[a.label]];
      if(!a.graph.storage || a.graph.size()!=size_t(layout.offsets[layout.label_arena[a.label]+1]-offset))
        throw std::logic_error("private arena graph crop mismatch");
      BROOK_CUDA(cudaMemcpyAsync(static_cast<uint32_t*>(result.data())+offset,a.graph.data(),a.graph.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
    }
    ctx.synchronize();state.fields.graph=std::move(result);
    for(auto &a:arenas) a.graph={};
  }
}
}
SomaPreparation prepare_somata(Context &ctx,const Components &cc,const Array &dbf,BatchedPreparation fields,
    const std::vector<Box> &boxes,const std::vector<std::vector<Point>> &borders,
    const std::vector<std::vector<Point>> &before,const std::vector<std::vector<Point>> &after,
    const SkeletonizeOptions &options,const Array *graph) {
  SomaPreparation result{cc,dbf,boxes,std::move(fields),{}};
  size_t k=cc.mapping.size(),n=cc.labels.size();int64_t foreground=0;
  if(graph && (graph->dtype!=BROOK_U32 || graph->shape!=cc.labels.shape)) throw std::invalid_argument("soma preparation graph must be uint32 and match the volume");
  if(graph && !result.fields.graph.storage) result.fields.graph=*graph;
  result.layout.offsets={0,int64_t(n)};result.layout.dimensions={cc.labels.shape};result.layout.origins={{0,0,0}};result.layout.label_arena.resize(k,0);
  auto &prep=result.fields;
  // These images belong only to preamble work. Unused images are released at
  // the end of this phase, including all early returns, before the path loop.
  auto filled_masks=std::move(prep.filled_masks);
  if(!enabled("BROOK_SOMA_LOCKSTEP") || (options.fix_branching?!prep.pdrf.storage:!prep.parents.storage)) return result;
  std::vector<PrivateArena> arenas;int64_t arena_voxels=0;
  std::vector<uint8_t> survivor(k,0),evicted(k,0);
  for(size_t id=1;id<k;++id) {
    survivor[id]=double(boxes[id].count)>options.dust;
    if(!after[id].empty()) prep.active[id]=0;
    if(prep.active[id]) foreground+=boxes[id].count;
  }
  size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
  double room=0.85*double(free)-(double(n)*(13+(options.fix_branching?4:0))+double(foreground)*32);
  bool copied=false;
  for(size_t id=1;id<k;++id) {
    if(!survivor[id] || prep.active[id] || evicted[id] || double(prep.dbf_max[id])<=options.teasar.soma_detection || !after[id].empty()) continue;
    const auto &box=boxes[id];Point ext;
    for(int a=0;a<3;++a) ext[a]=box.hi[a]-box.lo[a]+1;
    int64_t size=int64_t(volume_size(ext));if(size<=1 || size>=INT32_MAX) continue;
    int64_t foreground_after=std::min(size,boxes[id].count+prep.filled[id]);
    double needed=double(foreground_after)*56+(copied?0:double(n)*(dtype_size(cc.labels.dtype)+4));
    // With a graph, the refresh after the fill is the graph EDT, which allocates a doubled-grid
    // work field (16 B per crop voxel) besides its output (4 B).
    if(graph && prep.filled[id]) needed+=double(size)*20;
    if(needed>room) continue;
    Array graph_crop;
    auto input=crop_component(ctx,cc.labels,dbf,id,box,graph,graph?&graph_crop:nullptr);
    auto manual=borders[id];std::optional<Point> root;
    if(!manual.empty()) { root=manual.back();manual.pop_back();for(int a=0;a<3;++a) (*root)[a]-=box.lo[a]; }
    manual.insert(manual.end(),before[id].begin(),before[id].end());
    for(auto &point:manual) for(int a=0;a<3;++a) point[a]-=box.lo[a];
    std::optional<std::pair<Array,int>> checked;
    auto retained=filled_masks.find(id);
    if(retained!=filled_masks.end()) {
      checked.emplace(std::move(retained->second),prep.filled[id]);filled_masks.erase(retained);
    }
    auto p=prepare_component(ctx,std::move(input.first),std::move(input.second),options,std::move(manual),{},root,graph?&graph_crop:nullptr,checked?&*checked:nullptr);
    checked.reset();
    if(!p.valid) continue;
    std::vector<int> enclosed;
    bool conflict=false;
    if(p.filled) {
      Buffer<int> taken(ctx,k);taken.clear(ctx);
      switch(cc.labels.dtype) {
        case BROOK_U8:query<uint8_t>(ctx,cc.labels,result.components.labels,p,box,int(id),taken);break;
        case BROOK_U16:query<uint16_t>(ctx,cc.labels,result.components.labels,p,box,int(id),taken);break;
        case BROOK_U32:query<uint32_t>(ctx,cc.labels,result.components.labels,p,box,int(id),taken);break;
        case BROOK_U64:query<uint64_t>(ctx,cc.labels,result.components.labels,p,box,int(id),taken);break;
        default:throw std::invalid_argument("invalid soma label dtype");
      }
      auto counts=taken.get(ctx,k);
      for(size_t owner=1;owner<k;++owner) if(counts[owner] && survivor[owner]) {
        bool pending=owner>id && !prep.active[owner] && !evicted[owner] && double(prep.dbf_max[owner])>options.teasar.soma_detection && after[owner].empty();
        if(!enabled("BROOK_SOMA_EVICT")) { conflict=true;break; }
        if(prep.active[owner] || pending) {
          if(counts[owner]!=boxes[owner].count) { conflict=true;break; }
          enclosed.push_back(int(owner));
        }
      }
    }
    if(conflict) {
      double arena_bytes=double(size)*(dtype_size(cc.labels.dtype)+4+1+8+4+(options.fix_branching?8:4)+32+24);
      if(enabled("BROOK_SOMA_ARENAS") && arena_bytes<=room && int64_t(n)+arena_voxels+size<INT32_MAX && arenas.size()<65534) {
        arenas.push_back({int(id),box.lo,std::move(p),std::move(graph_crop)});arena_voxels+=size;room-=arena_bytes;
      }
      continue;
    }
    for(auto owner:enclosed) { prep.active[owner]=0;evicted[owner]=1; }
    if(p.filled && !copied) {
      result.components.labels=clone(ctx,cc.labels);result.dbf=clone(ctx,dbf);
      room-=double(n)*(dtype_size(cc.labels.dtype)+4);copied=true;
    }
    switch(cc.labels.dtype) {
      case BROOK_U8:apply<uint8_t>(ctx,result,p,box,int(id));break;
      case BROOK_U16:apply<uint16_t>(ctx,result,p,box,int(id));break;
      case BROOK_U32:apply<uint32_t>(ctx,result,p,box,int(id));break;
      case BROOK_U64:apply<uint64_t>(ctx,result,p,box,int(id));break;
      default:throw std::invalid_argument("invalid soma label dtype");
    }
    auto to_global=[&](Point point) { for(int a=0;a<3;++a) point[a]+=box.lo[a];return flatten(point,cc.labels.shape); };
    prep.root[id]=to_global(p.root);prep.target[id]=to_global(p.target);prep.dbf_max[id]=p.maximum;
    prep.soma[id]=p.soma;prep.active[id]=1;result.boxes[id].count=p.valid;
    room-=double(p.valid)*32;
  }
  if(!arenas.empty()) append_arenas(ctx,result,arenas,options.fix_branching);
  return result;
}
}
