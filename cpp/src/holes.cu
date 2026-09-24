// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "skeleton.hpp"
#include "trace_primitives.hpp"
#include "fillvoids_flood.cuh"
#include "cooperative.cuh"
#include "graph.hpp"
#include <cstdlib>
#include <algorithm>

namespace brook {
namespace holes_detail {
unsigned blocks(size_t n) {return unsigned((n+255)/256);}
template<class Label> __global__ void largest_label(const Label *input,long long n,unsigned long long *maximum) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) {unsigned long long label=input[i];if(label>*maximum) atomicMax(maximum,label);}
}
template<class Label> __global__ void make_mask(const Label *labels,uint8_t *mask,long long vx,long long vy,
    long long sx,long long sy,long long sz,long long ox,long long oy,long long oz,Label label) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=sx*sy*sz) return;
  long long x=i%sx,y=(i/sx)%sy,z=i/(sx*sy);
  mask[i]=labels[(x+ox)+vx*((y+oy)+vy*(z+oz))]==label;
}
template<class Label> __global__ void paint_holes(Label *labels,const uint8_t *mask,int *disabled,long long vx,long long vy,
    long long sx,long long sy,long long sz,long long ox,long long oy,long long oz,Label label) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=sx*sy*sz || !mask[i]) return;
  long long x=i%sx,y=(i/sx)%sy,z=i/(sx*sy),j=(x+ox)+vx*((y+oy)+vy*(z+oz));
  Label previous=labels[j];
  if(previous && previous!=label) atomicExch(disabled+previous,1);
  labels[j]=label;
}
template<class Label> std::pair<Array,int64_t> run(Context &ctx,Array labels,const std::vector<Box> &boxes) {
  if(labels.storage.use_count()>1) {
    auto copy=allocate(ctx,labels.shape,labels.dtype);
    BROOK_CUDA(cudaMemcpyAsync(copy.data(),labels.data(),labels.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
    labels=std::move(copy);
  }
  Buffer<int> disabled(ctx,boxes.size());disabled.clear(ctx);
  std::vector<int> skip(boxes.size(),0);int64_t total=0;
  // Preserve ascending original IDs and the original bounding boxes. An owner
  // touched by a fill is skipped later even when some of its voxels survive.
  for(size_t id=1;id<boxes.size();++id) {
    const auto &box=boxes[id];if(!box.count || skip[id]) continue;
    std::array<int64_t,3> shape;
    for(int a=0;a<3;++a) shape[a]=box.hi[a]-box.lo[a]+1;
    if(volume_size(shape)>=size_t{1}<<31) throw std::invalid_argument("hole-filling box exceeds int32 worklist capacity");
    auto mask=allocate(ctx,shape,BROOK_U8);
    make_mask<<<blocks(mask.size()),256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),static_cast<uint8_t*>(mask.data()),
      labels.shape[0],labels.shape[1],shape[0],shape[1],shape[2],box.lo[0],box.lo[1],box.lo[2],Label(id));
    BROOK_CUDA(cudaGetLastError());
    auto filled=fill_voids(ctx,mask);mask={};
    if(!filled.second) continue;
    total+=filled.second;
    paint_holes<<<blocks(filled.first.size()),256,0,ctx.stream>>>(static_cast<Label*>(labels.data()),static_cast<const uint8_t*>(filled.first.data()),
      disabled.data(),labels.shape[0],labels.shape[1],shape[0],shape[1],shape[2],box.lo[0],box.lo[1],box.lo[2],Label(id));
    BROOK_CUDA(cudaGetLastError());skip=disabled.get(ctx,boxes.size());
  }
  ctx.synchronize();return {std::move(labels),total};
}

struct HoleBox {long long sx,sy,sz,ox,oy,oz;};
struct HoleState {long long cursor,label,n;HoleBox box;};
__global__ void next_label(HoleState *state,const HoleBox *boxes,const long long *eligible,long long count,
    const int *disabled,int *counts,cudaGraphConditionalHandle condition) {
  long long cursor=state->cursor+1;
  while(cursor<count && disabled[eligible[cursor]]) ++cursor;
  state->cursor=cursor;state->label=cursor<count?eligible[cursor]:0;
  if(state->label) {state->box=boxes[state->label];state->n=state->box.sx*state->box.sy*state->box.sz;}
  else state->n=0;
  for(int i=0;i<8;++i) counts[i]=0;
  if(condition) cudaGraphSetConditional(condition,state->label!=0);
}
template<class Label> __global__ void seed_current(const Label *labels,uint8_t *mask,int *visited,int *front,
    int *counts,const HoleState *state,long long vx,long long vy) {
  auto box=state->box;long long n=state->n;Label label=Label(state->label);
  for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<long long>(blockDim.x)*gridDim.x) {
    long long x=i%box.sx,y=(i/box.sx)%box.sy,z=i/(box.sx*box.sy);
    bool foreground=labels[(x+box.ox)+vx*((y+box.oy)+vy*(z+box.oz))]==label;
    bool boundary=x==0 || y==0 || z==0 || x==box.sx-1 || y==box.sy-1 || z==box.sz-1;
    mask[i]=foreground;visited[i]=!foreground && boundary;
    if(!foreground && boundary) front[atomicAdd(counts,1)]=int(i);
  }
}
__global__ void __launch_bounds__(BROOK_COOP_BLOCK,BROOK_COOP_MIN_BLOCKS) flood_current(
    const uint8_t *mask,int *visited,int *a,int *b,int *counts,const HoleState *state) {
  if(!state->label) return;
  auto box=state->box;
  cuda::fill_flood_body(mask,visited,box.sx,box.sy,box.sz,a,b,counts);
}
template<class Label> __global__ void paint_current(Label *labels,const uint8_t *mask,const int *visited,int *disabled,
    const HoleState *state,long long vx,long long vy,unsigned long long *filled) {
  auto box=state->box;unsigned changed=0;Label label=Label(state->label);
  for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<state->n;i+=static_cast<long long>(blockDim.x)*gridDim.x) {
    if(mask[i] || visited[i]) continue;
    long long x=i%box.sx,y=(i/box.sx)%box.sy,z=i/(box.sx*box.sy);
    long long j=(x+box.ox)+vx*((y+box.oy)+vy*(z+box.oz));Label previous=labels[j];
    if(previous && previous!=label) atomicExch(disabled+previous,1);
    labels[j]=label;++changed;
  }
  for(int offset=16;offset;offset>>=1) changed+=__shfl_down_sync(0xffffffff,changed,offset);
  __shared__ unsigned sums[8];int lane=threadIdx.x%32,warp=threadIdx.x/32;
  if(!lane) sums[warp]=changed;
  __syncthreads();
  if(!warp) {
    changed=lane<8?sums[lane]:0;
    for(int offset=16;offset;offset>>=1) changed+=__shfl_down_sync(0xffffffff,changed,offset);
    if(!lane && changed) atomicAdd(filled,static_cast<unsigned long long>(changed));
  }
}
template<class Label> std::pair<Array,int64_t> run_graph(Context &ctx,Array labels,const std::vector<Box> &boxes) {
  std::vector<HoleBox> metadata(boxes.size());std::vector<long long> eligible;size_t largest=0;
  for(size_t id=1;id<boxes.size();++id) {
    const auto &b=boxes[id];if(!b.count) continue;
    HoleBox box{b.hi[0]-b.lo[0]+1,b.hi[1]-b.lo[1]+1,b.hi[2]-b.lo[2]+1,b.lo[0],b.lo[1],b.lo[2]};
    size_t n=volume_size({box.sx,box.sy,box.sz});
    // Such a box cannot enclose any six-connected background component.
    if(n==size_t(b.count) || box.sx<=2 || box.sy<=2 || box.sz<=2) continue;
    if(n>=size_t{1}<<31) { ctx.stats["hole_graph_declined"]=2;return run<Label>(ctx,std::move(labels),boxes); }   // 2: int32 box
    metadata[id]=box;eligible.push_back(static_cast<long long>(id));largest=std::max(largest,n);
  }
  size_t free=0,total_bytes=0;BROOK_CUDA(cudaMemGetInfo(&free,&total_bytes));
  double scratch=double(largest)*13+double(metadata.size())*(sizeof(HoleBox)+sizeof(int))+double(eligible.size())*8+sizeof(HoleState)+40;
  double copy_bytes=labels.storage.use_count()>1?double(labels.bytes()):0;
  const char *limit=std::getenv("BROOK_HOLE_WORKSPACE_BYTES");
  if(largest && (scratch+copy_bytes>0.85*double(free) || (limit && scratch>double(std::stoull(limit))))) {
    ctx.stats["hole_graph_declined"]=1;
    return run<Label>(ctx,std::move(labels),boxes);
  }
  if(labels.storage.use_count()>1) {
    auto copy=allocate(ctx,labels.shape,labels.dtype);
    BROOK_CUDA(cudaMemcpyAsync(copy.data(),labels.data(),labels.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));labels=std::move(copy);
  }
  if(!largest) {ctx.synchronize();return {std::move(labels),0};}
  Buffer<HoleBox> data(ctx,metadata.size());data.set(ctx,metadata.data(),metadata.size());
  Buffer<long long> ids(ctx,eligible.size());ids.set(ctx,eligible.data(),eligible.size());
  Buffer<int> disabled(ctx,boxes.size()),visited(ctx,largest),a(ctx,largest),b(ctx,largest),counts(ctx,8);disabled.clear(ctx);
  Buffer<uint8_t> mask(ctx,largest);Buffer<unsigned long long> total(ctx,1);total.clear(ctx);
  Buffer<HoleState> state(ctx,1);HoleState initial{};initial.cursor=-1;state.set(ctx,&initial,1);
  next_label<<<1,1,0,ctx.stream>>>(state.data(),data.data(),ids.data(),static_cast<long long>(eligible.size()),disabled.data(),counts.data(),0);
  BROOK_CUDA(cudaGetLastError());
  int grid=cooperative_grid(ctx,flood_current,largest,"BROOK_FLOOD_GRID");note_grid(ctx,"grid_hole_flood",grid);
  unsigned regular=std::min(blocks(largest),1024u);
  WhileGraph loop(ctx);
  loop.capture([&] {
    seed_current<<<regular,256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),mask.data(),visited.data(),a.data(),counts.data(),state.data(),labels.shape[0],labels.shape[1]);
    cooperative_launch(ctx,flood_current,grid,mask.data(),visited.data(),a.data(),b.data(),counts.data(),state.data());
    paint_current<<<regular,256,0,ctx.stream>>>(static_cast<Label*>(labels.data()),mask.data(),visited.data(),disabled.data(),state.data(),labels.shape[0],labels.shape[1],total.data());
    next_label<<<1,1,0,ctx.stream>>>(state.data(),data.data(),ids.data(),static_cast<long long>(eligible.size()),disabled.data(),counts.data(),loop.condition());
    BROOK_CUDA(cudaGetLastError());
  });
  loop.run();auto filled=total.get(ctx,1)[0];
  ctx.stats["hole_graph_labels"]=int64_t(eligible.size());ctx.stats["hole_graph_workspace_bytes"]=int64_t(largest*13);
  return {std::move(labels),int64_t(filled)};
}
template<class Label> std::pair<Array,int64_t> dispatch(Context &ctx,Array labels,const std::vector<Box> &boxes) {
  const char *setting=std::getenv("BROOK_HOLE_GRAPH");
  if(setting && std::string(setting)!="0" && graphs_enabled()) return run_graph<Label>(ctx,std::move(labels),boxes);
  return run<Label>(ctx,std::move(labels),boxes);
}
template<class Label> size_t count_labels(Context &ctx,const Array &labels) {
  Buffer<unsigned long long> maximum(ctx,1);maximum.clear(ctx);
  largest_label<<<blocks(labels.size()),256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),static_cast<long long>(labels.size()),maximum.data());
  BROOK_CUDA(cudaGetLastError());auto count=maximum.get(ctx,1)[0];
  if(count>labels.size()) throw std::invalid_argument("hole filling requires compact component labels");
  return size_t(count);
}
}
using holes_detail::dispatch;
using holes_detail::count_labels;
std::pair<Array,int64_t> fill_all_holes(Context &ctx,Array labels,const std::vector<Box> &boxes) {
  ctx.activate();
  switch(labels.dtype) {
    case BROOK_U8:return dispatch<uint8_t>(ctx,std::move(labels),boxes);
    case BROOK_U16:return dispatch<uint16_t>(ctx,std::move(labels),boxes);
    case BROOK_U32:return dispatch<uint32_t>(ctx,std::move(labels),boxes);
    case BROOK_U64:return dispatch<uint64_t>(ctx,std::move(labels),boxes);
    default:throw std::invalid_argument("hole filling requires unsigned component labels");
  }
}
std::pair<Array,int64_t> fill_all_holes(Context &ctx,Array labels) {
  ctx.activate();if(!labels.size()) return {std::move(labels),0};
  size_t count;
  switch(labels.dtype) {
    case BROOK_U8:count=count_labels<uint8_t>(ctx,labels);break;
    case BROOK_U16:count=count_labels<uint16_t>(ctx,labels);break;
    case BROOK_U32:count=count_labels<uint32_t>(ctx,labels);break;
    case BROOK_U64:count=count_labels<uint64_t>(ctx,labels);break;
    default:throw std::invalid_argument("hole filling requires unsigned component labels");
  }
  auto boxes=analyze(ctx,labels,count);
  return fill_all_holes(ctx,std::move(labels),boxes);
}
}
