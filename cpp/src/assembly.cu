// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "assembly.hpp"
#include "common.cuh"
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_assembly
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <numeric>

namespace cub=brook_cccl_assembly::cub;
namespace thrust=brook_cccl_assembly::thrust;
namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
template<class T> Buffer<T> upload(Context &ctx,const std::vector<T> &host) {
  Buffer<T> out(ctx,host.size());out.set(ctx,host.data(),host.size());return out;
}
__global__ void ends(uint8_t *inner,const int *start,const int *length,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count && length[i]>0) inner[start[i]+length[i]-1]=0;
}
template<class Label> __global__ void vertex_keys(const Label *labels,const uint8_t *handled,const int *store,int count,
    const long long *offsets,const long long *dimensions,const long long *origins,const int *base,int arenas,
    unsigned long long volume,unsigned long long *keys,int *order) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  int voxel=store[i];order[i]=i;auto label=labels[voxel];
  if(!handled[label]) { keys[i]=~0ULL;return; }
  int lo=0,hi=arenas;
  while(lo+1<hi) { int mid=lo+(hi-lo)/2;if(offsets[mid]<=voxel) lo=mid;else hi=mid; }
  long long x,y,z;cuda::brook_unravel(voxel-offsets[lo],dimensions[3*lo],dimensions[3*lo+1],x,y,z);
  x+=origins[3*lo];y+=origins[3*lo+1];z+=origins[3*lo+2];
  const int b=base[lo];const long long sy=dimensions[3*b+1],sz=dimensions[3*b+2];   // the base arena's frame
  keys[i]=static_cast<unsigned long long>(label)*volume+(x*sy+y)*sz+z;
}
struct NewRun {
  const unsigned long long *keys;
  __device__ int operator()(int i) const { return keys[i]!=~0ULL && (i==0 || keys[i]!=keys[i-1]); }
};
__global__ void unique_vertices(const unsigned long long *keys,const int *order,const int *ranks,int count,
    int *inverse,unsigned long long *unique,int *first) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  if(keys[i]==~0ULL) { inverse[order[i]]=-1;return; }
  int rank=ranks[i]-1;inverse[order[i]]=rank;
  if(i==0 || keys[i]!=keys[i-1]) { unique[rank]=keys[i];first[rank]=order[i]; }
}
__global__ void edge_keys(const int *inverse,const uint8_t *inner,int count,unsigned long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  unsigned long long key=~0ULL;
  if(i+1<count && inner[i]) {
    int a=inverse[i],b=inverse[i+1];
    if(a>=0 && b>=0 && a!=b) key=(static_cast<unsigned long long>(min(a,b))<<32)|unsigned(max(a,b));
  }
  keys[i]=key;
}
__global__ void mark_used(const unsigned long long *edges,int count,int *used) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  auto key=edges[i];atomicExch(used+int(key>>32),1);atomicExch(used+unsigned(key),1);
}
// Keys decode in the frame of the label's base arena: ldims holds that arena's dimensions per label,
// lz (null: none) the first z slice of the label's sample in a stack, subtracted before scaling.
__global__ void gather_vertices(const unsigned long long *keys,const int *first,const int *used,const int *rank,int count,
    unsigned long long volume,const long long *ldims,const long long *lz,const int *store,const float *dbf,
    float *vertices,float *radii,int *counts,float ax,float ay,float az) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !used[i]) return;
  int j=rank[i]-1;auto key=keys[i];int label=int(key/volume);long long loc=key%volume;
  const long long sy=ldims[3*label+1],sz=ldims[3*label+2];
  long long x=loc/(sy*sz),y=(loc/sz)%sy,z=loc%sz-(lz?lz[label]:0);
  vertices[3LL*j]=float(x)*ax;vertices[3LL*j+1]=float(y)*ay;vertices[3LL*j+2]=float(z)*az;
  radii[j]=dbf[store[first[i]]];atomicAdd(counts+label,1);
}
__global__ void edge_counts(const unsigned long long *edges,int count,const unsigned long long *vertices,
    unsigned long long volume,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) atomicAdd(counts+int(vertices[edges[i]>>32]/volume),1);
}
__global__ void pack_edges(const unsigned long long *edges,int count,const unsigned long long *vertices,
    unsigned long long volume,const int *rank,const long long *offsets,uint32_t *out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  unsigned a=edges[i]>>32,b=unsigned(edges[i]);int label=int(vertices[a]/volume);
  out[2LL*i]=uint32_t(rank[a]-1-offsets[label]);out[2LL*i+1]=uint32_t(rank[b]-1-offsets[label]);
}
ComponentAssembly empty(Context &ctx,size_t labels) {
  return {allocate(ctx,{3,0,1},BROOK_F32),allocate(ctx,{2,0,1},BROOK_U32),allocate(ctx,{0,1,1},BROOK_F32),
    std::vector<int64_t>(labels+1,0),std::vector<int64_t>(labels+1,0)};
}
template<class Label> ComponentAssembly run(Context &ctx,const Components &cc,const Array &dbf,const ArenaLayout &layout,
    const std::vector<uint8_t> &handled,const int *store,int count,const int *record_start,const int *record_length,int records,
    std::array<float,3> aniso) {
  std::vector<int> host_base=layout.base;
  if(host_base.size()!=layout.dimensions.size()) host_base.assign(layout.dimensions.size(),0);
  size_t volume=0;                                   // the largest base arena: keys stay unique per label
  for(size_t a=0;a<layout.dimensions.size();++a) if(host_base[a]==int(a)) volume=std::max(volume,volume_size(layout.dimensions[a]));
  if(!volume) volume=volume_size(layout.dimensions.front());
  const size_t labels=cc.mapping.size();
  if(labels>INT32_MAX || volume>UINT64_MAX/labels) throw std::invalid_argument("assembly key capacity exceeded");
  Buffer<uint8_t> inner(ctx,count);BROOK_CUDA(cudaMemsetAsync(inner.data(),1,count,ctx.stream));
  ends<<<blocks(records),256,0,ctx.stream>>>(inner.data(),record_start,record_length,records);
  auto active=upload(ctx,handled);
  std::vector<long long> host_offsets(layout.offsets.begin(),layout.offsets.end()),host_dims,host_origins;
  for(auto d:layout.dimensions) host_dims.insert(host_dims.end(),d.begin(),d.end());
  for(auto o:layout.origins) host_origins.insert(host_origins.end(),o.begin(),o.end());
  auto offsets=upload(ctx,host_offsets),dims=upload(ctx,host_dims),origins=upload(ctx,host_origins);
  auto bases=upload(ctx,host_base);
  std::vector<long long> host_ldims(3*labels,1);
  for(size_t l=0;l<labels;++l) {
    int arena=l<layout.label_arena.size()?layout.label_arena[l]:0;
    const auto &d=layout.dimensions[host_base[arena]];
    host_ldims[3*l]=d[0];host_ldims[3*l+1]=d[1];host_ldims[3*l+2]=d[2];
  }
  auto ldims=upload(ctx,host_ldims);
  std::vector<long long> host_lz(layout.label_z.empty()?0:labels,0);
  for(size_t l=0;l<host_lz.size() && l<layout.label_z.size();++l) host_lz[l]=layout.label_z[l];
  Buffer<long long> lz;if(!host_lz.empty()) lz=upload(ctx,host_lz);
  Buffer<unsigned long long> key_a(ctx,count),key_b(ctx,count);
  Buffer<int> order_a(ctx,count),order_b(ctx,count),ranks(ctx,count),inverse(ctx,count);
  vertex_keys<<<blocks(count),256,0,ctx.stream>>>(static_cast<const Label*>(cc.labels.data()),active.data(),store,count,
    offsets.data(),dims.data(),origins.data(),bases.data(),int(layout.dimensions.size()),volume,key_a.data(),order_a.data());
  BROOK_CUDA(cudaGetLastError());
  Buffer<uint8_t> scratch;
  auto primitive=[&](auto operation) {
    size_t bytes=0;BROOK_CUDA(operation(nullptr,bytes));
    if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(operation(scratch.data(),bytes));
  };
  primitive([&](void *p,size_t &bytes){return cub::DeviceRadixSort::SortPairs(p,bytes,key_a.data(),key_b.data(),order_a.data(),order_b.data(),count,0,64,ctx.stream);});
  auto flags=thrust::make_transform_iterator(thrust::counting_iterator<int>(0),NewRun{key_b.data()});
  primitive([&](void *p,size_t &bytes){return cub::DeviceScan::InclusiveSum(p,bytes,flags,ranks.data(),count,ctx.stream);});
  int unique_count=0;BROOK_CUDA(cudaMemcpyAsync(&unique_count,ranks.data()+count-1,4,cudaMemcpyDeviceToHost,ctx.stream));ctx.synchronize();
  if(!unique_count) return empty(ctx,labels);
  Buffer<unsigned long long> unique(ctx,unique_count);Buffer<int> first(ctx,unique_count);
  unique_vertices<<<blocks(count),256,0,ctx.stream>>>(key_b.data(),order_b.data(),ranks.data(),count,inverse.data(),unique.data(),first.data());
  edge_keys<<<blocks(count),256,0,ctx.stream>>>(inverse.data(),inner.data(),count,key_a.data());
  BROOK_CUDA(cudaGetLastError());
  order_a={};order_b={};
  primitive([&](void *p,size_t &bytes){return cub::DeviceRadixSort::SortKeys(p,bytes,key_a.data(),key_b.data(),count,0,64,ctx.stream);});
  Buffer<int> selected(ctx,1);
  primitive([&](void *p,size_t &bytes){return cub::DeviceSelect::Unique(p,bytes,key_b.data(),key_a.data(),selected.data(),count,ctx.stream);});
  // Every nonempty path has a last vertex, whose edge key is the sentinel.
  int edges=selected.get(ctx,1)[0]-1;
  if(!edges) return empty(ctx,labels);
  Buffer<int> used(ctx,unique_count);used.clear(ctx);
  mark_used<<<blocks(edges),256,0,ctx.stream>>>(key_a.data(),edges,used.data());
  BROOK_CUDA(cudaGetLastError());
  primitive([&](void *p,size_t &bytes){return cub::DeviceScan::InclusiveSum(p,bytes,used.data(),ranks.data(),unique_count,ctx.stream);});
  int vertices=0;BROOK_CUDA(cudaMemcpyAsync(&vertices,ranks.data()+unique_count-1,4,cudaMemcpyDeviceToHost,ctx.stream));ctx.synchronize();
  ComponentAssembly result{allocate(ctx,{3,vertices,1},BROOK_F32),allocate(ctx,{2,edges,1},BROOK_U32),allocate(ctx,{vertices,1,1},BROOK_F32),
    std::vector<int64_t>(labels+1,0),std::vector<int64_t>(labels+1,0)};
  Buffer<int> nv(ctx,labels),ne(ctx,labels);nv.clear(ctx);ne.clear(ctx);
  gather_vertices<<<blocks(unique_count),256,0,ctx.stream>>>(unique.data(),first.data(),used.data(),ranks.data(),unique_count,volume,ldims.data(),lz.data(),store,
    static_cast<const float*>(dbf.data()),static_cast<float*>(result.vertices.data()),static_cast<float*>(result.radii.data()),nv.data(),aniso[0],aniso[1],aniso[2]);
  edge_counts<<<blocks(edges),256,0,ctx.stream>>>(key_a.data(),edges,unique.data(),volume,ne.data());
  BROOK_CUDA(cudaGetLastError());
  auto host_nv=nv.get(ctx,labels),host_ne=ne.get(ctx,labels);
  for(size_t i=0;i<labels;++i) { result.vertex_offsets[i+1]=result.vertex_offsets[i]+host_nv[i];result.edge_offsets[i+1]=result.edge_offsets[i]+host_ne[i]; }
  std::vector<long long> voff(result.vertex_offsets.begin(),result.vertex_offsets.end());auto vbase=upload(ctx,voff);
  pack_edges<<<blocks(edges),256,0,ctx.stream>>>(key_a.data(),edges,unique.data(),volume,ranks.data(),vbase.data(),static_cast<uint32_t*>(result.edges.data()));
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return result;
}
}
ComponentAssembly assemble_paths(Context &ctx,const Components &cc,const Array &dbf,const ArenaLayout &layout,
    const std::vector<uint8_t> &handled,const int *store,int count,const int *starts,const int *lengths,int records,std::array<float,3> aniso) {
  ctx.activate();
  if(!count) return empty(ctx,cc.mapping.size());
  if(count<0 || records<=0 || layout.dimensions.empty() || handled.size()!=cc.mapping.size()) throw std::invalid_argument("invalid path assembly input");
  switch(cc.labels.dtype) {
    case BROOK_U8:return run<uint8_t>(ctx,cc,dbf,layout,handled,store,count,starts,lengths,records,aniso);
    case BROOK_U16:return run<uint16_t>(ctx,cc,dbf,layout,handled,store,count,starts,lengths,records,aniso);
    case BROOK_U32:return run<uint32_t>(ctx,cc,dbf,layout,handled,store,count,starts,lengths,records,aniso);
    case BROOK_U64:return run<uint64_t>(ctx,cc,dbf,layout,handled,store,count,starts,lengths,records,aniso);
    default:throw std::invalid_argument("invalid assembly component dtype");
  }
}
void download_components(const ComponentAssembly &packed,const Components &cc,const std::vector<uint8_t> &handled,
    std::array<float,3> aniso,std::vector<std::optional<Skeleton>> &out) {
  std::vector<std::array<float,3>> vertices(packed.vertex_offsets.back());
  std::vector<std::array<uint32_t,2>> edges(packed.edge_offsets.back());std::vector<float> radii(vertices.size());
  packed.vertices.copy_to_host(vertices.data(),vertices.size()*sizeof(vertices[0]));
  packed.edges.copy_to_host(edges.data(),edges.size()*sizeof(edges[0]));packed.radii.copy_to_host(radii.data(),radii.size()*sizeof(float));
  out.resize(handled.size());
  for(size_t id=1;id<handled.size();++id) if(handled[id]) {
    Skeleton s;s.label=cc.mapping[id];s.anisotropy=aniso;
    s.vertices.assign(vertices.begin()+packed.vertex_offsets[id],vertices.begin()+packed.vertex_offsets[id+1]);
    s.radii.assign(radii.begin()+packed.vertex_offsets[id],radii.begin()+packed.vertex_offsets[id+1]);
    s.edges.assign(edges.begin()+packed.edge_offsets[id],edges.begin()+packed.edge_offsets[id+1]);out[id]=std::move(s);
  }
}
}
