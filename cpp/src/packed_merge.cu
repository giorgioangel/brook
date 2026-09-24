// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "packed.hpp"
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_packed_merge
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <map>

namespace cub=brook_cccl_packed_merge::cub;
namespace thrust=brook_cccl_packed_merge::thrust;
namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
struct Part {
  const float *vertices,*radii;const uint32_t *edges;
  int vertices_count,edges_count,vertex_base,edge_base,group;
  float origin[3];
};
template<bool Edges> __device__ int owner(const Part *parts,int count,int index) {
  int lo=0,hi=count;
  while(lo+1<hi) { int mid=lo+(hi-lo)/2;int begin=Edges?parts[mid].edge_base:parts[mid].vertex_base;
    if(begin<=index) lo=mid;else hi=mid; }
  return lo;
}
__global__ void gather_input(const Part *parts,int part_count,int count,float *vertices,float *radii,int *groups,int *order) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  const auto &p=parts[owner<false>(parts,part_count,i)];int local=i-p.vertex_base;
  for(int a=0;a<3;++a) vertices[3LL*i+a]=p.vertices[3LL*local+a]+p.origin[a];
  radii[i]=p.radii[local];groups[i]=p.group;order[i]=i;
}
__device__ unsigned float_key(float value) {
  if(isnan(value)) return 0xffffffffu;
  unsigned bits=value==0.0f?0u:__float_as_uint(value);
  return bits ^ ((bits & 0x80000000u)?0xffffffffu:0x80000000u);
}
__global__ void sort_keys(const float *vertices,const int *groups,const int *order,unsigned *keys,int count,int axis) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) { int v=order[i];keys[i]=axis<3?float_key(vertices[3LL*v+axis]):unsigned(groups[v]); }
}
struct NewVertex {
  const float *vertices;const int *groups,*order;
  __device__ int operator()(int i) const {
    if(!i) return 1;int a=order[i-1],b=order[i];
    if(groups[a]!=groups[b]) return 1;
    for(int axis=0;axis<3;++axis) if(vertices[3LL*a+axis]!=vertices[3LL*b+axis]) return 1;
    return 0;
  }
};
__global__ void assign_vertices(const int *order,const int *ranks,int count,NewVertex is_new,int *inverse,int *first) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  int rank=ranks[i]-1;inverse[order[i]]=rank;if(is_new(i)) first[rank]=order[i];
}
__global__ void map_edges(const Part *parts,int part_count,int count,const int *inverse,unsigned long long *keys,int *error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  const auto &p=parts[owner<true>(parts,part_count,i)];int local=i-p.edge_base;
  unsigned a=p.edges[2LL*local],b=p.edges[2LL*local+1];keys[i]=~0ULL;
  if(a>=unsigned(p.vertices_count) || b>=unsigned(p.vertices_count)) { atomicExch(error,1);return; }
  a=unsigned(inverse[p.vertex_base+a]);b=unsigned(inverse[p.vertex_base+b]);
  if(a!=b) keys[i]=(static_cast<unsigned long long>(min(a,b))<<32)|max(a,b);
}
__global__ void mark_vertices(const unsigned long long *edges,int count,int *used) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) { atomicExch(used+(edges[i]>>32),1);atomicExch(used+unsigned(edges[i]),1); }
}
__global__ void output_vertices(const float *vertices,const float *radii,const int *groups,const int *first,
    const int *used,const int *rank,int count,float *out,float *out_radii,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !used[i]) return;
  int source=first[i],dest=rank[i]-1;
  for(int a=0;a<3;++a) out[3LL*dest+a]=vertices[3LL*source+a];
  out_radii[dest]=radii[source];atomicAdd(counts+groups[source],1);
}
__global__ void count_edges(const unsigned long long *edges,int count,const int *first,const int *groups,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) atomicAdd(counts+groups[first[edges[i]>>32]],1);
}
__global__ void output_edges(const unsigned long long *edges,int count,const int *first,const int *groups,
    const int *rank,const long long *offsets,uint32_t *out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  unsigned a=edges[i]>>32,b=unsigned(edges[i]);auto base=offsets[groups[first[a]]];
  out[2LL*i]=uint32_t(rank[a]-1-base);out[2LL*i+1]=uint32_t(rank[b]-1-base);
}
int last(Context &ctx,const Buffer<int> &values,int count) {
  int result=0;if(count) BROOK_CUDA(cudaMemcpyAsync(&result,values.data()+count-1,sizeof(int),cudaMemcpyDeviceToHost,ctx.stream));
  ctx.synchronize();return result;
}
}
PackedSkeletons merge_parts(Context &ctx,const std::vector<SkeletonPart> &input,std::array<float,3> aniso,bool sort_labels,bool drop_empty) {
  ctx.activate();auto result=empty_packed(ctx,aniso);
  std::map<int64_t,int> rank;
  for(const auto &p:input) if(p.radii.size()) {
    if(!rank.count(p.label)) { rank[p.label]=int(result.labels.size());result.labels.push_back(p.label); }
  }
  if(sort_labels) {
    std::sort(result.labels.begin(),result.labels.end());
    for(size_t i=0;i<result.labels.size();++i) rank[result.labels[i]]=int(i);
  }
  if(result.labels.empty()) return result;
  if(result.labels.size()>INT32_MAX) throw std::invalid_argument("too many packed labels");
  std::vector<Part> parts;size_t nv=0,ne=0;
  for(const auto &p:input) {
    size_t count=p.radii.size(),edges=p.edges.size()/2;
    if(p.vertices.dtype!=BROOK_F32 || p.radii.dtype!=BROOK_F32 || p.edges.dtype!=BROOK_U32 ||
        p.vertices.size()!=count*3 || p.edges.size()%2 || p.vertices.device!=ctx.device || p.edges.device!=ctx.device || p.radii.device!=ctx.device)
      throw std::invalid_argument("invalid packed skeleton buffers");
    if(!count && edges) throw std::invalid_argument("edges without vertices");
    if(!count) continue;
    if(count>size_t(INT32_MAX)-nv || edges>size_t(INT32_MAX)-ne) throw std::invalid_argument("packed merge exceeds int32 work capacity");
    parts.push_back({static_cast<const float*>(p.vertices.data()),static_cast<const float*>(p.radii.data()),static_cast<const uint32_t*>(p.edges.data()),
      int(count),int(edges),int(nv),int(ne),rank.at(p.label),{p.origin[0],p.origin[1],p.origin[2]}});
    nv+=count;ne+=edges;
  }
  Buffer<Part> descriptors(ctx,parts.size());descriptors.set(ctx,parts.data(),parts.size());
  Buffer<float> vertices(ctx,3*nv),radii(ctx,nv);Buffer<int> groups(ctx,nv),order_a(ctx,nv),order_b(ctx,nv),ranks(ctx,nv),inverse(ctx,nv);
  Buffer<unsigned> keys_a(ctx,nv),keys_b(ctx,nv);Buffer<uint8_t> scratch;
  auto primitive=[&](auto operation) {
    size_t bytes=0;BROOK_CUDA(operation(nullptr,bytes));
    if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(operation(scratch.data(),bytes));
  };
  gather_input<<<blocks(nv),256,0,ctx.stream>>>(descriptors.data(),int(parts.size()),int(nv),vertices.data(),radii.data(),groups.data(),order_a.data());
  for(int axis:{2,1,0,3}) {
    sort_keys<<<blocks(nv),256,0,ctx.stream>>>(vertices.data(),groups.data(),order_a.data(),keys_a.data(),int(nv),axis);
    primitive([&](void *p,size_t &bytes){return cub::DeviceRadixSort::SortPairs(p,bytes,keys_a.data(),keys_b.data(),order_a.data(),order_b.data(),int(nv),0,32,ctx.stream);});
    std::swap(order_a,order_b);
  }
  BROOK_CUDA(cudaGetLastError());
  NewVertex is_new{vertices.data(),groups.data(),order_a.data()};
  auto flags=thrust::make_transform_iterator(thrust::counting_iterator<int>(0),is_new);
  primitive([&](void *p,size_t &bytes){return cub::DeviceScan::InclusiveSum(p,bytes,flags,ranks.data(),int(nv),ctx.stream);});
  int unique=last(ctx,ranks,int(nv));Buffer<int> first(ctx,unique);
  assign_vertices<<<blocks(nv),256,0,ctx.stream>>>(order_a.data(),ranks.data(),int(nv),is_new,inverse.data(),first.data());
  BROOK_CUDA(cudaGetLastError());
  Buffer<unsigned long long> edge_a(ctx,ne+1),edge_b(ctx,ne+1);Buffer<int> error(ctx,1),selected(ctx,1);error.clear(ctx);
  if(ne) map_edges<<<blocks(ne),256,0,ctx.stream>>>(descriptors.data(),int(parts.size()),int(ne),inverse.data(),edge_a.data(),error.data());
  BROOK_CUDA(cudaMemsetAsync(edge_a.data()+ne,0xff,8,ctx.stream));
  if(ne==INT32_MAX) throw std::invalid_argument("packed merge edge sentinel exceeds capacity");
  primitive([&](void *p,size_t &bytes){return cub::DeviceRadixSort::SortKeys(p,bytes,edge_a.data(),edge_b.data(),int(ne+1),0,64,ctx.stream);});
  primitive([&](void *p,size_t &bytes){return cub::DeviceSelect::Unique(p,bytes,edge_b.data(),edge_a.data(),selected.data(),int(ne+1),ctx.stream);});
  int edges=selected.get(ctx,1)[0]-1;
  if(error.get(ctx,1)[0]) throw std::invalid_argument("packed edge is out of bounds");
  Buffer<int> used(ctx,unique);used.clear(ctx);
  if(edges) mark_vertices<<<blocks(edges),256,0,ctx.stream>>>(edge_a.data(),edges,used.data());
  primitive([&](void *p,size_t &bytes){return cub::DeviceScan::InclusiveSum(p,bytes,used.data(),ranks.data(),unique,ctx.stream);});
  int kept=last(ctx,ranks,unique);
  auto v=allocate(ctx,{3,kept,1},BROOK_F32),e=allocate(ctx,{2,edges,1},BROOK_U32),r=allocate(ctx,{kept,1,1},BROOK_F32);
  result.vertices=row_view(v,0,kept,3);result.edges=row_view(e,0,edges,2);result.radii=row_view(r,0,kept,1);
  Buffer<int> vertex_counts(ctx,result.labels.size()),edge_counts(ctx,result.labels.size());vertex_counts.clear(ctx);edge_counts.clear(ctx);
  output_vertices<<<blocks(unique),256,0,ctx.stream>>>(vertices.data(),radii.data(),groups.data(),first.data(),used.data(),ranks.data(),unique,
    static_cast<float*>(v.data()),static_cast<float*>(r.data()),vertex_counts.data());
  if(edges) count_edges<<<blocks(edges),256,0,ctx.stream>>>(edge_a.data(),edges,first.data(),groups.data(),edge_counts.data());
  BROOK_CUDA(cudaGetLastError());
  auto vc=vertex_counts.get(ctx,result.labels.size()),ec=edge_counts.get(ctx,result.labels.size());
  std::vector<long long> bases(result.labels.size()+1,0);
  result.vertex_offsets={0};result.edge_offsets={0};std::vector<int64_t> kept_labels;
  for(size_t i=0;i<result.labels.size();++i) {
    bases[i+1]=bases[i]+vc[i];
    if(drop_empty && !ec[i]) continue;
    kept_labels.push_back(result.labels[i]);
    result.vertex_offsets.push_back(result.vertex_offsets.back()+vc[i]);result.edge_offsets.push_back(result.edge_offsets.back()+ec[i]);
  }
  result.labels=std::move(kept_labels);Buffer<long long> offsets(ctx,bases.size());offsets.set(ctx,bases.data(),bases.size());
  if(edges) output_edges<<<blocks(edges),256,0,ctx.stream>>>(edge_a.data(),edges,first.data(),groups.data(),ranks.data(),offsets.data(),static_cast<uint32_t*>(e.data()));
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return result;
}
}
