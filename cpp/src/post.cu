// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "post.hpp"
#include "ccl_kernels.cuh"
#include "post_kernels.cuh"
#include <math_constants.h>
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_post
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace cub=brook_cccl_post::cub;
namespace thrust=brook_cccl_post::thrust;
namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
struct GraphTag {};
struct Graph {
  int n,m,k;
  Buffer<long long> voff,eoff;
  Buffer<int> vertex_label,edge_label,parent,degree;
  Buffer<int2> edges;
};
__device__ int group(const long long *offsets,int count,int i) {
  int lo=0,hi=count;
  while(lo+1<hi) { int mid=lo+(hi-lo)/2;if(offsets[mid]<=i) lo=mid;else hi=mid; }
  return lo;
}
__global__ void initialize(int *parent,int *degree,int *labels,int n,const long long *offsets,int k) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { parent[i]=i;degree[i]=0;labels[i]=group(offsets,k,i); }
}
__global__ void global_edges(const uint32_t *input,int2 *output,int *label,int *parent,int *degree,int m,
    const long long *voff,const long long *eoff,int k,int *error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m) return;
  int l=group(eoff,k,i);unsigned a=input[2LL*i],b=input[2LL*i+1];label[i]=l;
  if(a>=voff[l+1]-voff[l] || b>=voff[l+1]-voff[l]) { atomicExch(error,1);output[i]={0,0};return; }
  int ga=int(voff[l])+int(a),gb=int(voff[l])+int(b);output[i]={ga,gb};
  atomicAdd(degree+ga,1);atomicAdd(degree+gb,1);cuda::brook_union(parent,ga,gb);
}
Graph graph(Context &ctx,const PackedSkeletons &p) {
  if(p.vertices.device!=ctx.device || p.edges.device!=ctx.device || p.radii.device!=ctx.device ||
      p.vertices.dtype!=BROOK_F32 || p.edges.dtype!=BROOK_U32 || p.radii.dtype!=BROOK_F32 ||
      p.vertex_offsets.size()!=p.labels.size()+1 || p.edge_offsets.size()!=p.labels.size()+1 ||
      p.vertex_offsets.front()!=0 || p.edge_offsets.front()!=0 || !std::is_sorted(p.vertex_offsets.begin(),p.vertex_offsets.end()) ||
      !std::is_sorted(p.edge_offsets.begin(),p.edge_offsets.end()) || p.vertex_offsets.back()<0 || p.edge_offsets.back()<0 ||
      size_t(p.vertex_offsets.back())!=p.radii.size() || p.vertices.size()!=p.radii.size()*3 || size_t(p.edge_offsets.back())*2!=p.edges.size())
    throw std::invalid_argument("invalid packed graph");
  if(p.vertex_offsets.back()>INT32_MAX || p.edge_offsets.back()>INT32_MAX || p.labels.size()>INT32_MAX)
    throw std::invalid_argument("packed graph exceeds int32 work capacity");
  Graph g;g.n=int(p.vertex_offsets.back());g.m=int(p.edge_offsets.back());g.k=int(p.labels.size());
  g.voff=Buffer<long long>(ctx,g.k+1);g.eoff=Buffer<long long>(ctx,g.k+1);
  std::vector<long long> v(p.vertex_offsets.begin(),p.vertex_offsets.end()),e(p.edge_offsets.begin(),p.edge_offsets.end());
  g.voff.set(ctx,v.data(),v.size());g.eoff.set(ctx,e.data(),e.size());
  g.vertex_label=Buffer<int>(ctx,g.n);g.edge_label=Buffer<int>(ctx,g.m);g.parent=Buffer<int>(ctx,g.n);
  g.degree=Buffer<int>(ctx,g.n);g.edges=Buffer<int2>(ctx,g.m);
  if(g.n) initialize<<<blocks(g.n),256,0,ctx.stream>>>(g.parent.data(),g.degree.data(),g.vertex_label.data(),g.n,g.voff.data(),g.k);
  Buffer<int> error(ctx,1);error.clear(ctx);
  if(g.m) global_edges<<<blocks(g.m),256,0,ctx.stream>>>(static_cast<const uint32_t*>(p.edges.data()),g.edges.data(),g.edge_label.data(),g.parent.data(),
    g.degree.data(),g.m,g.voff.data(),g.eoff.data(),g.k,error.data());
  if(g.n) cuda::ccl_flatten<GraphTag,int,uint8_t><<<blocks(g.n),256,0,ctx.stream>>>(g.parent.data(),g.n);
  BROOK_CUDA(cudaGetLastError());if(error.get(ctx,1)[0]) throw std::invalid_argument("packed edge is out of bounds");return g;
}
__global__ void widen_parent(const int *parent,long long *out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[i]=parent[i];
}
struct KeepValue { const uint8_t *keep;__device__ int operator()(int i) const { return keep[i]!=0; } };
struct KeepEdge { const uint8_t *keep;const int2 *edges;__device__ int operator()(int i) const { int2 e=edges[i];return keep[e.x] && keep[e.y]; } };
__global__ void selected_counts(const uint8_t *keep,const int *labels,int n,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && keep[i]) atomicAdd(counts+labels[i],1);
}
__global__ void selected_edge_counts(KeepEdge keep,const int *labels,int n,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && keep(i)) atomicAdd(counts+labels[i],1);
}
__global__ void select_points(const float *vertices,const float *radii,const uint8_t *keep,const int *rank,int n,float *out,float *rout) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || !keep[i]) return;int j=rank[i]-1;
  for(int a=0;a<3;++a) out[3LL*j+a]=vertices[3LL*i+a];rout[j]=radii[i];
}
__global__ void select_edges(KeepEdge keep,const int *edge_rank,const int *vertex_rank,const int *label,int m,
    const long long *voff,uint32_t *out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m || !keep(i)) return;
  int2 e=keep.edges[i];int j=edge_rank[i]-1;long long base=voff[label[i]];
  out[2LL*j]=uint32_t(vertex_rank[e.x]-1-base);out[2LL*j+1]=uint32_t(vertex_rank[e.y]-1-base);
}
PackedSkeletons select(Context &ctx,const PackedSkeletons &p,const Array &keep,Graph &g) {
  if(keep.dtype!=BROOK_U8 || keep.size()!=size_t(g.n) || keep.device!=ctx.device) throw std::invalid_argument("invalid packed vertex selection");
  if(!g.n) return p;
  auto bits=static_cast<const uint8_t*>(keep.data());Buffer<int> vrank(ctx,g.n),erank(ctx,g.m),vc(ctx,g.k),ec(ctx,g.k);vc.clear(ctx);ec.clear(ctx);
  Buffer<uint8_t> scratch;
  auto scan=[&](auto input,int *output,int n) {
    if(!n) return;size_t bytes=0;BROOK_CUDA(cub::DeviceScan::InclusiveSum(nullptr,bytes,input,output,n,ctx.stream));
    if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(cub::DeviceScan::InclusiveSum(scratch.data(),bytes,input,output,n,ctx.stream));
  };
  auto sequence=thrust::counting_iterator<int>(0);KeepEdge ke{bits,g.edges.data()};
  scan(thrust::make_transform_iterator(sequence,KeepValue{bits}),vrank.data(),g.n);
  scan(thrust::make_transform_iterator(sequence,ke),erank.data(),g.m);
  selected_counts<<<blocks(g.n),256,0,ctx.stream>>>(bits,g.vertex_label.data(),g.n,vc.data());
  if(g.m) selected_edge_counts<<<blocks(g.m),256,0,ctx.stream>>>(ke,g.edge_label.data(),g.m,ec.data());
  BROOK_CUDA(cudaGetLastError());auto hv=vc.get(ctx,g.k),he=ec.get(ctx,g.k);
  PackedSkeletons out;out.labels=p.labels;out.anisotropy=p.anisotropy;
  for(int i=0;i<g.k;++i) { out.vertex_offsets.push_back(out.vertex_offsets.back()+hv[i]);out.edge_offsets.push_back(out.edge_offsets.back()+he[i]); }
  int64_t nv=out.vertex_offsets.back(),ne=out.edge_offsets.back();
  auto v=allocate(ctx,{3,nv,1},BROOK_F32),e=allocate(ctx,{2,ne,1},BROOK_U32),r=allocate(ctx,{nv,1,1},BROOK_F32);
  out.vertices=row_view(v,0,nv,3);out.edges=row_view(e,0,ne,2);out.radii=row_view(r,0,nv,1);
  std::vector<long long> voff(out.vertex_offsets.begin(),out.vertex_offsets.end());Buffer<long long> offsets(ctx,voff.size());offsets.set(ctx,voff.data(),voff.size());
  select_points<<<blocks(g.n),256,0,ctx.stream>>>(static_cast<const float*>(p.vertices.data()),static_cast<const float*>(p.radii.data()),bits,vrank.data(),g.n,
    static_cast<float*>(v.data()),static_cast<float*>(r.data()));
  if(g.m) select_edges<<<blocks(g.m),256,0,ctx.stream>>>(ke,erank.data(),vrank.data(),g.edge_label.data(),g.m,offsets.data(),static_cast<uint32_t*>(e.data()));
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return out;
}
__global__ void degree_keep(const int *degree,uint8_t *keep,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) keep[i]=degree[i]!=0;
}
__global__ void lengths(const float *vertices,const int2 *edges,const int *parent,int m,double *totals,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m) return;int2 e=edges[i];
  double x=double(vertices[3LL*e.y]-vertices[3LL*e.x]);
  double y=double(vertices[3LL*e.y+1]-vertices[3LL*e.x+1]);
  double z=double(vertices[3LL*e.y+2]-vertices[3LL*e.x+2]);
  int root=parent[e.x];atomicAdd(totals+root,sqrt((x*x+y*y)+z*z));atomicAdd(counts+root,1);
}
struct Uncertain {
  const double *totals;const int *counts;double threshold;
  __device__ int operator()(int i) const { return counts[i]>0 && fabs(totals[i]-threshold)<=counts[i]*(1.0/8388608.0)*threshold; }
};
__global__ void keep_lengths(const double *total,uint8_t *keep,int n,double threshold) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) keep[i]=total[i]>threshold;
}
__global__ void set_decisions(uint8_t *keep,const int *indices,const uint8_t *values,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) keep[indices[i]]=values[i];
}
__global__ void broadcast_keep(const uint8_t *roots,const int *parent,uint8_t *out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[i]=roots[parent[i]];
}
float pairwise_sum(const float *values,size_t n) {
  // NumPy 2.4.6's eight-lane pairwise reduction order; see licenses/NumPy.txt.
  if(n<8) { float out=-0.0f;for(size_t i=0;i<n;++i) out+=values[i];return out; }
  if(n<=128) {
    float r[8];std::copy(values,values+8,r);size_t i=8;
    for(;i<n-(n%8);i+=8) for(int j=0;j<8;++j) r[j]+=values[i+j];
    float out=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
    for(;i<n;++i) out+=values[i];return out;
  }
  size_t mid=n/2;mid-=mid%8;return pairwise_sum(values,mid)+pairwise_sum(values+mid,n-mid);
}
__global__ void component_sizes(const int *parent,const int *degree,int n,const int2 *edges,int m,int *nv,int *ne,int *branch) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { int root=parent[i];atomicAdd(nv+root,1);if(degree[i]>=3) atomicExch(branch+root,1); }
  if(i<m) atomicAdd(ne+parent[edges[i].x],1);
}
struct WorkRoot {
  const int *nv,*ne,*branch;bool trees;
  __device__ int operator()(int i) const { return trees?(branch[i] && ne[i]==nv[i]-1):(ne[i]>0 && ne[i]-nv[i]+1>0); }
};
__global__ void root_slots(const int *roots,int count,int *slot,const int *nv,const int *ne,int *sizes) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) { int root=roots[i];slot[root]=i;sizes[2*i]=nv[root];sizes[2*i+1]=ne[root]; }
}
struct ActiveVertex { const int *parent,*slot;__device__ int operator()(int i) const { return slot[parent[i]]>=0; } };
struct ActiveEdge { const int2 *edges;const int *parent,*slot;__device__ int operator()(int i) const { return slot[parent[edges[i].x]]>=0; } };
__global__ void arena_keys(const int *indices,int n,const int *parent,const int *slot,const int2 *edges,unsigned *keys,bool edge) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) { int v=edge?edges[indices[i]].x:indices[i];keys[i]=unsigned(slot[parent[v]]); }
}
__global__ void arena_vertices(const int *selected,int n,const int *parent,const int *slot,const long long *offsets,
    const float *vertices,const float *radii,int *local,float *xyz,float *rad) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;int v=selected[i];local[v]=int(i-offsets[slot[parent[v]]]);
  for(int a=0;a<3;++a) xyz[3LL*i+a]=vertices[3LL*v+a];rad[i]=radii[v];
}
__global__ void arena_edges(const int *selected,int n,const int2 *edges,const int *parent,const int *slot,const int *local,int *comp,int2 *out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) { int2 e=edges[selected[i]];comp[i]=slot[parent[e.x]];out[i]={local[e.x],local[e.y]}; }
}
struct Arena {
  int count=0,n=0,m=0;
  Buffer<int> roots,slot,vertices,edges,component,local,ne;
  Buffer<long long> vs,es;
  Buffer<int2> edge_rows;
  Buffer<float> xyz,radii;
};
Arena arena(Context &ctx,const PackedSkeletons &p,const Graph &g,bool trees) {
  Arena a;Buffer<int> nv(ctx,g.n),ne(ctx,g.n),branch(ctx,g.n);nv.clear(ctx);ne.clear(ctx);branch.clear(ctx);
  component_sizes<<<blocks(std::max(g.n,g.m)),256,0,ctx.stream>>>(g.parent.data(),g.degree.data(),g.n,g.edges.data(),g.m,nv.data(),ne.data(),branch.data());
  WorkRoot predicate{nv.data(),ne.data(),branch.data(),trees};auto sequence=thrust::counting_iterator<int>(0);
  auto flags=thrust::make_transform_iterator(sequence,predicate);Buffer<int> count(ctx,1);Buffer<uint8_t> scratch;
  auto primitive=[&](auto operation) {
    size_t bytes=0;BROOK_CUDA(operation(nullptr,bytes));if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(operation(scratch.data(),bytes));
  };
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceReduce::Sum(tmp,bytes,flags,count.data(),g.n,ctx.stream);});
  a.count=count.get(ctx,1)[0];if(!a.count) return a;
  a.roots=Buffer<int>(ctx,a.count);a.slot=Buffer<int>(ctx,g.n);BROOK_CUDA(cudaMemsetAsync(a.slot.data(),0xff,g.n*sizeof(int),ctx.stream));
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceSelect::Flagged(tmp,bytes,sequence,flags,a.roots.data(),count.data(),g.n,ctx.stream);});
  Buffer<int> sizes(ctx,2*a.count);root_slots<<<blocks(a.count),256,0,ctx.stream>>>(a.roots.data(),a.count,a.slot.data(),nv.data(),ne.data(),sizes.data());
  BROOK_CUDA(cudaGetLastError());auto host=sizes.get(ctx,2*a.count);
  std::vector<long long> vs{0},es{0};std::vector<int> edge_counts;
  for(int i=0;i<a.count;++i) { vs.push_back(vs.back()+host[2*i]);es.push_back(es.back()+host[2*i+1]);edge_counts.push_back(host[2*i+1]); }
  a.n=int(vs.back());a.m=int(es.back());a.vs=Buffer<long long>(ctx,vs.size());a.es=Buffer<long long>(ctx,es.size());
  a.vs.set(ctx,vs.data(),vs.size());a.es.set(ctx,es.data(),es.size());a.ne=Buffer<int>(ctx,a.count);a.ne.set(ctx,edge_counts.data(),a.count);
  a.vertices=Buffer<int>(ctx,a.n);a.edges=Buffer<int>(ctx,a.m);
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceSelect::If(tmp,bytes,sequence,a.vertices.data(),count.data(),g.n,ActiveVertex{g.parent.data(),a.slot.data()},ctx.stream);});
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceSelect::If(tmp,bytes,sequence,a.edges.data(),count.data(),g.m,ActiveEdge{g.edges.data(),g.parent.data(),a.slot.data()},ctx.stream);});
  auto sort=[&](Buffer<int> &ids,int n,bool edge) {
    Buffer<unsigned> keys(ctx,n),sorted_keys(ctx,n);Buffer<int> sorted(ctx,n);
    arena_keys<<<blocks(n),256,0,ctx.stream>>>(ids.data(),n,g.parent.data(),a.slot.data(),g.edges.data(),keys.data(),edge);
    primitive([&](void *tmp,size_t &bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,keys.data(),sorted_keys.data(),ids.data(),sorted.data(),n,0,32,ctx.stream);});
    ctx.synchronize();ids=std::move(sorted);
  };
  sort(a.vertices,a.n,false);sort(a.edges,a.m,true);
  a.local=Buffer<int>(ctx,g.n);a.xyz=Buffer<float>(ctx,3*size_t(a.n));a.radii=Buffer<float>(ctx,a.n);
  a.component=Buffer<int>(ctx,a.m);a.edge_rows=Buffer<int2>(ctx,a.m);
  arena_vertices<<<blocks(a.n),256,0,ctx.stream>>>(a.vertices.data(),a.n,g.parent.data(),a.slot.data(),a.vs.data(),
    static_cast<const float*>(p.vertices.data()),static_cast<const float*>(p.radii.data()),a.local.data(),a.xyz.data(),a.radii.data());
  arena_edges<<<blocks(a.m),256,0,ctx.stream>>>(a.edges.data(),a.m,g.edges.data(),g.parent.data(),a.slot.data(),a.local.data(),a.component.data(),a.edge_rows.data());
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return a;
}
__global__ void untouched_keys(const int2 *edges,const int *parent,const int *slot,int m,unsigned long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m) {
    int2 e=edges[i];keys[i]=slot[parent[e.x]]<0 && e.x!=e.y?(static_cast<unsigned long long>(min(e.x,e.y))<<32)|unsigned(max(e.x,e.y)):~0ULL;
  }
}
__global__ void loop_keys(const int *selected,const int2 *edges,const int *comp,const int *ne,const long long *es,const long long *vs,
    const int *vertices,int m,unsigned long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m) return;int c=comp[i];if(i-es[c]>=ne[c]) return;
  int2 local=edges[i];unsigned a=unsigned(vertices[vs[c]+local.x]),b=unsigned(vertices[vs[c]+local.y]);
  if(a!=b) keys[selected[i]]=(static_cast<unsigned long long>(min(a,b))<<32)|max(a,b);
}
__global__ void edge_key_counts(const unsigned long long *keys,int m,const int *labels,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m) atomicAdd(counts+labels[keys[i]>>32],1);
}
__global__ void local_edge_keys(const unsigned long long *keys,int m,const int *labels,const long long *offsets,uint32_t *out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m) {
    unsigned a=keys[i]>>32,b=unsigned(keys[i]);auto base=offsets[labels[a]];
    out[2LL*i]=uint32_t(a-base);out[2LL*i+1]=uint32_t(b-base);
  }
}
PackedSkeletons edge_result(Context &ctx,const PackedSkeletons &p,const Graph &g,Buffer<unsigned long long> &keys,int count) {
  if(count==INT32_MAX) throw std::invalid_argument("edge sorting capacity exceeded");
  BROOK_CUDA(cudaMemsetAsync(keys.data()+count,0xff,8,ctx.stream));Buffer<unsigned long long> sorted(ctx,count+1);
  Buffer<int> selected(ctx,1);Buffer<uint8_t> scratch;
  auto primitive=[&](auto operation) {
    size_t bytes=0;BROOK_CUDA(operation(nullptr,bytes));if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(operation(scratch.data(),bytes));
  };
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceRadixSort::SortKeys(tmp,bytes,keys.data(),sorted.data(),count+1,0,64,ctx.stream);});
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceSelect::Unique(tmp,bytes,sorted.data(),keys.data(),selected.data(),count+1,ctx.stream);});
  int n=selected.get(ctx,1)[0]-1;Buffer<int> counts(ctx,g.k);counts.clear(ctx);
  auto e=allocate(ctx,{2,n,1},BROOK_U32);
  if(n) {
    edge_key_counts<<<blocks(n),256,0,ctx.stream>>>(keys.data(),n,g.vertex_label.data(),counts.data());
    local_edge_keys<<<blocks(n),256,0,ctx.stream>>>(keys.data(),n,g.vertex_label.data(),g.voff.data(),static_cast<uint32_t*>(e.data()));
  }
  BROOK_CUDA(cudaGetLastError());auto sizes=counts.get(ctx,g.k);
  PackedSkeletons out=p;out.edges=row_view(e,0,n,2);out.edge_offsets={0};
  for(auto size:sizes) out.edge_offsets.push_back(out.edge_offsets.back()+size);return out;
}
__global__ void keep_surviving_components(const int *parent,const int *slot,const int *ne,uint8_t *keep,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) { int c=slot[parent[i]];keep[i]=c<0 || ne[c]>0; }
}
__global__ void csr_entries(const int2 *edges,const int *component,const long long *vs,const long long *es,int m,
    unsigned *source,unsigned long long *values) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m) return;int c=component[i];int2 e=edges[i];
  unsigned row=unsigned(i-es[c]);
  source[i]=unsigned(vs[c]+e.x);values[i]=(static_cast<unsigned long long>(unsigned(e.y))<<32)|row;
  source[m+i]=unsigned(vs[c]+e.y);values[m+i]=(static_cast<unsigned long long>(unsigned(e.x))<<32)|row;
}
struct ArenaDegree { const int *vertices,*degree;__device__ long long operator()(int i) const { return degree[vertices[i]]; } };
__global__ void split_csr(const unsigned long long *values,int *dst,int *row,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) { dst[i]=int(values[i]>>32);row[i]=int(unsigned(values[i])); }
}
__global__ void fill_int(int *out,int n,int value) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[i]=value;
}
__global__ void tick_keep(const int *selected,const int *keep,uint8_t *out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[selected[i]]=keep[i]!=0;
}
__global__ void copy_selected_edges(const uint32_t *in,const uint8_t *keep,const int *rank,uint32_t *out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && keep[i]) {
    int j=rank[i]-1;out[2LL*j]=in[2LL*i];out[2LL*j+1]=in[2LL*i+1];
  }
}
PackedSkeletons select_edge_rows(Context &ctx,const PackedSkeletons &p,const Graph &g,const Buffer<uint8_t> &keep) {
  if(!g.m) return p;
  auto flags=thrust::make_transform_iterator(thrust::counting_iterator<int>(0),KeepValue{keep.data()});
  Buffer<int> rank(ctx,g.m),counts(ctx,g.k);counts.clear(ctx);size_t bytes=0;
  BROOK_CUDA(cub::DeviceScan::InclusiveSum(nullptr,bytes,flags,rank.data(),g.m,ctx.stream));Buffer<uint8_t> scratch(ctx,bytes);
  BROOK_CUDA(cub::DeviceScan::InclusiveSum(scratch.data(),bytes,flags,rank.data(),g.m,ctx.stream));
  selected_counts<<<blocks(g.m),256,0,ctx.stream>>>(keep.data(),g.edge_label.data(),g.m,counts.data());
  BROOK_CUDA(cudaGetLastError());auto sizes=counts.get(ctx,g.k);PackedSkeletons out=p;out.edge_offsets={0};
  for(auto size:sizes) out.edge_offsets.push_back(out.edge_offsets.back()+size);
  auto e=allocate(ctx,{2,out.edge_offsets.back(),1},BROOK_U32);
  copy_selected_edges<<<blocks(g.m),256,0,ctx.stream>>>(static_cast<const uint32_t*>(p.edges.data()),keep.data(),rank.data(),static_cast<uint32_t*>(e.data()),g.m);
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();out.edges=row_view(e,0,out.edge_offsets.back(),2);return out;
}
}
Array skeleton_components(Context &ctx,const PackedSkeletons &p) {
  ctx.activate();auto g=graph(ctx,p);auto out=allocate(ctx,{g.n,1,1},BROOK_I64);
  if(g.n) widen_parent<<<blocks(g.n),256,0,ctx.stream>>>(g.parent.data(),static_cast<long long*>(out.data()),g.n);
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return row_view(out,0,g.n,1);
}
PackedSkeletons select_vertices(Context &ctx,const PackedSkeletons &p,const Array &keep) {
  ctx.activate();auto g=graph(ctx,p);return select(ctx,p,keep,g);
}
PackedSkeletons prune_unused(Context &ctx,const PackedSkeletons &p) {
  ctx.activate();auto g=graph(ctx,p);auto keep=allocate(ctx,{g.n,1,1},BROOK_U8);
  if(!g.n) return p;
  Buffer<int> minimum(ctx,1);size_t bytes=0;
  BROOK_CUDA(cub::DeviceReduce::Min(nullptr,bytes,g.degree.data(),minimum.data(),g.n,ctx.stream));Buffer<uint8_t> scratch(ctx,bytes);
  BROOK_CUDA(cub::DeviceReduce::Min(scratch.data(),bytes,g.degree.data(),minimum.data(),g.n,ctx.stream));
  if(minimum.get(ctx,1)[0]>0) return p;
  if(g.n) degree_keep<<<blocks(g.n),256,0,ctx.stream>>>(g.degree.data(),static_cast<uint8_t*>(keep.data()),g.n);
  BROOK_CUDA(cudaGetLastError());return select(ctx,p,keep,g);
}
PackedSkeletons remove_dust(Context &ctx,const PackedSkeletons &p,double threshold) {
  ctx.activate();if(threshold==0 || p.radii.size()==0) return p;
  auto g=graph(ctx,p);Buffer<double> totals(ctx,g.n);Buffer<int> counts(ctx,g.n);totals.clear(ctx);counts.clear(ctx);
  if(g.m) lengths<<<blocks(g.m),256,0,ctx.stream>>>(static_cast<const float*>(p.vertices.data()),g.edges.data(),g.parent.data(),g.m,totals.data(),counts.data());
  Buffer<uint8_t> keep_root(ctx,g.n);
  keep_lengths<<<blocks(g.n),256,0,ctx.stream>>>(totals.data(),keep_root.data(),g.n,threshold);
  auto sequence=thrust::counting_iterator<int>(0);
  auto flags=thrust::make_transform_iterator(sequence,Uncertain{totals.data(),counts.data(),threshold});
  Buffer<int> selected(ctx,1);size_t bytes=0;
  BROOK_CUDA(cub::DeviceReduce::Sum(nullptr,bytes,flags,selected.data(),g.n,ctx.stream));
  Buffer<uint8_t> scratch(ctx,bytes);
  BROOK_CUDA(cub::DeviceReduce::Sum(scratch.data(),bytes,flags,selected.data(),g.n,ctx.stream));
  int count=selected.get(ctx,1)[0];
  if(count) {
    Buffer<int> indices(ctx,count);
    BROOK_CUDA(cub::DeviceSelect::Flagged(nullptr,bytes,sequence,flags,indices.data(),selected.data(),g.n,ctx.stream));
    if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(cub::DeviceSelect::Flagged(scratch.data(),bytes,sequence,flags,indices.data(),selected.data(),g.n,ctx.stream));
    auto roots=indices.get(ctx,count),parents=g.parent.get(ctx,g.n);auto edges=g.edges.get(ctx,g.m);
    std::vector<std::array<float,3>> vertices(g.n);p.vertices.copy_to_host(vertices.data(),vertices.size()*sizeof(vertices[0]));
    std::vector<uint8_t> decisions(count);std::vector<float> distances;
    for(int r=0;r<count;++r) {
      distances.clear();
      for(auto e:edges) if(parents[e.x]==roots[r]) {
        float x=vertices[e.y][0]-vertices[e.x][0],y=vertices[e.y][1]-vertices[e.x][1],z=vertices[e.y][2]-vertices[e.x][2];
        x=x*x;y=y*y;z=z*z;distances.push_back(std::sqrt((x+y)+z));
      }
      decisions[r]=double(pairwise_sum(distances.data(),distances.size()))>threshold;
    }
    Buffer<uint8_t> values(ctx,count);values.set(ctx,decisions.data(),count);
    set_decisions<<<blocks(count),256,0,ctx.stream>>>(keep_root.data(),indices.data(),values.data(),count);
    BROOK_CUDA(cudaGetLastError());ctx.synchronize();
  }
  auto keep=allocate(ctx,{g.n,1,1},BROOK_U8);
  broadcast_keep<<<blocks(g.n),256,0,ctx.stream>>>(keep_root.data(),g.parent.data(),static_cast<uint8_t*>(keep.data()),g.n);
  BROOK_CUDA(cudaGetLastError());return select(ctx,p,keep,g);
}
PackedSkeletons remove_loops(Context &ctx,const PackedSkeletons &p) {
  ctx.activate();if(!p.radii.size() || !p.edges.size()) return p;
  auto g=graph(ctx,p);auto a=arena(ctx,p,g,false);if(!a.count) return p;
  Buffer<int> rows(ctx,4*size_t(a.m)+4),ent(ctx,12*size_t(a.m)+12),nodes(ctx,11*size_t(a.n)+11),
    stack(ctx,3*(2*size_t(a.m)+a.count)+3),path(ctx,size_t(a.n)+a.count+1);
  cuda::post::remove_loops_k<<<(a.count+63)/64,64,0,ctx.stream>>>(a.count,a.vs.data(),a.es.data(),a.ne.data(),
    a.xyz.data(),a.radii.data(),reinterpret_cast<int*>(a.edge_rows.data()),rows.data(),ent.data(),nodes.data(),stack.data(),path.data());
  Buffer<unsigned long long> keys(ctx,size_t(g.m)+1);
  untouched_keys<<<blocks(g.m),256,0,ctx.stream>>>(g.edges.data(),g.parent.data(),a.slot.data(),g.m,keys.data());
  loop_keys<<<blocks(a.m),256,0,ctx.stream>>>(a.edges.data(),a.edge_rows.data(),a.component.data(),a.ne.data(),a.es.data(),a.vs.data(),a.vertices.data(),a.m,keys.data());
  BROOK_CUDA(cudaGetLastError());auto out=edge_result(ctx,p,g,keys,g.m);auto counts=a.ne.get(ctx,a.count);
  if(std::find(counts.begin(),counts.end(),0)!=counts.end()) {
    auto keep=allocate(ctx,{g.n,1,1},BROOK_U8);
    keep_surviving_components<<<blocks(g.n),256,0,ctx.stream>>>(g.parent.data(),a.slot.data(),a.ne.data(),static_cast<uint8_t*>(keep.data()),g.n);
    BROOK_CUDA(cudaGetLastError());auto next=graph(ctx,out);return select(ctx,out,keep,next);
  }
  return out;
}
PackedSkeletons remove_ticks(Context &ctx,const PackedSkeletons &p,double threshold) {
  ctx.activate();if(threshold==0 || !p.radii.size() || !p.edges.size()) return p;
  auto g=graph(ctx,p);auto a=arena(ctx,p,g,true);if(!a.count) return p;
  if(a.m>INT32_MAX/2) throw std::invalid_argument("tick adjacency exceeds int32 capacity");
  int twice=2*a.m;Buffer<unsigned> source(ctx,twice),sorted_source(ctx,twice);
  Buffer<unsigned long long> values(ctx,twice),sorted_values(ctx,twice);
  csr_entries<<<blocks(a.m),256,0,ctx.stream>>>(a.edge_rows.data(),a.component.data(),a.vs.data(),a.es.data(),a.m,source.data(),values.data());
  Buffer<uint8_t> scratch;
  auto primitive=[&](auto operation) {
    size_t bytes=0;BROOK_CUDA(operation(nullptr,bytes));if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(operation(scratch.data(),bytes));
  };
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,source.data(),sorted_source.data(),values.data(),sorted_values.data(),twice,0,32,ctx.stream);});
  Buffer<int> dst(ctx,twice),row(ctx,twice);split_csr<<<blocks(twice),256,0,ctx.stream>>>(sorted_values.data(),dst.data(),row.data(),twice);
  Buffer<long long> off(ctx,size_t(a.n)+1);BROOK_CUDA(cudaMemsetAsync(off.data(),0,8,ctx.stream));
  auto degrees=thrust::make_transform_iterator(thrust::counting_iterator<int>(0),ArenaDegree{a.vertices.data(),g.degree.data()});
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceScan::InclusiveSum(tmp,bytes,degrees,off.data()+1,a.n,ctx.stream);});
  auto i32=[&](size_t k){return Buffer<int>(ctx,k*size_t(a.n)+8);};
  auto stack=i32(4),bc=i32(1),su=i32(1),sv=i32(1),flag=i32(1),first=i32(1),last=i32(1),next=i32(1),dead=i32(1),heap=i32(14);
  Buffer<float> distance(ctx,size_t(a.n)+8);Buffer<double> length(ctx,size_t(a.n)+8),heap_length(ctx,2*size_t(a.n)+8);
  Buffer<int> chain(ctx,size_t(a.m)+8),keep(ctx,size_t(a.m)+1);fill_int<<<blocks(a.m+1),256,0,ctx.stream>>>(keep.data(),a.m+1,1);
  cuda::post::remove_ticks_k<<<(a.count+63)/64,64,0,ctx.stream>>>(a.count,a.vs.data(),a.es.data(),threshold,a.xyz.data(),off.data(),dst.data(),row.data(),
    stack.data(),distance.data(),bc.data(),su.data(),sv.data(),length.data(),flag.data(),first.data(),last.data(),next.data(),dead.data(),chain.data(),keep.data(),heap.data(),heap_length.data());
  Buffer<uint8_t> live(ctx,g.m);BROOK_CUDA(cudaMemsetAsync(live.data(),1,g.m,ctx.stream));
  tick_keep<<<blocks(a.m),256,0,ctx.stream>>>(a.edges.data(),keep.data(),live.data(),a.m);
  BROOK_CUDA(cudaGetLastError());return select_edge_rows(ctx,p,g,live);
}
namespace {
__global__ void join_statistics(const int *parent,const int *label,const float *radii,int n,int *counts,unsigned *rmax) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
  int l=label[i];if(parent[i]==i) atomicAdd(counts+l,1);
  float radius=radii[i];if(radius>0) atomicMax(rmax+l,__float_as_uint(radius));
}
struct MultiVertex { const int *label,*rank;__device__ int operator()(int i) const { return rank[label[i]]>=0; } };
struct MultiRoot { const int *label,*rank,*parent;__device__ int operator()(int i) const { return parent[i]==i && rank[label[i]]>=0; } };
__global__ void cell_keys(const int *selected,const int *order,int n,const float *vertices,const int *label,const int *rank,
    const double *bound,int axis,long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;int v=selected[order[i]],l=rank[label[v]];
  keys[i]=axis==3?l:static_cast<long long>(floor(double(vertices[3LL*v+axis])/bound[l]));
}
__global__ void ordered_cells(const int *selected,const int *order,int n,const float *vertices,const int *parent,const int *label,const int *rank,
    const double *bound,int *kl,long long *kx,long long *ky,long long *kz,float *xyz,long long *gid,long long *comp) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;int v=selected[order[i]],l=rank[label[v]];
  kl[i]=l;gid[i]=v;comp[i]=parent[v];
  for(int a=0;a<3;++a) xyz[3LL*i+a]=vertices[3LL*v+a];
  kx[i]=static_cast<long long>(floor(double(xyz[3LL*i])/bound[l]));
  ky[i]=static_cast<long long>(floor(double(xyz[3LL*i+1])/bound[l]));
  kz[i]=static_cast<long long>(floor(double(xyz[3LL*i+2])/bound[l]));
}
__global__ void identity_int(int *out,int n) { int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[i]=i; }
__global__ void identity_long(long long *out,int n) { int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[i]=i; }
// Median-partitioned trees, one per component. Nodes occupy their in-order
// positions; links permit stack-free traversal. No quadratic neighbor buffer.
struct JoinTrees {
  Buffer<int2> children,component_ranges;
  Buffer<int> parent,root_index;
  Buffer<uint8_t> axis;
};
__global__ void component_sort_keys(const int *selected,const int *parent,int n,long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) keys[i]=parent[selected[i]];
}
__global__ void component_indices(const int *roots,int count,int *index) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) index[roots[i]]=i;
}
__global__ void tree_ranges(const int *selected,const int *order,const int *parent,const int *root_index,int n,int2 *ranges) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
  int root=parent[selected[order[i]]],c=root_index[root];
  if(!i || parent[selected[order[i-1]]]!=root) ranges[c].x=i;
  if(i+1==n || parent[selected[order[i+1]]]!=root) ranges[c].y=i+1;
}
__global__ void tree_segments(const int *selected,const int *order,const int *parent,const int *root_index,
    const int2 *ranges,int n,int2 *segments,int *node_parent) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
  segments[i]=ranges[root_index[parent[selected[order[i]]]]];node_parent[i]=-1;
}
struct RangeSize { const int2 *ranges;__device__ int operator()(int i) const { return ranges[i].y-ranges[i].x; } };
__global__ void tree_keys(const int *selected,const int *order,const float *vertices,const int2 *segments,
    int n,int axis,unsigned long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
  float coordinate=vertices[3LL*selected[order[i]]+axis];
  unsigned bits=coordinate==0?0:__float_as_uint(coordinate);
  unsigned sortable=bits & 0x80000000U?~bits:bits^0x80000000U;
  keys[i]=(static_cast<unsigned long long>(segments[i].x)<<32)|sortable;
}
__global__ void tree_split(int2 *segments,int2 *children,int *parent,uint8_t *axes,int n,int axis) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
  int2 segment=segments[i];int middle=segment.x+(segment.y-segment.x)/2;
  if(i==middle) {
    // Settled singleton nodes retain the links/axis of their original split.
    if(children[i].x!=-2) return;
    int left=segment.x<middle?segment.x+(middle-segment.x)/2:-1;
    int right=middle+1<segment.y?middle+1+(segment.y-middle-1)/2:-1;
    children[i]={left,right};axes[i]=uint8_t(axis);
    if(left>=0) parent[left]=i;if(right>=0) parent[right]=i;
    segments[i]={i,i+1};
  } else if(i<middle) segments[i].y=middle;
  else segments[i].x=middle+1;
}
__global__ void tree_links_init(int2 *children,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) children[i]={-2,-2};
}
__global__ void ordered_tree(const int *selected,const int *order,int n,const float *vertices,const int *parent,
    const int *label,const int *rank,int *kl,float *xyz,long long *gid,long long *comp) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;int v=selected[order[i]];
  kl[i]=rank[label[v]];gid[i]=v;comp[i]=parent[v];
  for(int a=0;a<3;++a) xyz[3LL*i+a]=vertices[3LL*v+a];
}
__global__ void tree_scan(int n,int mode,const int *labels,const float *xyz,const long long *gid,const long long *comp,
    const double *bound,const long long *component_offsets,const int *root_index,const int2 *ranges,
    const int2 *children,const int *parent,const uint8_t *axis,long long *counts,const long long *offsets,
    long long *out_a,long long *out_b,double *out_distance,long long *out_v,long long *out_u) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
  int label=labels[i],component=root_index[comp[i]],emitted=0;
  long long base=mode?offsets[i]:0;
  const float *point=xyz+3LL*i;double reach=bound[label];
  for(int c=component+1;c<component_offsets[label+1];++c) {
    int2 range=ranges[c];int current=range.x+(range.y-range.x)/2,previous=-1,best_node=-1;
    double best_distance=reach;
    while(current>=0) {
      int p=parent[current];int2 links=children[current];
      const float *other=xyz+3LL*current;
      int a=axis[current];double delta=double(point[a])-double(other[a]);
      int near=delta<0?links.x:links.y,far=delta<0?links.y:links.x,next=p;
      if(previous==p) {
        double x=double(point[0])-double(other[0]),y=double(point[1])-double(other[1]),z=double(point[2])-double(other[2]);
        double d=sqrt((x*x+y*y)+z*z);
        if(d<reach && (d<best_distance || (d==best_distance && (best_node<0 || gid[current]<gid[best_node])))) {
          best_distance=d;best_node=current;
        }
        if(near>=0) next=near;
        else if(far>=0 && fabs(delta)<=nextafter(best_distance,CUDART_INF)) next=far;
      } else if(previous==near && far>=0 && fabs(delta)<=nextafter(best_distance,CUDART_INF)) next=far;
      previous=current;current=next;
    }
    if(best_node>=0) {
      if(mode) { long long j=base+emitted;out_a[j]=comp[i];out_b[j]=comp[best_node];out_distance[j]=best_distance;out_v[j]=i;out_u[j]=best_node; }
      ++emitted;
    }
  }
  if(!mode) counts[i]=emitted;
}
__global__ void pair_keys(const int *order,int n,const long long *a,const long long *b,const double *distance,
    const long long *v,const long long *u,const long long *gid,int axis,unsigned long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;int p=order[i];
  if(axis==0 || axis==1) { long long x=gid[v[p]],y=gid[u[p]];keys[i]=axis==0?max(x,y):min(x,y); }
  else if(axis==2) keys[i]=static_cast<unsigned long long>(__double_as_longlong(distance[p]));
  else keys[i]=axis==3?b[p]:a[p];
}
struct NewPair {
  const int *order;const long long *a,*b;
  __device__ int operator()(int i) const { if(!i) return 1;int x=order[i-1],y=order[i];return a[x]!=a[y] || b[x]!=b[y]; }
};
__global__ void root_positions(const int *roots,int count,const int *labels,const int *rank,const long long *cs,int *local) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) { int root=roots[i];local[root]=int(i-cs[rank[labels[root]]]); }
}
__global__ void compact_pairs(const int *selected,int count,const long long *a,const long long *b,const double *distance,
    const long long *v,const long long *u,const long long *gid,const int *local,const int *labels,const int *rank,
    int *cx,int *cy,double *d,long long *va,long long *vb,unsigned long long *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;int p=selected[i];
  cx[i]=local[a[p]];cy[i]=local[b[p]];d[i]=distance[p];va[i]=gid[v[p]];vb[i]=gid[u[p]];
  atomicAdd(counts+rank[labels[a[p]]],1ULL);
}
__global__ void join_output_keys(const int2 *edges,int m,const long long *a,const long long *b,int roots,
    const long long *cs,int labels,const int *made,unsigned long long *keys) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<m) { int2 e=edges[i];keys[i]=(static_cast<unsigned long long>(min(e.x,e.y))<<32)|unsigned(max(e.x,e.y)); }
  if(i<roots) {
    int l=group(cs,labels,i);unsigned long long key=~0ULL;
    if(i-cs[l]<made[l]) key=(static_cast<unsigned long long>(min(a[i],b[i]))<<32)|static_cast<unsigned long long>(max(a[i],b[i]));
    keys[m+i]=key;
  }
}
}
PackedSkeletons join_components(Context &ctx,const PackedSkeletons &input) {
  ctx.activate();auto p=prune_unused(ctx,input);if(!p.radii.size() || !p.edges.size()) return p;
  auto g=graph(ctx,p);Buffer<int> counts(ctx,g.k);counts.clear(ctx);Buffer<unsigned> maxima(ctx,g.k);maxima.clear(ctx);
  join_statistics<<<blocks(g.n),256,0,ctx.stream>>>(g.parent.data(),g.vertex_label.data(),static_cast<const float*>(p.radii.data()),g.n,counts.data(),maxima.data());
  BROOK_CUDA(cudaGetLastError());auto nc=counts.get(ctx,g.k);auto bits=maxima.get(ctx,g.k);
  std::vector<int> rank(g.k,-1);std::vector<double> radius,bound;std::vector<long long> cs{0};int ns=0;
  for(int i=0;i<g.k;++i) if(nc[i]>1) {
    rank[i]=int(radius.size());float r;std::memcpy(&r,&bits[i],4);double reach=std::max(2.0*double(r),0.0);
    radius.push_back(reach);bound.push_back(reach+0.000001);cs.push_back(cs.back()+nc[i]);ns+=int(p.vertex_offsets[i+1]-p.vertex_offsets[i]);
  }
  int nm=int(radius.size()),nr=int(cs.back());if(!nm) return p;
  Buffer<int> label_rank(ctx,g.k);label_rank.set(ctx,rank.data(),g.k);
  Buffer<double> bounds(ctx,nm),radii(ctx,nm);bounds.set(ctx,bound.data(),nm);radii.set(ctx,radius.data(),nm);
  Buffer<long long> component_offsets(ctx,cs.size());component_offsets.set(ctx,cs.data(),cs.size());
  Buffer<int> selected(ctx,ns),roots(ctx,nr),selected_count(ctx,1);Buffer<uint8_t> scratch;
  auto primitive=[&](auto operation) {
    size_t bytes=0;BROOK_CUDA(operation(nullptr,bytes));if(bytes>scratch.size()) scratch=Buffer<uint8_t>(ctx,bytes);
    BROOK_CUDA(operation(scratch.data(),bytes));
  };
  auto sequence=thrust::counting_iterator<int>(0);
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceSelect::If(tmp,bytes,sequence,selected.data(),selected_count.data(),g.n,MultiVertex{g.vertex_label.data(),label_rank.data()},ctx.stream);});
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceSelect::If(tmp,bytes,sequence,roots.data(),selected_count.data(),g.n,MultiRoot{g.vertex_label.data(),label_rank.data(),g.parent.data()},ctx.stream);});
  Buffer<int> order(ctx,ns),sorted_order(ctx,ns);Buffer<long long> key(ctx,ns),sorted_key(ctx,ns);
  identity_int<<<blocks(ns),256,0,ctx.stream>>>(order.data(),ns);
  const char *search=std::getenv("BROOK_JOIN_SEARCH");
  bool use_tree=search && std::strcmp(search,"kdtree")==0;
  JoinTrees trees;
  if(use_tree) {
    trees.children=Buffer<int2>(ctx,ns);trees.parent=Buffer<int>(ctx,ns);trees.axis=Buffer<uint8_t>(ctx,ns);
    trees.component_ranges=Buffer<int2>(ctx,nr);trees.root_index=Buffer<int>(ctx,g.n);
    component_indices<<<blocks(nr),256,0,ctx.stream>>>(roots.data(),nr,trees.root_index.data());
    component_sort_keys<<<blocks(ns),256,0,ctx.stream>>>(selected.data(),g.parent.data(),ns,key.data());
    primitive([&](void *tmp,size_t &bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,key.data(),sorted_key.data(),order.data(),sorted_order.data(),ns,0,64,ctx.stream);});
    std::swap(order,sorted_order);
    tree_ranges<<<blocks(ns),256,0,ctx.stream>>>(selected.data(),order.data(),g.parent.data(),trees.root_index.data(),ns,trees.component_ranges.data());
    Buffer<int2> segments(ctx,ns);
    tree_segments<<<blocks(ns),256,0,ctx.stream>>>(selected.data(),order.data(),g.parent.data(),trees.root_index.data(),trees.component_ranges.data(),ns,segments.data(),trees.parent.data());
    tree_links_init<<<blocks(ns),256,0,ctx.stream>>>(trees.children.data(),ns);
    auto sizes=thrust::make_transform_iterator(sequence,RangeSize{trees.component_ranges.data()});
    primitive([&](void *tmp,size_t &bytes){return cub::DeviceReduce::Max(tmp,bytes,sizes,selected_count.data(),nr,ctx.stream);});
    int maximum=selected_count.get(ctx,1)[0],level=0;
    for(int remaining=maximum;remaining;remaining>>=1,++level) {
      auto keys=reinterpret_cast<unsigned long long*>(key.data()),sorted_keys=reinterpret_cast<unsigned long long*>(sorted_key.data());
      tree_keys<<<blocks(ns),256,0,ctx.stream>>>(selected.data(),order.data(),static_cast<const float*>(p.vertices.data()),segments.data(),ns,level%3,keys);
      primitive([&](void *tmp,size_t &bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,keys,sorted_keys,order.data(),sorted_order.data(),ns,0,64,ctx.stream);});
      std::swap(order,sorted_order);
      tree_split<<<blocks(ns),256,0,ctx.stream>>>(segments.data(),trees.children.data(),trees.parent.data(),trees.axis.data(),ns,level%3);
    }
  } else {
    for(int axis:{2,1,0,3}) {
      cell_keys<<<blocks(ns),256,0,ctx.stream>>>(selected.data(),order.data(),ns,static_cast<const float*>(p.vertices.data()),g.vertex_label.data(),label_rank.data(),bounds.data(),axis,key.data());
      primitive([&](void *tmp,size_t &bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,key.data(),sorted_key.data(),order.data(),sorted_order.data(),ns,0,64,ctx.stream);});
      std::swap(order,sorted_order);
    }
  }
  Buffer<int> kl(ctx,ns);Buffer<long long> kx,ky,kz,gid(ctx,ns),comp(ctx,ns);Buffer<float> xyz(ctx,3*size_t(ns));
  if(use_tree) ordered_tree<<<blocks(ns),256,0,ctx.stream>>>(selected.data(),order.data(),ns,static_cast<const float*>(p.vertices.data()),g.parent.data(),g.vertex_label.data(),label_rank.data(),kl.data(),xyz.data(),gid.data(),comp.data());
  else {
    kx=Buffer<long long>(ctx,ns);ky=Buffer<long long>(ctx,ns);kz=Buffer<long long>(ctx,ns);
    ordered_cells<<<blocks(ns),256,0,ctx.stream>>>(selected.data(),order.data(),ns,static_cast<const float*>(p.vertices.data()),g.parent.data(),g.vertex_label.data(),label_rank.data(),
      bounds.data(),kl.data(),kx.data(),ky.data(),kz.data(),xyz.data(),gid.data(),comp.data());
  }
  // Construction buffers are dead before the candidate allocation peak.
  order={};sorted_order={};key={};sorted_key={};
  Buffer<long long> candidate_counts(ctx,ns),offs(ctx,size_t(ns)+1);candidate_counts.clear(ctx);BROOK_CUDA(cudaMemsetAsync(offs.data(),0,8,ctx.stream));
  auto scan=[&](int mode,long long *a,long long *b,double *d,long long *v,long long *u) {
    if(use_tree) tree_scan<<<(ns+127)/128,128,0,ctx.stream>>>(ns,mode,kl.data(),xyz.data(),gid.data(),comp.data(),bounds.data(),component_offsets.data(),trees.root_index.data(),trees.component_ranges.data(),
      trees.children.data(),trees.parent.data(),trees.axis.data(),candidate_counts.data(),offs.data(),a,b,d,v,u);
    else cuda::post::join_scan<<<(ns+127)/128,128,0,ctx.stream>>>(ns,mode,kl.data(),kx.data(),ky.data(),kz.data(),xyz.data(),gid.data(),comp.data(),bounds.data(),candidate_counts.data(),offs.data(),a,b,d,v,u);
  };
  scan(0,nullptr,nullptr,nullptr,nullptr,nullptr);
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceScan::InclusiveSum(tmp,bytes,candidate_counts.data(),offs.data()+1,ns,ctx.stream);});
  long long total=0;BROOK_CUDA(cudaMemcpyAsync(&total,offs.data()+ns,8,cudaMemcpyDeviceToHost,ctx.stream));ctx.synchronize();
  if(!total) return p;if(total>INT32_MAX) throw std::invalid_argument("join candidates exceed int32 capacity");int nt=int(total);
  Buffer<long long> a(ctx,nt),b(ctx,nt),v(ctx,nt),u(ctx,nt);Buffer<double> distance(ctx,nt);
  scan(1,a.data(),b.data(),distance.data(),v.data(),u.data());
  Buffer<int> pair_order(ctx,nt),pair_sorted(ctx,nt);Buffer<unsigned long long> pk(ctx,nt),sk(ctx,nt);
  identity_int<<<blocks(nt),256,0,ctx.stream>>>(pair_order.data(),nt);
  for(int axis=0;axis<5;++axis) {
    pair_keys<<<blocks(nt),256,0,ctx.stream>>>(pair_order.data(),nt,a.data(),b.data(),distance.data(),v.data(),u.data(),gid.data(),axis,pk.data());
    primitive([&](void *tmp,size_t &bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,pk.data(),sk.data(),pair_order.data(),pair_sorted.data(),nt,0,64,ctx.stream);});
    std::swap(pair_order,pair_sorted);
  }
  auto flags=thrust::make_transform_iterator(sequence,NewPair{pair_order.data(),a.data(),b.data()});
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceReduce::Sum(tmp,bytes,flags,selected_count.data(),nt,ctx.stream);});
  int np=selected_count.get(ctx,1)[0];Buffer<int> pairs(ctx,np);
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceSelect::Flagged(tmp,bytes,pair_order.data(),flags,pairs.data(),selected_count.data(),nt,ctx.stream);});
  Buffer<int> local(ctx,g.n),cx(ctx,np),cy(ctx,np);Buffer<double> d(ctx,np);Buffer<long long> va(ctx,np),vb(ctx,np);
  Buffer<unsigned long long> pair_counts(ctx,nm);pair_counts.clear(ctx);
  root_positions<<<blocks(nr),256,0,ctx.stream>>>(roots.data(),nr,g.vertex_label.data(),label_rank.data(),component_offsets.data(),local.data());
  compact_pairs<<<blocks(np),256,0,ctx.stream>>>(pairs.data(),np,a.data(),b.data(),distance.data(),v.data(),u.data(),gid.data(),local.data(),g.vertex_label.data(),label_rank.data(),
    cx.data(),cy.data(),d.data(),va.data(),vb.data(),pair_counts.data());
  BROOK_CUDA(cudaGetLastError());auto pc=pair_counts.get(ctx,nm);std::vector<long long> ps{0};for(auto n:pc) ps.push_back(ps.back()+n);
  Buffer<long long> pair_offsets(ctx,ps.size());pair_offsets.set(ctx,ps.data(),ps.size());
  Buffer<int> alive(ctx,np),pos(ctx,nr),slot(ctx,nr),made(ctx,nm);made.clear(ctx);fill_int<<<blocks(np),256,0,ctx.stream>>>(alive.data(),np,1);
  Buffer<long long> ident(ctx,g.n),out_a(ctx,nr),out_b(ctx,nr);identity_long<<<blocks(g.n),256,0,ctx.stream>>>(ident.data(),g.n);
  cuda::post::join_fuse<<<(nm+63)/64,64,0,ctx.stream>>>(nm,pair_offsets.data(),component_offsets.data(),radii.data(),cx.data(),cy.data(),d.data(),va.data(),vb.data(),
    ident.data(),static_cast<const float*>(p.radii.data()),alive.data(),pos.data(),slot.data(),out_a.data(),out_b.data(),made.data());
  primitive([&](void *tmp,size_t &bytes){return cub::DeviceReduce::Max(tmp,bytes,made.data(),selected_count.data(),nm,ctx.stream);});
  if(!selected_count.get(ctx,1)[0]) return p; // Preserve original edge order when no join is accepted.
  if(nr>INT32_MAX-g.m) throw std::invalid_argument("joined edge capacity exceeded");
  Buffer<unsigned long long> edges(ctx,size_t(g.m)+nr+1);
  join_output_keys<<<blocks(std::max(g.m,nr)),256,0,ctx.stream>>>(g.edges.data(),g.m,out_a.data(),out_b.data(),nr,component_offsets.data(),nm,made.data(),edges.data());
  BROOK_CUDA(cudaGetLastError());return edge_result(ctx,p,g,edges,g.m+nr);
}
PackedSkeletons postprocess(Context &ctx,const PackedSkeletons &p,double dust,double ticks) {
  auto out=remove_dust(ctx,p,dust);out=remove_loops(ctx,out);out=join_components(ctx,out);out=remove_ticks(ctx,out,ticks);return prune_unused(ctx,out);
}
}
