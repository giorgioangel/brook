// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "voxel_graph.hpp"
#include "edt_kernels.cuh"
#include "graph.hpp"
#include <algorithm>
#include <cmath>

namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }

// The edt package's graph-aware EDT places blocked face interfaces at half a voxel.
// Values at original voxel centers are sampled from a doubled binary grid.
// This accessor offers the SAME grid without allocating an eight-volume mask.
template<class Label> struct GraphMask {
  const Label *labels;
  const uint32_t *graph;
  long long sx,sy,sz;
  bool border;
  long long doubled_segment;   // > 0: a stack of samples along z, each 2 * segment doubled positions deep
  __device__ __forceinline__ uint8_t operator[](long long index) const {
    long long x,y,z;cuda::brook_unravel(index,2*sx,2*sy,x,y,z);
    if(border && (x==2*sx-1 || y==2*sy-1 || (doubled_segment>0?z%doubled_segment==doubled_segment-1:z==2*sz-1))) return 0;
    long long source=(x>>1)+sx*((y>>1)+sy*(z>>1));
    if(!(labels[source]>0)) return 0;
    int sub=int((x&1)|((y&1)<<1)|((z&1)<<2));
    if(sub==1) return (graph[source]&1u)!=0;
    if(sub==2) return (graph[source]&4u)!=0;
    if(sub==4) return (graph[source]&16u)!=0;
    return 1;
  }
};
template<class Label> __global__ void expand(GraphMask<Label> mask,uint8_t *out,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n) out[i]=mask[i];
}
template<class Label> __global__ void sample(const float *squared,const Label *labels,float *out,
    long long sx,long long sy,long long sz) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=sx*sy*sz) return;
  long long x,y,z;cuda::brook_unravel(i,sx,sy,x,y,z);
  out[i]=labels[i]>0?sqrtf(squared[2*x+4*sx*y+8*sx*sy*z]):0.0f;
}
// Retain only even x coordinates after pass 1, and only evaluate even y/z
// coordinates in later passes. All lower-envelope inputs and rounding stay
// unchanged. Y writes in-place to its own lines; Z samples and square-roots
// directly into the final output. The scratch field needs 4N instead of 8N.
template<class Label,bool Final> struct CompactField {
  float *squared,*result;
  const Label *labels;
  long long sx,sy;
  struct Reference {
    float *input,*output;
    bool live;
    __device__ operator float() const { return *input; }
    __device__ void operator=(float value) const {
      if constexpr(Final) *output=live?sqrtf(value):0.0f;
      else *output=value;
    }
  };
  __device__ Reference operator[](long long index) const {
    long long x,y,z;cuda::brook_unravel(index,2*sx,2*sy,x,y,z);
    long long source=(x>>1)+sx*(y+2*sy*z);
    if constexpr(Final) {
      long long dest=(x>>1)+sx*((y>>1)+sy*(z>>1));
      return {squared+source,result+dest,labels[dest]>0};
    } else return {squared+source,squared+source,true};
  }
};
template<class Label> Array run(Context &ctx,const Array &labels,const Array &graph,std::array<float,3> aniso,bool border,bool capture,bool fused,bool compact,int64_t segment) {
  auto s=labels.shape;
  std::array<int64_t,3> doubled={2*s[0],2*s[1],2*s[2]};
  size_t n=volume_size(doubled);
  // A stack of samples: the doubled z lines are cut per sample, so the interface behind a sample's
  // last slice keeps its own (open or black) border and no parabola of the next sample is seen.
  const long long zseg=segment>0?2*segment:0,zsegments=zseg?doubled[2]/zseg:1;
  auto result=allocate(ctx,s,BROOK_F32);Buffer<float> squared(ctx,compact?n/2:n);
  Buffer<uint8_t> materialized;if(!fused && !compact) materialized=Buffer<uint8_t>(ctx,n);
  std::vector<float> table(size_t(doubled[0])+2,0);
  for(size_t i=1;i<table.size();++i) table[i]=table[i-1]+aniso[0]*0.5f;
  Buffer<float> accumulated(ctx,table.size());accumulated.set(ctx,table.data(),table.size());
  size_t available=0,total=0;BROOK_CUDA(cudaMemGetInfo(&available,&total));
  size_t budget=std::max(available/64,size_t{16}<<20);
  long long batch[3]={0,0,0},lines[3]={0,doubled[0]*doubled[2],doubled[0]*doubled[1]*zsegments},pitches[3]={0,doubled[1]+3,(zseg?zseg:doubled[2])+3};
  if(compact) { lines[1]/=2;lines[2]/=4; }
  size_t elements=0;
  for(int axis=1;axis<3;++axis) {
    long long pitch=pitches[axis];
    batch[axis]=std::max(1LL,std::min(lines[axis],static_cast<long long>(budget/(16*pitch))));
    elements=std::max(elements,size_t(batch[axis]*pitch));
  }
  Buffer<int> v(ctx,elements);Buffer<float> h(ctx,elements);Buffer<double> z(ctx,elements);
  GraphMask<Label> mask{static_cast<const Label*>(labels.data()),static_cast<const uint32_t*>(graph.data()),s[0],s[1],s[2],border,zseg};
  ctx.stats["graph_edt_work_bytes"]=squared.size()*sizeof(float)+materialized.size()+accumulated.size()*sizeof(float)+elements*16;
  auto passes=[&](auto input) {
    cuda::edt_pass1_x<<<blocks(doubled[1]*doubled[2]),256,0,ctx.stream>>>(squared.data(),input,
      doubled[0],doubled[1],doubled[2],accumulated.data(),0,border);
    BROOK_CUDA(cudaGetLastError());
    for(int axis=1;axis<3;++axis) {
      for(long long first=0;first<lines[axis];first+=batch[axis]) {
        long long count=std::min(batch[axis],lines[axis]-first);
        cuda::edt_pass_fh<<<blocks(count),256,0,ctx.stream>>>(squared.data(),input,v.data(),h.data(),z.data(),
          doubled[0],doubled[1],doubled[2],axis,aniso[axis]*0.5f,nullptr,border,pitches[axis],first,count,axis==2?zseg:0LL);
        BROOK_CUDA(cudaGetLastError());
      }
    }
    sample<<<blocks(labels.size()),256,0,ctx.stream>>>(squared.data(),static_cast<const Label*>(labels.data()),static_cast<float*>(result.data()),s[0],s[1],s[2]);
    BROOK_CUDA(cudaGetLastError());
  };
  auto body=[&] {
    if(compact) {
      CompactField<Label,false> intermediate{squared.data(),nullptr,static_cast<const Label*>(labels.data()),s[0],s[1]};
      CompactField<Label,true> final{squared.data(),static_cast<float*>(result.data()),static_cast<const Label*>(labels.data()),s[0],s[1]};
      cuda::edt_pass1_x<2><<<blocks(doubled[1]*doubled[2]),256,0,ctx.stream>>>(intermediate,mask,
        doubled[0],doubled[1],doubled[2],accumulated.data(),0,border);
      BROOK_CUDA(cudaGetLastError());
      auto axis_pass=[&](int axis,auto distance) {
        for(long long first=0;first<lines[axis];first+=batch[axis]) {
          long long count=std::min(batch[axis],lines[axis]-first);
          cuda::edt_pass_fh<2><<<blocks(count),256,0,ctx.stream>>>(distance,mask,v.data(),h.data(),z.data(),
            doubled[0],doubled[1],doubled[2],axis,aniso[axis]*0.5f,nullptr,border,pitches[axis],first,count,axis==2?zseg:0LL);
          BROOK_CUDA(cudaGetLastError());
        }
      };
      axis_pass(1,intermediate);axis_pass(2,final);
    }
    else if(fused) passes(mask);
    else { expand<<<blocks(n),256,0,ctx.stream>>>(mask,materialized.data(),int64_t(n));BROOK_CUDA(cudaGetLastError());passes(materialized.data()); }
  };
  if(capture) { FixedGraph captured(ctx);captured.capture(body);captured.run(); }
  else { body();ctx.synchronize(); }
  return result;
}
}
Array graph_edt(Context &ctx,const Array &labels,const Array &input_graph,std::array<float,3> aniso,bool border,bool capture,bool fused,bool compact,int64_t segment) {
  ctx.activate();
  auto graph=graph_for_trace(ctx,input_graph);
  if(labels.shape!=graph.shape || graph.dtype!=BROOK_U32) throw std::invalid_argument("voxel_graph must be uint32 and match the volume");
  for(float a:aniso) if(!std::isfinite(a) || a<=0) throw std::invalid_argument("anisotropy must be finite and positive");
  if(segment<0 || (segment>0 && labels.shape[2]%segment)) throw std::invalid_argument("graph EDT segment must divide the z extent");
  if(!labels.size()) return allocate(ctx,labels.shape,BROOK_F32);
  for(auto s:labels.shape) if(s>(INT32_MAX-3)/2) throw std::invalid_argument("graph EDT axis exceeds envelope capacity");
  switch(labels.dtype) {
    case BROOK_U8:return run<uint8_t>(ctx,labels,graph,aniso,border,capture,fused,compact,segment);
    case BROOK_U16:return run<uint16_t>(ctx,labels,graph,aniso,border,capture,fused,compact,segment);
    case BROOK_U32:return run<uint32_t>(ctx,labels,graph,aniso,border,capture,fused,compact,segment);
    case BROOK_U64:return run<uint64_t>(ctx,labels,graph,aniso,border,capture,fused,compact,segment);
    default:throw std::invalid_argument("graph EDT expects unsigned component labels");
  }
}
}
