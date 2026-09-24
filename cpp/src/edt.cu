// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include "edt_kernels.cuh"
#include "edt_bounds.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace brook {
namespace {
template<class Label> __global__ void finish(const Label *labels, float *distance, long long n) {
  const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) distance[i]=labels[i]==0 ? 0.0f : sqrtf(distance[i]);
}
template<class Label> Array run(Context &ctx, const Array &labels, std::array<float,3> aniso, bool bb,int dimensions,int64_t segment=0) {
  auto out=allocate(ctx,labels.shape,BROOK_F32);
  auto s=labels.shape;
  float *dist=static_cast<float*>(out.data());
  const Label *lab=static_cast<const Label*>(labels.data());
  std::vector<float> host_table(s[0]+2,0.0f);
  for(size_t k=1;k<host_table.size();++k) host_table[k]=host_table[k-1]+aniso[0];
  Buffer<float> table(ctx,host_table.size()); table.set(ctx,host_table.data(),host_table.size());
  const char *index_mode=std::getenv("BROOK_EDT_INDEX32");
  const bool narrow=edt_index32_volume_fits(out.size()) &&
    (!index_mode || std::string(index_mode)!="0");
  if(narrow) cuda::edt_pass1_x<1,int><<<static_cast<unsigned>((s[1]*s[2]+255)/256),256,0,ctx.stream>>>(
    dist,lab,s[0],s[1],s[2],table.data(),0,bb);
  else cuda::edt_pass1_x<<<static_cast<unsigned>((s[1]*s[2]+255)/256),256,0,ctx.stream>>>(
    dist,lab,s[0],s[1],s[2],table.data(),0,bb);
  BROOK_CUDA(cudaGetLastError());
  size_t free=0,total=0; BROOK_CUDA(cudaMemGetInfo(&free,&total));
  const size_t budget=std::max(free/64,size_t{16}<<20);
  const char *workspace_mode=std::getenv("BROOK_EDT_WORKSPACE");
  bool shared_workspace=!workspace_mode || std::string(workspace_mode)!="0";
  size_t capacity=0;
  if(shared_workspace) for(int axis=1;axis<dimensions;++axis) {
    const long long segs=axis==2 && segment>0 && dimensions==3?s[2]/segment:1;
    long long pitch=axis==2 && segs>1?segment+3:s[axis]+3,nlines=axis==1?s[0]*s[2]:s[0]*s[1]*segs;
    long long batch=std::max(1LL,std::min(nlines,static_cast<long long>(budget/(16*pitch))));
    capacity=std::max(capacity,size_t(batch*pitch));
  }
  // One call owns this workspace; both axis passes use it in stream order.
  // V/H/Z require 4/4/8 bytes each. Z's offset is naturally eight-byte aligned.
  Buffer<uint8_t> workspace;
  if(capacity) workspace=Buffer<uint8_t>(ctx,capacity*16);
  const long long segments=segment>0 && dimensions==3?s[2]/segment:1;   // z segments (a sample stack)
  for(int axis=1;axis<dimensions;++axis) {
    long long pitch=axis==2 && segments>1?segment+3:s[axis]+3;
    long long nlines=axis==1 ? s[0]*s[2] : s[0]*s[1]*segments;
    long long batch=std::max(1LL,std::min(nlines,static_cast<long long>(budget/(16*pitch))));
    Buffer<int> v;Buffer<float> h;Buffer<double> z;
    int *vp=nullptr;float *hp=nullptr;double *zp=nullptr;
    if(shared_workspace) {
      vp=reinterpret_cast<int*>(workspace.data());hp=reinterpret_cast<float*>(workspace.data()+capacity*4);
      zp=reinterpret_cast<double*>(workspace.data()+capacity*8);
    } else {
      v=Buffer<int>(ctx,batch*pitch);h=Buffer<float>(ctx,batch*pitch);z=Buffer<double>(ctx,batch*pitch);
      vp=v.data();hp=h.data();zp=z.data();
    }
    for(long long start=0;start<nlines;start+=batch) {
      long long count=std::min(batch,nlines-start);
      const long long seg=axis==2 && segments>1?segment:0;
      if(narrow && edt_index32_scratch_fits(pitch,count))
        cuda::edt_pass_fh<1,int><<<static_cast<unsigned>((count+255)/256),256,0,ctx.stream>>>(
          dist,lab,vp,hp,zp,s[0],s[1],s[2],axis,aniso[axis],nullptr,bb,pitch,start,count,int(seg));
      else cuda::edt_pass_fh<<<static_cast<unsigned>((count+255)/256),256,0,ctx.stream>>>(
        dist,lab,vp,hp,zp,s[0],s[1],s[2],axis,aniso[axis],nullptr,bb,pitch,start,count,seg);
      BROOK_CUDA(cudaGetLastError());
    }
    if(!shared_workspace) ctx.synchronize();
  }
  finish<<<static_cast<unsigned>((out.size()+255)/256),256,0,ctx.stream>>>(lab,dist,static_cast<long long>(out.size()));
  BROOK_CUDA(cudaGetLastError());
  ctx.synchronize();
  return out;
}
}
Array edt(Context &ctx, const Array &labels, std::array<float,3> aniso, bool bb,int dimensions,int64_t segment) {
  ctx.activate();
  if(segment<0 || (segment>0 && (dimensions!=3 || labels.shape[2]%segment))) throw std::invalid_argument("EDT segment must divide the z extent");
  if(dimensions<1 || dimensions>3) throw std::invalid_argument("EDT dimensions must be 1, 2 or 3");
  for(int axis=dimensions;axis<3;++axis) if(labels.shape[axis]!=1) throw std::invalid_argument("non-singleton omitted EDT axis");
  for(float a:aniso) if(!std::isfinite(a) || a<=0) throw std::invalid_argument("anisotropy must be finite and positive");
  if(!labels.size()) return allocate(ctx,labels.shape,BROOK_F32);
  for(auto d:labels.shape) if(d>INT32_MAX-3) throw std::invalid_argument("EDT axis exceeds int32 envelope capacity");
  switch(labels.dtype) {
    case BROOK_U8: return run<uint8_t>(ctx,labels,aniso,bb,dimensions,segment);
    case BROOK_U16: return run<uint16_t>(ctx,labels,aniso,bb,dimensions,segment);
    case BROOK_U32: return run<uint32_t>(ctx,labels,aniso,bb,dimensions,segment);
    case BROOK_U64: return run<uint64_t>(ctx,labels,aniso,bb,dimensions,segment);
    case BROOK_I8: return run<int8_t>(ctx,labels,aniso,bb,dimensions,segment);
    case BROOK_I16: return run<int16_t>(ctx,labels,aniso,bb,dimensions,segment);
    case BROOK_I32: return run<int32_t>(ctx,labels,aniso,bb,dimensions,segment);
    case BROOK_I64: return run<int64_t>(ctx,labels,aniso,bb,dimensions,segment);
    default: throw std::invalid_argument("EDT requires integer labels");
  }
}
}
