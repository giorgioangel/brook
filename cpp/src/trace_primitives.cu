// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "trace_primitives.hpp"
#include "cooperative.cuh"
#include "sssp_kernels.cuh"
#include "invalidation_kernels.cuh"
#include "fillvoids_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>

namespace brook {
namespace {
constexpr float infinity=std::numeric_limits<float>::infinity();
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
bool path_reuse_enabled() {
  const char *setting=std::getenv("BROOK_TRACE_PATH_REUSE");
  return !setting||std::string(setting)=="1";
}
constexpr size_t workspace_alignment=256;
size_t align_workspace(size_t offset) {
  if(offset>SIZE_MAX-(workspace_alignment-1)) throw std::invalid_argument("workspace alignment overflow");
  return (offset+workspace_alignment-1)&~(workspace_alignment-1);
}
std::shared_ptr<Allocation> workspace_arena(Context &ctx,int size,int lanes,bool railroad) {
  if(size<=0) throw std::invalid_argument("invalid workspace size");
  const char *setting=std::getenv("BROOK_TRACE_ARENA");
  if(setting && std::string(setting)!="1") return {};
  size_t n=size,l=lanes,r=railroad?n:0,total=0;
  // Exactly the constructor's buffer order, including optional railroad state.
  for(size_t bytes:{8*n,size_t{8},size_t{8},size_t{32},8*l,8*l,
                    4*n,4*n,size_t{32},4*n,4*n,4*n,4*n,4*n,4*n,4*r,size_t{32},4*n,size_t{4},
                    4*r,size_t{4},4*l,16*l}) {
    if(!bytes) continue;
    total=align_workspace(total);
    if(bytes>SIZE_MAX-total) throw std::invalid_argument("workspace arena overflow");
    total+=bytes;
  }
  return std::make_shared<Allocation>(total,ctx.device);
}
template<class T> Buffer<T> workspace_buffer(Context &ctx,const std::shared_ptr<Allocation> &arena,
                                            size_t &offset,size_t count) {
  if(!arena) return Buffer<T>(ctx,count);
  if(!count) return {};
  offset=align_workspace(offset);
  Buffer<T> result(arena,offset,count); // validates alignment and remaining capacity
  offset+=count*sizeof(T);
  return result;
}
void component_size(const Array &a) {
  if(!a.size() || a.size()>=size_t{1}<<31) throw std::invalid_argument("component must contain 1..2^31-1 voxels");
}
void require_type(const Array &a,brook_dtype d) {
  if(a.dtype!=d) throw std::invalid_argument("incorrect primitive array dtype");
}
void require_shape(const Array &a,const Array &b) {
  if(a.shape!=b.shape || a.device!=b.device) throw std::invalid_argument("primitive array shape/device mismatch");
}
const unsigned *graph_pointer(const Array *graph,const Array &a) {
  if(!graph) return nullptr;
  require_type(*graph,BROOK_U32); require_shape(a,*graph);
  return static_cast<const unsigned*>(graph->data());
}
__global__ void initialize_distance(float *d,long long n,long long source) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) d[i]=i==source?0.0f:__int_as_float(0x7f800000);
}
__global__ void seed_frontier(const float *d,int *a,int *inq,int *count,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n && isfinite(d[i])) { a[atomicAdd(count,1)]=i; inq[i]=1; }
}
__global__ void max_finite(const float *d,int n,unsigned long long *best) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n && isfinite(d[i]) && d[i]>=0) {
    auto key=(static_cast<unsigned long long>(__float_as_uint(d[i]))<<32) | (0xffffffffu-static_cast<unsigned>(i));
    atomicMax(best,key);
  }
}
__global__ void max_finite_reduced(const float *d,int n,unsigned long long *best) {
  unsigned long long value=0;
  for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<long long>(gridDim.x)*blockDim.x) {
    float distance=d[i];
    if(isfinite(distance) && distance>=0) {
      auto key=(static_cast<unsigned long long>(__float_as_uint(distance))<<32)|(0xffffffffu-static_cast<unsigned>(i));
      value=key>value?key:value;
    }
  }
  int lane=threadIdx.x&31;
  for(int offset=16;offset;offset>>=1) { auto other=__shfl_down_sync(0xffffffffu,value,offset);value=other>value?other:value; }
  __shared__ unsigned long long partial[8];
  if(!lane) partial[threadIdx.x>>5]=value;
  __syncthreads();
  if(threadIdx.x<32) {
    value=lane<8?partial[lane]:0;
    for(int offset=16;offset;offset>>=1) { auto other=__shfl_down_sync(0xffffffffu,value,offset);value=other>value?other:value; }
    if(!lane) atomicMax(best,value);
  }
}
void flood(Context &ctx,Array &dist,const Array *mask,const Array *field,const Buffer<float> &weights,
           int64_t source,bool rails,const Array *graph,size_t foreground_bound=0) {
  int n=static_cast<int>(dist.size());
  size_t capacity=foreground_bound?foreground_bound:size_t(n);
  if(!capacity || capacity>size_t(n) || (foreground_bound && (!mask || field)))
    throw std::invalid_argument("invalid foreground frontier bound");
  Buffer<int> a(ctx,capacity),b(ctx,capacity),inq(ctx,n),counts(ctx,4);
  inq.clear(ctx); counts.clear(ctx);
  seed_frontier<<<blocks(n),256,0,ctx.stream>>>(static_cast<const float*>(dist.data()),a.data(),inq.data(),counts.data(),n);
  BROOK_CUDA(cudaGetLastError());
  int count=counts.get(ctx,1)[0];
  auto shape=dist.shape;
  int grid=cooperative_grid(ctx,cuda::sssp_coop,capacity,"BROOK_FLOOD_GRID");note_grid(ctx,"grid_flood",grid);
  cooperative_launch(ctx,cuda::sssp_coop,grid,static_cast<float*>(dist.data()),
    mask?static_cast<const unsigned char*>(mask->data()):nullptr,
    field?static_cast<const float*>(field->data()):nullptr,weights.data(),graph_pointer(graph,dist),int(graph!=nullptr),
    int(field!=nullptr),int(rails),infinity,shape[0],shape[1],shape[2],a.data(),b.data(),counts.data(),inq.data(),count,n);
  if(counts.get(ctx,4)[3]) throw std::runtime_error("shortest-path flood did not converge");
}
__global__ void compute_pdrf(const float *dbf,const float *daf,float *out,int n,float m,float scale,
                           float inverse,float exponent,int squarings) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  float p=1.0f-dbf[i]*m;
  if(squarings>=0) for(int k=0;k<squarings;++k) p=p*p;
  else p=powf(p,exponent);
  p=p*scale;
  if(inverse!=0) p=p+daf[i]*inverse;
  out[i]=p;
}
__global__ void compute_normalized_pdrf(float *dbf,float *daf,float *out,int n,float m,float scale,
                                      float inverse,float exponent,int squarings) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  float boundary=dbf[i],distance=daf[i];
  // Preserve the normalized fields used later for radii, invalidation and targets.
  // Conditional stores as in Kimimaro's zero2inf/inf2zero, including signed-zero/NaN behavior.
  if(boundary==0) { boundary=__int_as_float(0x7f800000);dbf[i]=boundary; }
  if(isinf(distance)) { distance=0;daf[i]=distance; }
  float p=1.0f-boundary*m;
  if(squarings>=0) for(int k=0;k<squarings;++k) p=p*p;
  else p=powf(p,exponent);
  p=p*scale;
  if(inverse!=0) p=p+distance*inverse;
  out[i]=p;
}
}

DistanceField euclidean_distance_field(Context &ctx,const Array &mask,int64_t source,
                                      std::array<float,3> aniso,float free_radius,const Array *graph,bool reference_weights,
                                      size_t foreground_bound,const std::array<double,3> *step_spacing) {
  ctx.activate(); component_size(mask); require_type(mask,BROOK_U8);
  if(source<0 || static_cast<size_t>(source)>=mask.size()) throw std::invalid_argument("source outside volume");
  for(float a:aniso) if(!std::isfinite(a) || a<=0) throw std::invalid_argument("invalid anisotropy");
  auto dist=allocate(ctx,mask.shape,BROOK_F32);
  initialize_distance<<<blocks(mask.size()),256,0,ctx.stream>>>(static_cast<float*>(dist.data()),dist.size(),source);
  double x=step_spacing?(*step_spacing)[0]:aniso[0],y=step_spacing?(*step_spacing)[1]:aniso[1],z=step_spacing?(*step_spacing)[2]:aniso[2];
  std::vector<float> w={float(x),float(x),float(y),float(y),float(z),float(z)};
  for(double pair:{std::sqrt(x*x+y*y),std::sqrt(y*y+z*z),std::sqrt(x*x+z*z)})
    for(int k=0;k<4;++k) w.push_back(static_cast<float>(pair));
  for(int k=0;k<8;++k) w.push_back(static_cast<float>(std::sqrt(x*x+y*y+z*z)));
  if(reference_weights) {
    float a=aniso[0],b=aniso[1],c=aniso[2];
    float diagonal[4]={std::sqrt(a*a+b*b),std::sqrt(b*b+c*c),std::sqrt(a*a+c*c),std::sqrt((a*a+b*b)+c*c)};
    for(int k=6;k<26;++k) w[k]=diagonal[k<18?(k-6)/4:3];
  }
  Buffer<float> weights(ctx,26); weights.set(ctx,w.data(),26);
  if(free_radius>0) {
    auto s=mask.shape;
    cuda::edf_freespace_seed<<<blocks(mask.size()),256,0,ctx.stream>>>(static_cast<float*>(dist.data()),
      static_cast<const unsigned char*>(mask.data()),source,s[0],s[1],s[2],aniso[0],aniso[1],aniso[2],free_radius);
  }
  BROOK_CUDA(cudaGetLastError());
  // A supplied bound comes from a foreground reduction. The caller must seed
  // only foreground voxels: each next frontier then has at most that many
  // entries because inq grants a single append per voxel per round.
  flood(ctx,dist,&mask,nullptr,weights,source,false,graph,foreground_bound);
  Buffer<unsigned long long> best(ctx,1); best.clear(ctx);
  const char *reduction=std::getenv("BROOK_MAX_REDUCTION");
  if(!reduction || std::string(reduction)!="0")
    max_finite_reduced<<<std::min(blocks(dist.size()),1024u),256,0,ctx.stream>>>(static_cast<const float*>(dist.data()),dist.size(),best.data());
  else max_finite<<<blocks(dist.size()),256,0,ctx.stream>>>(static_cast<const float*>(dist.data()),dist.size(),best.data());
  BROOK_CUDA(cudaGetLastError());
  uint64_t key=best.get(ctx,1)[0];
  return {std::move(dist),static_cast<int64_t>(0xffffffffu-static_cast<unsigned>(key))};
}

std::pair<Array,Array> parental_field(Context &ctx,const Array &field,int64_t source,bool rails,const Array *graph) {
  ctx.activate(); component_size(field); require_type(field,BROOK_F32);
  if(source<0 || static_cast<size_t>(source)>=field.size()) throw std::invalid_argument("source outside volume");
  auto dist=allocate(ctx,field.shape,BROOK_F32);
  initialize_distance<<<blocks(field.size()),256,0,ctx.stream>>>(static_cast<float*>(dist.data()),dist.size(),source);
  BROOK_CUDA(cudaGetLastError());
  Buffer<float> unused;
  flood(ctx,dist,nullptr,&field,unused,source,rails,graph);
  auto parents=assign_parents(ctx,field,dist,source,rails,graph);
  return {std::move(dist),std::move(parents)};
}
Array assign_parents(Context &ctx,const Array &field,const Array &dist,int64_t source,bool rails,const Array *graph) {
  ctx.activate();component_size(field);require_type(field,BROOK_F32);require_type(dist,BROOK_F32);
  if(field.shape!=dist.shape || source<0 || static_cast<size_t>(source)>=field.size())
    throw std::invalid_argument("invalid parent assignment input");
  auto parents=allocate(ctx,field.shape,BROOK_I64);
  auto s=field.shape;
  cuda::vw_assign_parents<<<blocks(field.size()),256,0,ctx.stream>>>(static_cast<const float*>(dist.data()),
    static_cast<const float*>(field.data()),graph_pointer(graph,field),int(graph!=nullptr),
    static_cast<long long*>(parents.data()),s[0],s[1],s[2],source,int(rails));
  BROOK_CUDA(cudaGetLastError()); ctx.synchronize();
  return parents;
}

namespace {
template<bool Normalize,class ArrayType>
Array make_pdrf(Context &ctx,ArrayType &dbf,ArrayType &daf,double maximum,double max_daf,double scale,double exponent) {
  ctx.activate(); component_size(dbf); require_shape(dbf,daf); require_type(dbf,BROOK_F32); require_type(daf,BROOK_F32);
  float m=static_cast<float>(1.0/std::pow(maximum,1.01));
  int squarings=-1;
  if(exponent>0 && exponent<65536 && std::floor(exponent)==exponent && !(int(exponent)&(int(exponent)-1))) {
    squarings=0; for(int e=int(exponent);e>1;e>>=1) ++squarings;
  }
  auto out=allocate(ctx,dbf.shape,BROOK_F32);
  if constexpr(Normalize)
    compute_normalized_pdrf<<<blocks(dbf.size()),256,0,ctx.stream>>>(static_cast<float*>(dbf.data()),static_cast<float*>(daf.data()),
      static_cast<float*>(out.data()),dbf.size(),m,static_cast<float>(scale),max_daf?static_cast<float>(1.0/max_daf):0.0f,static_cast<float>(exponent),squarings);
  else
    compute_pdrf<<<blocks(dbf.size()),256,0,ctx.stream>>>(static_cast<const float*>(dbf.data()),static_cast<const float*>(daf.data()),
      static_cast<float*>(out.data()),dbf.size(),m,static_cast<float>(scale),max_daf?static_cast<float>(1.0/max_daf):0.0f,static_cast<float>(exponent),squarings);
  BROOK_CUDA(cudaGetLastError()); ctx.synchronize(); return out;
}

}
Array pdrf(Context &ctx,const Array &dbf,const Array &daf,double maximum,double max_daf,double scale,double exponent) {
  return make_pdrf<false>(ctx,dbf,daf,maximum,max_daf,scale,exponent);
}
Array normalize_pdrf(Context &ctx,Array &dbf,Array &daf,double maximum,double max_daf,double scale,double exponent) {
  return make_pdrf<true>(ctx,dbf,daf,maximum,max_daf,scale,exponent);
}

std::pair<Array,int> fill_voids(Context &ctx,const Array &mask) {
  ctx.activate(); require_type(mask,BROOK_U8);
  if(!mask.size()) return {mask,0};
  component_size(mask);
  int n=static_cast<int>(mask.size()); auto s=mask.shape;
  Buffer<int> vis(ctx,n),a(ctx,n),b(ctx,n),counts(ctx,8); vis.clear(ctx); counts.clear(ctx);
  auto fg=static_cast<const unsigned char*>(mask.data());
  cuda::fill_seed<<<blocks(n),256,0,ctx.stream>>>(fg,vis.data(),s[0],s[1],s[2],a.data(),counts.data());
  BROOK_CUDA(cudaGetLastError());
  int grid=cooperative_grid(ctx,cuda::fill_flood,n,"BROOK_FILL_GRID");note_grid(ctx,"grid_fill",grid);
  cooperative_launch(ctx,cuda::fill_flood,grid,fg,vis.data(),s[0],s[1],s[2],a.data(),b.data(),counts.data());
  auto out=allocate(ctx,s,BROOK_U8);
  cuda::fill_finish<<<blocks(n),256,0,ctx.stream>>>(fg,vis.data(),static_cast<long long>(n),static_cast<unsigned char*>(out.data()),counts.data());
  BROOK_CUDA(cudaGetLastError()); int count=counts.get(ctx,8)[4];
  return {std::move(out),count};
}

TraceWorkspace::TraceWorkspace(Context &ctx,int size,bool need_railroad)
 : n(size),lane_capacity(std::min(size,1<<18)),reuse_path(path_reuse_enabled()),
   arena(workspace_arena(ctx,n,lane_capacity,need_railroad||!reuse_path)),
   key(workspace_buffer<unsigned long long>(ctx,arena,arena_offset,n)),
   minfar(workspace_buffer<unsigned long long>(ctx,arena,arena_offset,1)),
   rail_best(workspace_buffer<unsigned long long>(ctx,arena,arena_offset,1)),
   stats(workspace_buffer<unsigned long long>(ctx,arena,arena_offset,4)),
   pba(workspace_buffer<unsigned long long>(ctx,arena,arena_offset,lane_capacity)),
   pbb(workspace_buffer<unsigned long long>(ctx,arena,arena_offset,lane_capacity)),
   snap(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   snapm(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   hdr(workspace_buffer<int>(ctx,arena,arena_offset,8)),
   wa(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   wb(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   fa(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   fb(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   inq(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   touched(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   infar(workspace_buffer<int>(ctx,arena,arena_offset,need_railroad||!reuse_path?n:0)),
   counts(workspace_buffer<int>(ctx,arena,arena_offset,8)),
   out(workspace_buffer<int>(ctx,arena,arena_offset,n)),
   out_len(workspace_buffer<int>(ctx,arena,arena_offset,1)),
   dist(workspace_buffer<float>(ctx,arena,arena_offset,need_railroad||!reuse_path?n:0)),
   threshold(workspace_buffer<float>(ctx,arena,arena_offset,1)),
   sdu(workspace_buffer<float>(ctx,arena,arena_offset,lane_capacity)),
   sxyz(workspace_buffer<long long>(ctx,arena,arena_offset,2*lane_capacity)) {
  if(size<=0) throw std::invalid_argument("invalid workspace size");
  if(arena && arena_offset!=arena->bytes) throw std::logic_error("workspace arena layout mismatch");
  BROOK_CUDA(cudaMemsetAsync(key.data(),0xff,n*sizeof(unsigned long long),ctx.stream));
  inq.clear(ctx); infar.clear(ctx); hdr.clear(ctx); counts.clear(ctx); out_len.clear(ctx); stats.clear(ctx);
  if(dist.size()) {
    initialize_distance<<<blocks(n),256,0,ctx.stream>>>(dist.data(),n,-1);
    BROOK_CUDA(cudaGetLastError());
  }
}

void TraceWorkspace::stage_path(Context &ctx,const std::vector<int64_t> &path) {
  if(path.empty()) return;
  if(path.size()>path_seeds.size()) {
    // All prior uses have completed before the host receives the next path.
    // Release first: growth must not retain both allocations at the peak.
    path_seeds={};
    path_seeds=Buffer<long long>(ctx,path.size());
  }
  staged_path.assign(path.begin(),path.end());
  path_seeds.set(ctx,staged_path.data(),staged_path.size());
}

std::vector<int64_t> backtrace(Context &ctx,const Array &parents,int64_t target,int64_t source) {
  component_size(parents); require_type(parents,BROOK_I64);
  if(target<0 || static_cast<size_t>(target)>=parents.size()) throw std::invalid_argument("target outside volume");
  Buffer<long long> out(ctx,parents.size()); Buffer<int> length(ctx,1); length.clear(ctx);
  cuda::backtrace<<<1,1,0,ctx.stream>>>(static_cast<const long long*>(parents.data()),target,source,out.data(),length.data(),parents.size());
  BROOK_CUDA(cudaGetLastError()); int n=length.get(ctx,1)[0];
  auto path=out.get(ctx,n); return {path.begin(),path.end()};
}

std::vector<int64_t> backtrace_workspace(Context &ctx,const Array &parents,int64_t target,int64_t source,TraceWorkspace &ws) {
  if(!ws.reuse_path) return backtrace(ctx,parents,target,source);
  component_size(parents);require_type(parents,BROOK_I64);
  if(ws.n!=parents.size() || target<0 || target>=ws.n) throw std::invalid_argument("backtrace workspace/target mismatch");
  // Component bounds prove every valid parent index fits int32. Reuse the
  // same output scratch as the railroad's distance-field backtrace.
  cuda::backtrace<<<1,1,0,ctx.stream>>>(static_cast<const long long*>(parents.data()),target,source,
                                     ws.out.data(),ws.out_len.data(),ws.n);
  BROOK_CUDA(cudaGetLastError());int length=ws.out_len.get(ctx,1)[0];
  auto path=ws.out.get(ctx,length);return {path.begin(),path.end()};
}

std::vector<int64_t> railroad(Context &ctx,const Array &field,int64_t target,TraceWorkspace &ws,const Array *graph) {
  ctx.activate(); component_size(field); require_type(field,BROOK_F32);
  if(ws.n!=field.size() || ws.dist.size()!=field.size() || ws.infar.size()!=field.size() || target<0 || target>=ws.n)
    throw std::invalid_argument("railroad workspace/target mismatch");
  float cost=0;
  BROOK_CUDA(cudaMemcpyAsync(&cost,static_cast<const float*>(field.data())+target,sizeof(float),cudaMemcpyDeviceToHost,ctx.stream));
  ctx.synchronize(); if(cost==0) return {target};
  auto s=field.shape;
  int grid=cooperative_grid(ctx,cuda::sssp_dstep,ws.n,"BROOK_DSTEP_GRID");note_grid(ctx,"grid_railroad",grid);
  cooperative_launch(ctx,cuda::sssp_dstep,grid,ws.dist.data(),static_cast<const float*>(field.data()),
    graph_pointer(graph,field),int(graph!=nullptr),s[0],s[1],s[2],infinity,ws.wa.data(),ws.wb.data(),ws.fa.data(),ws.fb.data(),
    ws.inq.data(),ws.infar.data(),ws.counts.data(),ws.threshold.data(),ws.minfar.data(),ws.rail_best.data(),ws.touched.data(),
    target,INT32_MAX,ws.n,0,ws.stats.data(),ws.lane_capacity,ws.sdu.data(),ws.sxyz.data(),2.0f,5e-4f);
  if(ws.counts.get(ctx,8)[4]!=1) {
    cuda::dstep_reset<<<256,256,0,ctx.stream>>>(ws.dist.data(),ws.inq.data(),ws.infar.data(),ws.touched.data(),ws.counts.data());
    BROOK_CUDA(cudaGetLastError()); ctx.synchronize();
    auto tree=parental_field(ctx,field,target,true,graph);
    std::vector<float> f(field.size()),d(field.size()); field.copy_to_host(f.data(),field.bytes()); tree.first.copy_to_host(d.data(),tree.first.bytes());
    // No rail is reachable when every rail is at infinite distance (a voxel graph, or an infinite
    // penalty such as a painted avocado cavity's, can cut a part of a component off): no path, as
    // dijkstra3d's railroad returns none. The flood above runs without a phase or round limit, so
    // it ends without a settled rail only then, and this branch returns {} (no join to re-pick).
    int64_t entry=-1; float best=infinity;
    for(int64_t i=0;i<ws.n;++i) if(f[i]==0 && d[i]<best) { entry=i;best=d[i]; }
    return entry<0?std::vector<int64_t>{}:backtrace_workspace(ctx,tree.second,entry,target,ws);
  }
  int64_t entry=static_cast<unsigned>(ws.rail_best.get(ctx,1)[0]);
  cuda::backtrace_by_dist_warp<<<1,32,0,ctx.stream>>>(ws.dist.data(),static_cast<const float*>(field.data()),graph_pointer(graph,field),
    int(graph!=nullptr),entry,target,1,s[0],s[1],s[2],ws.out.data(),ws.out_len.data(),ws.n);
  BROOK_CUDA(cudaGetLastError()); int length=ws.out_len.get(ctx,1)[0];
  auto p=ws.out.get(ctx,length);
  cuda::dstep_reset<<<256,256,0,ctx.stream>>>(ws.dist.data(),ws.inq.data(),ws.infar.data(),ws.touched.data(),ws.counts.data());
  BROOK_CUDA(cudaGetLastError()); ctx.synchronize(); return {p.begin(),p.end()};
}

int invalidate(Context &ctx,Array &mask,const Array &dbf,float scale,float constant,std::array<float,3> aniso,
               const std::vector<int64_t> &path,TraceWorkspace &ws,const Array *graph) {
  return invalidate_device_path(ctx,mask,dbf,scale,constant,aniso,path,ws,graph,nullptr);
}

int invalidate_device_path(Context &ctx,Array &mask,const Array &dbf,float scale,float constant,std::array<float,3> aniso,
               const std::vector<int64_t> &path,TraceWorkspace &ws,const Array *graph,const long long *device_path) {
  ctx.activate(); require_type(mask,BROOK_U8); require_type(dbf,BROOK_F32); require_shape(mask,dbf);
  if(path.empty()) return 0;
  if(ws.n!=mask.size()) throw std::invalid_argument("invalidation workspace mismatch");
  for(auto p:path) if(p<0 || p>=ws.n) throw std::invalid_argument("path outside component");
  Buffer<long long> seeds;
  std::vector<long long> p;
  if(!device_path) {
    seeds=Buffer<long long>(ctx,path.size());p.assign(path.begin(),path.end());seeds.set(ctx,p.data(),p.size());
    device_path=seeds.data();
  }
  auto s=mask.shape; float delta=*std::min_element(aniso.begin(),aniso.end());
  int grid=cooperative_grid(ctx,cuda::inval_coop,ws.n,"BROOK_INVAL_GRID");note_grid(ctx,"grid_invalidation",grid);
  cooperative_launch(ctx,cuda::inval_coop,grid,static_cast<unsigned char*>(mask.data()),static_cast<const float*>(dbf.data()),
    ws.key.data(),ws.snap.data(),ws.snapm.data(),graph_pointer(graph,mask),int(graph!=nullptr),s[0],s[1],s[2],
    aniso[0],aniso[1],aniso[2],scale,constant,delta,device_path,static_cast<int>(path.size()),
    ws.wa.data(),ws.wb.data(),ws.fa.data(),ws.fb.data(),ws.inq.data(),ws.touched.data(),ws.hdr.data(),ws.n,
    ws.lane_capacity,ws.sdu.data(),ws.sxyz.data(),ws.pba.data(),ws.pbb.data());
  int count=ws.hdr.get(ctx,8)[3];
  if(count<0) throw std::runtime_error("invalidation did not converge");
  return count;
}
}
