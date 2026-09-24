// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "batched.hpp"
#include "trace_primitives.hpp"
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_batched
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>
#include <cstdlib>
namespace cub=brook_cccl_batched::cub;
namespace thrust=brook_cccl_batched::thrust;

namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
template<class T> Buffer<T> upload_vector(Context &ctx,const std::vector<T> &values) {
  Buffer<T> out(ctx,values.size());out.set(ctx,values.data(),values.size());ctx.synchronize();return out;
}
__global__ void fill_distance(float *dist,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) dist[i]=__int_as_float(0x7f800000);
}
void reset(Context &ctx,Array &dist) {
  fill_distance<<<blocks(dist.size()),256,0,ctx.stream>>>(static_cast<float*>(dist.data()),int(dist.size()));
  BROOK_CUDA(cudaGetLastError());
}
__global__ void seed(float *dist,const int *sources,int count,int *inq,bool keep_distance) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) { int v=sources[i];if(!keep_distance) dist[v]=0;if(inq) inq[v]=1; }
}
struct Finite {
  const float *field;
  __device__ bool operator()(int i) const { return isfinite(field[i]); }
};
struct FlagValue { __device__ int operator()(uint8_t value) const { return int(value); } };
struct SumCount { double sum;unsigned long long count; };
struct FiniteValue {
  __device__ SumCount operator()(float value) const { return isfinite(value)?SumCount{double(value),1}:SumCount{0,0}; }
};
struct AddSumCount {
  __device__ SumCount operator()(SumCount a,SumCount b) const { return {a.sum+b.sum,a.count+b.count}; }
};
double mean_finite(Context &ctx,const Array &field) {
  using Iterator=thrust::transform_iterator<FiniteValue,const float*>;
  Iterator input(static_cast<const float*>(field.data()),FiniteValue{});
  Buffer<SumCount> result(ctx,1);size_t bytes=0;
  BROOK_CUDA(cub::DeviceReduce::Reduce(nullptr,bytes,input,result.data(),int(field.size()),AddSumCount{},SumCount{0,0},ctx.stream));
  Buffer<uint8_t> scratch(ctx,bytes);
  BROOK_CUDA(cub::DeviceReduce::Reduce(scratch.data(),bytes,input,result.data(),int(field.size()),AddSumCount{},SumCount{0,0},ctx.stream));
  auto value=result.get(ctx,1)[0];return value.count?value.sum/double(value.count):0;
}
template<class Label> __global__ void gather_labels(const Label *labels,const int *indices,int *out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[i]=int(labels[indices[i]]);
}
std::vector<int64_t> find_soma_roots(Context &ctx,const Array &labels,const Array &dbf,const std::vector<float> &maximum,
                                    const std::vector<uint8_t> &soma,const std::vector<Box> &boxes) {
  auto module=batched_module(labels.dtype);
  auto dmax=upload_vector(ctx,maximum);
  auto flags=upload_vector(ctx,soma);
  Buffer<uint8_t> mark(ctx,labels.size());
  module.get("dbf_maxima_lab").launch(ctx,blocks(labels.size()),256,dbf.data(),labels.data(),dmax.data(),flags.data(),int64_t(labels.size()),mark.data());
  Buffer<int> count(ctx,1);size_t scratch_bytes=0;
  thrust::transform_iterator<FlagValue,const uint8_t*> integers(mark.data(),FlagValue{});
  BROOK_CUDA(cub::DeviceReduce::Sum(nullptr,scratch_bytes,integers,count.data(),int(labels.size()),ctx.stream));
  Buffer<uint8_t> scratch(ctx,scratch_bytes);
  BROOK_CUDA(cub::DeviceReduce::Sum(scratch.data(),scratch_bytes,integers,count.data(),int(labels.size()),ctx.stream));
  int total=count.get(ctx,1)[0];
  std::vector<int64_t> roots(maximum.size(),-1);
  if(!total) return roots;
  Buffer<int> indices(ctx,total),owners(ctx,total);
  thrust::counting_iterator<int> sequence(0);
  scratch_bytes=0;
  BROOK_CUDA(cub::DeviceSelect::Flagged(nullptr,scratch_bytes,sequence,mark.data(),indices.data(),count.data(),int(labels.size()),ctx.stream));
  scratch=Buffer<uint8_t>(ctx,scratch_bytes);
  BROOK_CUDA(cub::DeviceSelect::Flagged(scratch.data(),scratch_bytes,sequence,mark.data(),indices.data(),count.data(),int(labels.size()),ctx.stream));
  switch(labels.dtype) {
    case BROOK_U8:gather_labels<<<blocks(total),256,0,ctx.stream>>>(static_cast<const uint8_t*>(labels.data()),indices.data(),owners.data(),total);break;
    case BROOK_U16:gather_labels<<<blocks(total),256,0,ctx.stream>>>(static_cast<const uint16_t*>(labels.data()),indices.data(),owners.data(),total);break;
    case BROOK_U32:gather_labels<<<blocks(total),256,0,ctx.stream>>>(static_cast<const uint32_t*>(labels.data()),indices.data(),owners.data(),total);break;
    case BROOK_U64:gather_labels<<<blocks(total),256,0,ctx.stream>>>(static_cast<const uint64_t*>(labels.data()),indices.data(),owners.data(),total);break;
    default:throw std::invalid_argument("invalid soma component dtype");
  }
  BROOK_CUDA(cudaGetLastError());auto ids=indices.get(ctx,total),labs=owners.get(ctx,total);
  struct Candidate { int label,index;Point point; };
  std::vector<Candidate> candidates;
  for(int i=0;i<total;++i) {
    auto p=unravel(ids[i],labels.shape);for(int k=0;k<3;++k) p[k]-=boxes[labs[i]].lo[k];
    candidates.push_back({labs[i],ids[i],p});
  }
  std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return std::tie(a.label,a.point)<std::tie(b.label,b.point);});
  for(size_t begin=0;begin<candidates.size();) {
    size_t end=begin;Point sum={0,0,0};
    while(end<candidates.size() && candidates[end].label==candidates[begin].label) {
      for(int k=0;k<3;++k) sum[k]+=candidates[end].point[k];++end;
    }
    std::array<float,3> center;for(int k=0;k<3;++k) center[k]=float(double(sum[k])/double(end-begin));
    double best=INFINITY;
    for(size_t i=begin;i<end;++i) {
      double cost=0;for(int k=0;k<3;++k) { double d=double(candidates[i].point[k])-center[k];cost+=d*d; }
      if(cost<best) { best=cost;roots[candidates[i].label]=candidates[i].index; }
    }
    begin=end;
  }
  return roots;
}
}

BatchedFlooder::BatchedFlooder(Context &ctx,const Array &labels,const Array *graph)
 :ctx_(ctx),labels_(labels),module_(batched_module(labels.dtype)),a_(ctx,labels.size()),b_(ctx,labels.size()),
  inq_(ctx,labels.size()),snapshot_(ctx,std::min(labels.size(),size_t{1}<<18)),
  coordinates_(ctx,std::min(labels.size(),size_t{1}<<18)),stats_(ctx,2),neighbors_(ctx,labels.size()) {
  if(labels.size()>=size_t{1}<<31) throw std::invalid_argument("batched volume exceeds int32 capacity");
  if(graph && (graph->dtype!=BROOK_U32 || graph->shape!=labels.shape || graph->device!=labels.device))
    throw std::invalid_argument("batched flood graph must be uint32 and match the labels");
  inq_.clear(ctx);auto s=labels.shape;
  const void *vg=graph?graph->data():nullptr;
  if(labels.size()) module_.get("nbr_mask").launch(ctx,blocks(labels.size()),256,labels.data(),vg,s[0],s[1],s[2],neighbors_.data());
}
void BatchedFlooder::flood(Array &dist,const std::vector<int64_t> &sources,int mode,const Buffer<float> &weights,
                          const Array *field,bool seeded,bool banded) {
  const int n=int(labels_.size());auto s=labels_.shape;
  Buffer<int> counts(ctx_,8);counts.clear(ctx_);int count=0;
  if(seeded) {
    size_t bytes=0;thrust::counting_iterator<int> indices(0);Finite test{static_cast<const float*>(dist.data())};
    BROOK_CUDA(cub::DeviceSelect::If(nullptr,bytes,indices,a_.data(),counts.data(),n,test,ctx_.stream));
    Buffer<uint8_t> scratch(ctx_,bytes);
    BROOK_CUDA(cub::DeviceSelect::If(scratch.data(),bytes,indices,a_.data(),counts.data(),n,test,ctx_.stream));
    count=counts.get(ctx_,1)[0];
  } else {
    std::vector<int> roots(sources.begin(),sources.end());count=int(roots.size());a_.set(ctx_,roots.data(),roots.size());ctx_.synchronize();
  }
  if(!count) return;
  seed<<<blocks(count),256,0,ctx_.stream>>>(static_cast<float*>(dist.data()),a_.data(),count,inq_.data(),seeded);
  BROOK_CUDA(cudaGetLastError());
  const auto &fn=module_.get(banded?"flood_lab_nf":"flood_lab");
  int grid=fn.grid(ctx_,n,"BROOK_BATCH_GRID");note_grid(ctx_,"grid_batch_flood",grid);
  const void *cost=field?field->data():nullptr;
  if(!banded) fn.launch(ctx_,grid,128,dist.data(),labels_.data(),cost,weights.data(),mode,s[0],s[1],s[2],
    a_.data(),b_.data(),counts.data(),inq_.data(),count,n,int(snapshot_.size()),snapshot_.data(),coordinates_.data(),stats_.data(),neighbors_.data(),1);
  else {
    float first=float(mean_finite(ctx_,*field));
    auto tbuf=upload_vector(ctx_,std::vector<float>{0,first,1.1f,first});
    Buffer<int> minimum(ctx_,16),publication(ctx_,16);minimum.clear(ctx_);publication.clear(ctx_);
    fn.launch(ctx_,grid,128,dist.data(),labels_.data(),cost,weights.data(),mode,s[0],s[1],s[2],
      a_.data(),b_.data(),counts.data(),inq_.data(),count,1<<30,4096,tbuf.data(),minimum.data(),publication.data(),
      int(snapshot_.size()),snapshot_.data(),coordinates_.data(),stats_.data(),neighbors_.data(),1);
    ctx_.synchronize();
  }
  if(counts.get(ctx_,8)[3]) throw std::runtime_error("batched shortest-path flood did not converge");
}
SegmentedMaximum BatchedFlooder::argmax(const Array &values,size_t count) {
  Buffer<unsigned long long> best(ctx_,count+1);best.clear(ctx_);
  module_.get("seg_argmax").launch(ctx_,blocks(labels_.size()),256,values.data(),labels_.data(),int64_t(labels_.size()),best.data());
  auto keys=best.get(ctx_,count+1);SegmentedMaximum result;result.value.resize(count+1);result.index.resize(count+1);
  for(size_t i=0;i<=count;++i) {
    uint32_t bits=uint32_t(keys[i]>>32);std::memcpy(&result.value[i],&bits,sizeof(bits));
    result.index[i]=keys[i]?int64_t(0xffffffffu-uint32_t(keys[i])):-1;
  }
  return result;
}

std::pair<Array,Array> parental_field_banded(Context &ctx,const Array &mask,const Array &field,int64_t source) {
  ctx.activate();
  if(mask.dtype!=BROOK_U8 || field.dtype!=BROOK_F32 || mask.shape!=field.shape ||
      source<0 || static_cast<size_t>(source)>=field.size()) throw std::invalid_argument("invalid banded parent field input");
  BatchedFlooder flooder(ctx,mask);
  auto distance=allocate(ctx,field.shape,BROOK_F32);reset(ctx,distance);
  Buffer<float> weights;
  flooder.flood(distance,{source},1,weights,&field,false,true);
  auto parents=assign_parents(ctx,field,distance,source);
  return {std::move(distance),std::move(parents)};
}

BatchedPreparation prepare_batched(Context &ctx,const Components &cc,const Array &dbf,const std::vector<Box> &boxes,
    const std::vector<std::vector<Point>> &borders,const SkeletonizeOptions &options,const Array *graph) {
  const auto &labels=cc.labels;const auto &p=options.teasar;auto shape=labels.shape;
  size_t k=cc.mapping.size(),n=labels.size();BatchedFlooder flooder(ctx,labels,graph);auto module=batched_module(labels.dtype);
  BatchedPreparation out;out.dbf_max=flooder.argmax(dbf,k-1).value;out.active.resize(k,0);out.soma.resize(k,0);out.root.resize(k,-1);
  if(graph) out.graph=*graph;
  const void *vg=graph?graph->data():nullptr;const int has_vg=graph?1:0;
  out.filled.resize(k,0);
  const char *reuse=std::getenv("BROOK_VOID_REUSE");
  const bool keep_filled=!reuse || std::string(reuse)!="0";
  // Keep the one-byte-per-input-voxel budget. Keep only masks
  // whose positive fill result will otherwise be recomputed in preparation.
  size_t retained_masks=0;
  for(size_t id=1;id<k;++id) {
    const auto &box=boxes[id];auto ext=box.hi;for(int a=0;a<3;++a) ext[a]-=box.lo[a]-1;
    if(double(box.count)<=options.dust || volume_size(ext)<=1) continue;
    out.active[id]=1;
    if(double(out.dbf_max[id])>p.soma_detection) {
      auto crop=crop_component(ctx,labels,dbf,id,box);auto filled=fill_voids(ctx,crop.first);
      out.filled[id]=filled.second;
      if(keep_filled && filled.second && filled.first.size()<=n-retained_masks) {
        retained_masks+=filled.first.size();out.filled_masks.emplace(id,std::move(filled.first));
      }
      if(filled.second) out.active[id]=0;
      else out.soma[id]=double(out.dbf_max[id])>p.soma_acceptance;
    }
  }
  bool has_somata=std::any_of(out.soma.begin(),out.soma.end(),[](auto v){return v;});
  if(has_somata) out.root=find_soma_roots(ctx,labels,dbf,out.dbf_max,out.soma,boxes);
  std::vector<int64_t> search,search_sources;
  for(size_t id=1;id<k;++id) if(out.active[id] && !out.soma[id]) {
    if(!borders[id].empty()) out.root[id]=flatten(borders[id].back(),shape);
    else { search.push_back(id);search_sources.push_back(cc.roots[id-1]); }
  }
  double x=options.anisotropy[0],y=options.anisotropy[1],z=options.anisotropy[2];
  std::vector<float> w={float(x),float(x),float(y),float(y),float(z),float(z)};
  for(double d:{std::sqrt(x*x+y*y),std::sqrt(y*y+z*z),std::sqrt(x*x+z*z)}) for(int i=0;i<4;++i) w.push_back(float(d));
  for(int i=0;i<8;++i) w.push_back(float(std::sqrt(x*x+y*y+z*z)));
  auto weights=upload_vector(ctx,w);
  out.daf=allocate(ctx,shape,BROOK_F32);reset(ctx,out.daf);
  if(!search.empty()) {
    flooder.flood(out.daf,search_sources,0,weights);auto maxima=flooder.argmax(out.daf,k-1);
    for(auto id:search) out.root[id]=maxima.index[id];reset(ctx,out.daf);
  }
  std::vector<int64_t> roots;for(size_t id=1;id<k;++id) if(out.active[id]) roots.push_back(out.root[id]);
  if(has_somata) {
    auto all_roots=upload_vector(ctx,std::vector<int>(roots.begin(),roots.end()));
    seed<<<blocks(roots.size()),256,0,ctx.stream>>>(static_cast<float*>(out.daf.data()),all_roots.data(),int(roots.size()),nullptr,false);
    std::vector<long long> soma_roots(out.root.begin(),out.root.end());
    for(size_t id=0;id<k;++id) if(!out.soma[id]) soma_roots[id]=-1;
    auto root_table=upload_vector(ctx,soma_roots);
    auto radius=upload_vector(ctx,out.dbf_max);
    module.get("freespace_seed_lab").launch(ctx,blocks(n),256,out.daf.data(),labels.data(),root_table.data(),radius.data(),
      shape[0],shape[1],shape[2],options.anisotropy[0],options.anisotropy[1],options.anisotropy[2]);
    flooder.flood(out.daf,{},0,weights,nullptr,true);
  } else flooder.flood(out.daf,roots,0,weights);
  auto maxima=flooder.argmax(out.daf,k-1);out.max_daf=std::move(maxima.value);out.target=std::move(maxima.index);
  double exponent=p.pdrf_exponent;
  if(exponent>0 && exponent<65536 && exponent==std::floor(exponent) && !(int(exponent)&(int(exponent)-1))) {
    std::vector<float> m(k,0),inv(k,0);std::vector<uint8_t> use(k,0);
    for(size_t id=1;id<k;++id) if(out.active[id]) {
      m[id]=float(1.0/std::pow(double(out.dbf_max[id]),1.01));
      if(out.max_daf[id]!=0) { inv[id]=float(1.0/double(out.max_daf[id]));use[id]=2; }else use[id]=1;
    }
    auto dm=upload_vector(ctx,m),di=upload_vector(ctx,inv);
    auto du=upload_vector(ctx,use);
    int squarings=0;for(int e=int(exponent);e>1;e>>=1) ++squarings;
    out.pdrf=allocate(ctx,shape,BROOK_F32);
    module.get("pdrf_lab").launch(ctx,blocks(n),256,dbf.data(),out.daf.data(),labels.data(),dm.data(),di.data(),du.data(),squarings,float(p.pdrf_scale),int64_t(n),out.pdrf.data());
    if(!options.fix_branching) {
      auto distance=allocate(ctx,shape,BROOK_F32);reset(ctx,distance);
      flooder.flood(distance,roots,1,weights,&out.pdrf,false,true);
      out.parents=allocate(ctx,shape,BROOK_I32);
      module.get("parents_lab").launch(ctx,blocks(n),256,distance.data(),labels.data(),du.data(),vg,has_vg,shape[0],shape[1],shape[2],out.parents.data());
      ctx.synchronize();out.pdrf={};
    }
  }
  ctx.synchronize();return out;
}
}
