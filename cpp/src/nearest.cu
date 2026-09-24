// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "nearest.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <type_traits>
#define THRUST_CUB_WRAPPED_NAMESPACE brook_cccl_nearest
#include <cub/block/block_reduce.cuh>

namespace brook {
namespace {
template<int A,int B,int C,class Visit> void ordered_box(const brook_volume &volume,
    const std::array<int64_t,3> &lo,const std::array<int64_t,3> &hi,Visit &visit) {
  std::array<int64_t,3> point;
  for(point[A]=lo[A];point[A]<hi[A];++point[A]) for(point[B]=lo[B];point[B]<hi[B];++point[B]) {
    const char *row=static_cast<const char*>(volume.data)+point[A]*volume.strides[A]+point[B]*volume.strides[B];
    for(point[C]=lo[C];point[C]<hi[C];++point[C]) visit(point[0],point[1],point[2],row+point[C]*volume.strides[C]);
  }
}
template<class Visit> void visit_box(const brook_volume &volume,int order,
    const std::array<int64_t,3> &lo,const std::array<int64_t,3> &hi,Visit visit) {
  switch(order) {
    case 1:ordered_box<0,1,2>(volume,lo,hi,visit);break;
    case 2:ordered_box<0,2,1>(volume,lo,hi,visit);break;
    case 3:ordered_box<1,0,2>(volume,lo,hi,visit);break;
    case 5:ordered_box<1,2,0>(volume,lo,hi,visit);break;
    case 6:ordered_box<2,0,1>(volume,lo,hi,visit);break;
    case 7:ordered_box<2,1,0>(volume,lo,hi,visit);break;
  }
}
struct Candidate { double distance;int64_t index; };
struct Best {
  __host__ __device__ Candidate operator()(Candidate a,Candidate b) const {
    if(a.index==INT64_MAX) return b;if(b.index==INT64_MAX) return a;
    bool an=isnan(a.distance),bn=isnan(b.distance);
    if(an!=bn) return an?a:b;
    if(!an && a.distance!=b.distance) return a.distance<b.distance?a:b;
    return a.index<b.index?a:b;
  }
};
template<class T> __host__ __device__ bool matches(T value,const brook_label_query &q) {
  if(q.kind==BROOK_QUERY_FLOAT) return double(value)==q.floating_label;
  if(q.kind==BROOK_QUERY_WEAK_FLOAT) {
    if constexpr(std::is_floating_point_v<T>) return value==T(q.floating_label);
    else return double(value)==q.floating_label;
  }
  if(q.kind==BROOK_QUERY_UNSIGNED) {
    if constexpr(std::is_floating_point_v<T>) return value==T(q.integer_label);
    else return value>=0 && uint64_t(value)==q.integer_label;
  }
  if(q.kind==BROOK_QUERY_SIGNED) {
    if constexpr(std::is_floating_point_v<T>) return value==T(static_cast<int64_t>(q.integer_label));
    else if constexpr(std::is_signed_v<T>) return int64_t(value)==static_cast<int64_t>(q.integer_label);
    else return static_cast<int64_t>(q.integer_label)>=0 && uint64_t(value)==q.integer_label;
  }
  return false;
}
template<class T> __global__ void nearest_tiles(const T *labels,int64_t n,int64_t sx,int64_t sy,int64_t sz,
    const brook_label_query *queries,int query_base,int tiles,Candidate *partial) {
  int qid=query_base+blockIdx.y;auto q=queries[qid];
  Candidate best{0,INT64_MAX};Best choose;
  for(int64_t i=q.kind==BROOK_QUERY_NO_MATCH?n:int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(tiles)*blockDim.x) {
    if(!matches(labels[i],q)) continue;
    int64_t x=i%sx,y=(i/sx)%sy,z=i/(sx*sy);
    double dx=double(x)-q.centroid[0],dy=double(y)-q.centroid[1],dz=double(z)-q.centroid[2];
    best=choose(best,{sqrt((dx*dx+dy*dy)+dz*dz),(x*sy+y)*sz+z});
  }
  using Reduce=brook_cccl_nearest::cub::BlockReduce<Candidate,256>;
  __shared__ typename Reduce::TempStorage temporary;
  best=Reduce(temporary).Reduce(best,choose);
  if(!threadIdx.x) partial[int64_t(qid)*tiles+blockIdx.x]=best;
}
__global__ void nearest_finish(const Candidate *partial,int tiles,int count,int64_t *output) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count) return;
  Candidate best{0,INT64_MAX};Best choose;
  for(int i=0;i<tiles;++i) best=choose(best,partial[int64_t(q)*tiles+i]);
  output[q]=best.index;
}
}
std::vector<std::array<int64_t,3>> nearest_label_voxels(Context &ctx,const Array &labels,
    const std::vector<brook_label_query> &queries) {
  if(labels.device!=ctx.device || labels.logical_shape) throw std::invalid_argument("nearest-label search requires a volume on the context device");
  if(queries.size()>INT32_MAX) throw std::invalid_argument("too many nearest-label queries");
  std::vector<std::array<int64_t,3>> result(queries.size(),{-1,-1,-1});
  if(queries.empty() || !labels.size()) return result;
  for(const auto &q:queries) if(q.kind>BROOK_QUERY_WEAK_FLOAT) throw std::invalid_argument("invalid label query kind");
  int count=int(queries.size()),tiles=int(std::min<size_t>(128,(labels.size()+255)/256));
  Buffer<brook_label_query> input(ctx,queries.size());input.set(ctx,queries.data(),queries.size());
  Buffer<Candidate> partial(ctx,size_t(count)*tiles);Buffer<int64_t> indices(ctx,count);
  auto launch=[&](auto type) {
    using T=decltype(type);
    for(int first=0;first<count;) {
      int chunk=std::min(count-first,65535);
      nearest_tiles<<<dim3(tiles,chunk),256,0,ctx.stream>>>(static_cast<const T*>(labels.data()),labels.size(),labels.shape[0],labels.shape[1],labels.shape[2],input.data(),first,tiles,partial.data());
      first+=chunk;
    }
  };
  switch(labels.dtype) {
    case BROOK_U8:launch(uint8_t{});break;case BROOK_U16:launch(uint16_t{});break;
    case BROOK_U32:launch(uint32_t{});break;case BROOK_U64:launch(uint64_t{});break;
    case BROOK_I8:launch(int8_t{});break;case BROOK_I16:launch(int16_t{});break;
    case BROOK_I32:launch(int32_t{});break;case BROOK_I64:launch(int64_t{});break;
    case BROOK_F32:launch(float{});break;case BROOK_F64:launch(double{});break;
    default:throw std::invalid_argument("invalid label dtype");
  }
  nearest_finish<<<(count+255LL)/256,256,0,ctx.stream>>>(partial.data(),tiles,count,indices.data());
  BROOK_CUDA(cudaGetLastError());auto host=indices.get(ctx,count);
  for(int i=0;i<count;++i) if(host[i]!=INT64_MAX) {
    int64_t yz=labels.shape[1]*labels.shape[2];
    result[i]={host[i]/yz,(host[i]/labels.shape[2])%labels.shape[1],host[i]%labels.shape[2]};
  }
  return result;
}
std::vector<std::array<int64_t,3>> nearest_label_voxels_host(const brook_volume &labels,
    const std::vector<brook_label_query> &queries,bool bounded,Context *fallback) {
  if(labels.struct_size<sizeof(labels) || labels.abi_version!=BROOK_ABI_VERSION) throw std::invalid_argument("incompatible label volume descriptor");
  size_t n=volume_size({labels.shape[0],labels.shape[1],labels.shape[2]});
  dtype_size(labels.dtype);
  if(queries.size()>INT32_MAX) throw std::invalid_argument("too many nearest-label queries");
  for(const auto &q:queries) if(q.kind>BROOK_QUERY_WEAK_FLOAT) throw std::invalid_argument("invalid label query kind");
  if(n && !labels.data) throw std::invalid_argument("null label data");
  if(labels.memory!=BROOK_HOST) throw std::invalid_argument("host nearest-label search requires host memory");
  if(queries.empty()) return {};
  std::vector<std::array<int64_t,3>> result(queries.size(),{-1,-1,-1});
  if(!n) return result;
  std::array<int,3> axes{0,1,2};
  auto magnitude=[&](int a) { auto value=uint64_t(labels.strides[a]);return labels.strides[a]<0?uint64_t(0)-value:value; };
  std::stable_sort(axes.begin(),axes.end(),[&](int a,int b) {
    if((labels.shape[a]==1)!=(labels.shape[b]==1)) return labels.shape[a]==1;
    return magnitude(a)>magnitude(b);
  });
  int order=3*axes[0]+axes[1];
  auto search=[&](auto type) {
    using T=decltype(type);
    std::set<std::pair<uint32_t,uint64_t>> absent;
    std::map<std::pair<uint32_t,uint64_t>,std::vector<size_t>> deferred;
    for(size_t qi=0;qi<queries.size();++qi) {
      const auto &q=queries[qi];if(q.kind==BROOK_QUERY_NO_MATCH) continue;
      uint64_t value_key=0;
      if(q.kind==BROOK_QUERY_FLOAT || q.kind==BROOK_QUERY_WEAK_FLOAT) std::memcpy(&value_key,&q.floating_label,8);
      else value_key=q.integer_label;
      auto pattern=std::make_pair(q.kind,value_key);if(absent.count(pattern)) continue;
      Candidate best{0,INT64_MAX};Best choose;
      auto scan=[&](const std::array<int64_t,3> &lo,const std::array<int64_t,3> &hi) {
        visit_box(labels,order,lo,hi,[&](int64_t x,int64_t y,int64_t z,const char *p) {
          T value;std::memcpy(&value,p,sizeof(T));if(!matches(value,q)) return;
          double dx=double(x)-q.centroid[0],dy=double(y)-q.centroid[1],dz=double(z)-q.centroid[2];
          best=choose(best,{std::sqrt((dx*dx+dy*dy)+dz*dz),(x*labels.shape[1]+y)*labels.shape[2]+z});
        });
      };
      std::array<int64_t,3> lo{0,0,0},hi{labels.shape[0],labels.shape[1],labels.shape[2]};
      bool finite=std::isfinite(q.centroid[0]) && std::isfinite(q.centroid[1]) && std::isfinite(q.centroid[2]);
      if(bounded && finite) {
        std::array<int64_t,3> near_lo,near_hi;
        for(int axis=0;axis<3;++axis) {
          int64_t center=int64_t(std::max(0.0,std::min(double(labels.shape[axis]-1),std::floor(q.centroid[axis]))));
          near_lo[axis]=std::max<int64_t>(0,center-2);near_hi[axis]=std::min(labels.shape[axis],center+3);
        }
        scan(near_lo,near_hi);
        if(best.index!=INT64_MAX && std::isfinite(best.distance)) {
          // Any equal-or-closer point lies in this box. Round outwards and
          // retain a full extra voxel on each face to cover endpoint rounding.
          double radius=std::nextafter(best.distance,INFINITY);
          for(int axis=0;axis<3;++axis) {
            lo[axis]=int64_t(std::max(0.0,std::min(double(labels.shape[axis]),std::floor(q.centroid[axis]-radius)-1)));
            hi[axis]=int64_t(std::max(0.0,std::min(double(labels.shape[axis]),std::ceil(q.centroid[axis]+radius)+2)));
          }
          if(size_t(hi[0]-lo[0])*size_t(hi[1]-lo[1])*size_t(hi[2]-lo[2])>4096) {
            std::array<int64_t,3> anchor;
            for(int a=0;a<3;++a) {
              anchor[a]=int64_t(std::max(0.0,std::min(double(labels.shape[a]-1),std::floor(q.centroid[a]))));
              if(anchor[a]+1<labels.shape[a] && std::fabs(double(anchor[a]+1)-q.centroid[a])<std::fabs(double(anchor[a])-q.centroid[a])) ++anchor[a];
            }
            // Other coordinates at their independently nearest voxels give
            // an exact floating-point lower bound. Monotonic binary searches
            // retain every point whose computed norm can equal or beat best,
            // including ties created by rounding at very distant centroids.
            auto admissible=[&](int a,int64_t coordinate) {
              auto point=anchor;point[a]=coordinate;
              double dx=double(point[0])-q.centroid[0],dy=double(point[1])-q.centroid[1],dz=double(point[2])-q.centroid[2];
              return std::sqrt((dx*dx+dy*dy)+dz*dz)<=best.distance;
            };
            for(int a=0;a<3;++a) {
              int64_t left=0,right=anchor[a];
              while(left<right) { int64_t middle=left+(right-left)/2;if(admissible(a,middle)) right=middle;else left=middle+1; }
              lo[a]=left;left=anchor[a];right=labels.shape[a];
              while(left+1<right) { int64_t middle=left+(right-left)/2;if(admissible(a,middle)) left=middle;else right=middle; }
              hi[a]=left+1;
            }
          }
        }
      }
      if(bounded && (best.index==INT64_MAX || !std::isfinite(best.distance) ||
          size_t(hi[0]-lo[0])*size_t(hi[1]-lo[1])*size_t(hi[2]-lo[2])>4096)) {
        deferred[pattern].push_back(qi);continue;
      }
      scan(lo,hi);
      if(best.index!=INT64_MAX) {
        int64_t yz=labels.shape[1]*labels.shape[2];
        result[qi]={best.index/yz,(best.index/labels.shape[2])%labels.shape[1],best.index%labels.shape[2]};
      } else absent.insert(pattern);
    }
    // Difficult queries share one volume scan per label. This keeps absent or
    // distant labels from causing a complete scan for every centroid, and
    // still never materializes a point cloud or a distance matrix.
    if(fallback && !deferred.empty()) {
      std::vector<size_t> positions;std::vector<brook_label_query> gpu_queries;
      for(const auto &entry:deferred) for(size_t i:entry.second) { positions.push_back(i);gpu_queries.push_back(queries[i]); }
      auto resident=upload(*fallback,labels);
      auto gpu_result=nearest_label_voxels(*fallback,resident,gpu_queries);
      for(size_t i=0;i<positions.size();++i) result[positions[i]]=gpu_result[i];
      return;
    }
    for(const auto &entry:deferred) {
      const auto &indices=entry.second;const auto &pattern=queries[indices.front()];
      std::vector<Candidate> best(indices.size(),{0,INT64_MAX});Best choose;
      visit_box(labels,order,{0,0,0},{labels.shape[0],labels.shape[1],labels.shape[2]},[&](int64_t x,int64_t y,int64_t z,const char *p) {
        T value;std::memcpy(&value,p,sizeof(T));if(!matches(value,pattern)) return;
        int64_t index=(x*labels.shape[1]+y)*labels.shape[2]+z;
        for(size_t i=0;i<indices.size();++i) {
          const auto &q=queries[indices[i]];
          double dx=double(x)-q.centroid[0],dy=double(y)-q.centroid[1],dz=double(z)-q.centroid[2];
          best[i]=choose(best[i],{std::sqrt((dx*dx+dy*dy)+dz*dz),index});
        }
      });
      for(size_t i=0;i<indices.size();++i) if(best[i].index!=INT64_MAX) {
        int64_t yz=labels.shape[1]*labels.shape[2],index=best[i].index;
        result[indices[i]]={index/yz,(index/labels.shape[2])%labels.shape[1],index%labels.shape[2]};
      }
    }
  };
  switch(labels.dtype) {
    case BROOK_U8:search(uint8_t{});break;case BROOK_U16:search(uint16_t{});break;
    case BROOK_U32:search(uint32_t{});break;case BROOK_U64:search(uint64_t{});break;
    case BROOK_I8:search(int8_t{});break;case BROOK_I16:search(int16_t{});break;
    case BROOK_I32:search(int32_t{});break;case BROOK_I64:search(int64_t{});break;
    case BROOK_F32:search(float{});break;case BROOK_F64:search(double{});break;
    default:throw std::invalid_argument("invalid label dtype");
  }
  return result;
}
std::vector<std::array<int64_t,3>> nearest_label_voxels(Context &ctx,const brook_volume &labels,
    const std::vector<brook_label_query> &queries) {
  if(labels.struct_size<sizeof(labels) || labels.abi_version!=BROOK_ABI_VERSION) throw std::invalid_argument("incompatible label volume descriptor");
  size_t n=volume_size({labels.shape[0],labels.shape[1],labels.shape[2]});
  const char *mode=std::getenv("BROOK_NEAREST_MODE");
  bool automatic=!mode || std::strcmp(mode,"auto")==0;
  bool cheap=!n || queries.size()<=16384/n;
  bool bounded=mode && std::strcmp(mode,"bounded")==0;
  if(automatic && labels.memory==BROOK_HOST && !cheap) return nearest_label_voxels_host(labels,queries,true,&ctx);
  bool cpu=labels.memory==BROOK_HOST && ((automatic && cheap) || bounded || (mode && std::strcmp(mode,"cpu")==0));
  if(cpu) return nearest_label_voxels_host(labels,queries,bounded);
  auto resident=upload(ctx,labels);return nearest_label_voxels(ctx,resident,queries);
}
}
