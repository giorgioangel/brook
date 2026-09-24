// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "streaming.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <numeric>

namespace brook {
namespace {
brook_volume slab(const brook_volume &input,int64_t first,int64_t end) {
  auto v=input;v.shape[2]=end-first;
  v.data=static_cast<const char*>(input.data)+first*input.strides[2];return v;
}
template<class Label> std::vector<uint64_t> read_plane(const Array &array,size_t offset,size_t n) {
  auto plane=array;plane.byte_offset+=offset*sizeof(Label);plane.shape={int64_t(n),1,1};
  std::vector<Label> values(n);plane.copy_to_host(values.data(),n*sizeof(Label));
  return {values.begin(),values.end()};
}
std::vector<uint64_t> read_plane(const Array &array,size_t offset,size_t n) {
  switch(array.dtype) {
    case BROOK_U8:return read_plane<uint8_t>(array,offset,n);
    case BROOK_U16:return read_plane<uint16_t>(array,offset,n);
    case BROOK_U32:return read_plane<uint32_t>(array,offset,n);
    case BROOK_U64:return read_plane<uint64_t>(array,offset,n);
    default:throw std::invalid_argument("invalid streamed component type");
  }
}
size_t root(std::vector<size_t> &parents,size_t node) {
  size_t r=node;while(parents[r]!=r) r=parents[r];
  while(parents[node]!=node) {size_t next=parents[node];parents[node]=r;node=next;}
  return r;
}
template<class Label> int64_t scalar(const char *pointer) {
  Label value;std::memcpy(&value,pointer,sizeof(value));return static_cast<int64_t>(value);
}
int64_t original_label(const brook_volume &v,int64_t flat) {
  int64_t x=flat%v.shape[0],y=(flat/v.shape[0])%v.shape[1],z=flat/(v.shape[0]*v.shape[1]);
  const char *p=static_cast<const char*>(v.data)+x*v.strides[0]+y*v.strides[1]+z*v.strides[2];
  switch(v.dtype) {
    case BROOK_U8:return scalar<uint8_t>(p);case BROOK_U16:return scalar<uint16_t>(p);
    case BROOK_U32:return scalar<uint32_t>(p);case BROOK_U64:return scalar<uint64_t>(p);
    case BROOK_I8:return scalar<int8_t>(p);case BROOK_I16:return scalar<int16_t>(p);
    case BROOK_I32:return scalar<int32_t>(p);case BROOK_I64:return scalar<int64_t>(p);
    default:throw std::invalid_argument("streamed components require integer labels");
  }
}
template<class Input,class Output> __global__ void remap_ids(const Input *input,Output *output,
    const uint64_t *ids,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) output[i]=static_cast<Output>(ids[input[i]]);
}
template<class Output> void remap(Context &ctx,const Array &input,Array &out,const Buffer<uint64_t> &ids) {
  unsigned blocks=static_cast<unsigned>((input.size()+255)/256);
  auto *target=static_cast<Output*>(out.data());long long n=static_cast<long long>(input.size());
  switch(input.dtype) {
    case BROOK_U8:remap_ids<<<blocks,256,0,ctx.stream>>>(static_cast<const uint8_t*>(input.data()),target,ids.data(),n);break;
    case BROOK_U16:remap_ids<<<blocks,256,0,ctx.stream>>>(static_cast<const uint16_t*>(input.data()),target,ids.data(),n);break;
    case BROOK_U32:remap_ids<<<blocks,256,0,ctx.stream>>>(static_cast<const uint32_t*>(input.data()),target,ids.data(),n);break;
    case BROOK_U64:remap_ids<<<blocks,256,0,ctx.stream>>>(static_cast<const uint64_t*>(input.data()),target,ids.data(),n);break;
    default:throw std::invalid_argument("invalid local component type");
  }
  BROOK_CUDA(cudaGetLastError());
}
void remap(Context &ctx,const Array &input,Array &out,const Buffer<uint64_t> &ids) {
  switch(out.dtype) {
    case BROOK_U8:return remap<uint8_t>(ctx,input,out,ids);
    case BROOK_U16:return remap<uint16_t>(ctx,input,out,ids);
    case BROOK_U32:return remap<uint32_t>(ctx,input,out,ids);
    case BROOK_U64:return remap<uint64_t>(ctx,input,out,ids);
    default:throw std::invalid_argument("invalid global component type");
  }
}
}

brook_volume HostComponents::view() const {
  brook_volume v{};v.struct_size=sizeof(v);v.abi_version=BROOK_ABI_VERSION;
  v.data=labels.get();v.dtype=dtype;v.memory=BROOK_HOST;
  int64_t stride=int64_t(dtype_size(dtype));
  for(int a=0;a<3;++a) {v.shape[a]=shape[a];v.strides[a]=stride;stride*=shape[a];}
  return v;
}

HostComponents connected_components_streamed(Context &ctx,const brook_volume &input,size_t budget,
                                             const std::vector<int64_t> *object_ids) {
  ctx.activate();
  if(input.struct_size<sizeof(input) || input.abi_version!=BROOK_ABI_VERSION || input.memory!=BROOK_HOST)
    throw std::invalid_argument("streamed components require a compatible host volume");
  HostComponents result;result.shape={input.shape[0],input.shape[1],input.shape[2]};
  const size_t n=volume_size(result.shape),item=dtype_size(input.dtype);
  if(input.dtype==BROOK_F32 || input.dtype==BROOK_F64 || n>SIZE_MAX/item || (n && !input.data))
    throw std::invalid_argument("streamed components require integer host labels");
  if(!n) {result.dtype=BROOK_U32;return result;}
  budget=default_stream_budget(budget);
  const auto s=result.shape;const size_t plane=size_t(s[0])*size_t(s[1]);
  const int64_t depth=int64_t(std::max(size_t{1},std::min(size_t(s[2]),budget/plane/(item+32))));
  std::vector<size_t> parents,offsets;
  std::vector<int64_t> nodes,previous_mapping;
  std::vector<uint64_t> previous;
  size_t previous_offset=0;
  const char *seam_mode=std::getenv("BROOK_STREAM_SEAMS");
  const bool skip_repeated=!seam_mode || std::string(seam_mode)!="0";
  auto components=[&](const brook_volume &v) {
    auto input=upload_stream_region(ctx,v);
    if(object_ids) input=filter_labels(ctx,input,*object_ids);
    return connected_components(ctx,input);
  };
  // Slab roots are sorted local minima. Appending in Z order also sorts their
  // whole-volume minima, so minimum DSU index is the exact in-core root order.
  for(int64_t first=0;first<s[2];first+=depth) {
    auto view=slab(input,first,std::min(s[2],first+depth));
    auto local=components(view);
    size_t offset=nodes.size();offsets.push_back(offset);
    for(auto r:local.roots) {parents.push_back(parents.size());nodes.push_back(r+int64_t(plane)*first);}
    if(first) {
      auto current=read_plane(local.labels,0,plane);
      auto connect=[&](uint64_t a,uint64_t b) {
        if(!a || !b || previous_mapping[size_t(a)]!=local.mapping[size_t(b)]) return;
        size_t ra=root(parents,previous_offset+size_t(a)-1),rb=root(parents,offset+size_t(b)-1);
        if(ra!=rb) parents[std::max(ra,rb)]=std::min(ra,rb);
      };
      if(skip_repeated) {
        // Scan each of the nine displacements contiguously. Repeated local ID
        // pairs already contributed their union; later unions cannot undo it.
        // This skips duplicate host DSU work without storing a seam-sized set.
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
          uint64_t last_a=UINT64_MAX,last_b=UINT64_MAX;
          for(int64_t y=std::max(0,-dy);y<s[1]-std::max(0,dy);++y)
            for(int64_t x=std::max(0,-dx);x<s[0]-std::max(0,dx);++x) {
              auto a=previous[size_t(x+dx+s[0]*(y+dy))],b=current[size_t(x+s[0]*y)];
              if(a==last_a && b==last_b) continue;
              last_a=a;last_b=b;connect(a,b);
            }
        }
      } else {
        // Direct pair enumeration retained for schedule/measurement comparisons.
        for(int64_t y=0;y<s[1];++y) for(int64_t x=0;x<s[0];++x) {
          auto b=current[size_t(x+s[0]*y)];if(!b) continue;
          for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            int64_t ax=x+dx,ay=y+dy;if(ax<0 || ax>=s[0] || ay<0 || ay>=s[1]) continue;
            connect(previous[size_t(ax+s[0]*ay)],b);
          }
        }
      }
    }
    previous=read_plane(local.labels,local.labels.size()-plane,plane);
    previous_mapping=std::move(local.mapping);previous_offset=offset;
  }
  offsets.push_back(nodes.size());
  previous.clear();previous.shrink_to_fit();previous_mapping.clear();previous_mapping.shrink_to_fit();
  std::vector<uint64_t> global_ids(nodes.size(),0);
  for(size_t i=0;i<nodes.size();++i) if(root(parents,i)==i) {
    result.roots.push_back(nodes[i]);result.mapping.push_back(original_label(input,nodes[i]));
    global_ids[i]=result.roots.size();
  }
  for(size_t i=0;i<nodes.size();++i) global_ids[i]=global_ids[parents[i]];
  const size_t count=result.roots.size();
  result.dtype=count<=UINT8_MAX?BROOK_U8:count<=UINT16_MAX?BROOK_U16:count<=UINT32_MAX?BROOK_U32:BROOK_U64;
  const size_t out_item=dtype_size(result.dtype);
  if(n>SIZE_MAX/out_item) throw std::invalid_argument("component output size overflow");
  // A blank result should retain demand-zero host pages, just as np.zeros does.
  // Nonempty results are filled slab by slab, without an eager full-volume clear.
  const size_t bytes=n*out_item;
  result.labels.reset(static_cast<uint8_t*>(std::calloc(bytes,1)));
  if(!result.labels) throw std::bad_alloc();
  const char *eager=std::getenv("BROOK_STREAM_HOST_EAGER");
  if(eager && std::string(eager)=="1") std::memset(result.labels.get(),0,bytes);
  if(!count) return result;
  size_t block=0;
  for(int64_t first=0;first<s[2];first+=depth,++block) {
    auto view=slab(input,first,std::min(s[2],first+depth));
    auto local=components(view);
    const size_t count_local=offsets[block+1]-offsets[block];
    if(local.roots.size()!=count_local) throw std::runtime_error("streamed component input changed between passes");
    std::vector<uint64_t> ids(count_local+1,0);
    std::copy(global_ids.begin()+offsets[block],global_ids.begin()+offsets[block+1],ids.begin()+1);
    Buffer<uint64_t> device_ids(ctx,ids.size());device_ids.set(ctx,ids.data(),ids.size());
    auto out=allocate(ctx,local.labels.shape,result.dtype);remap(ctx,local.labels,out,device_ids);
    ctx.synchronize();out.copy_to_host(result.labels.get()+size_t(first)*plane*out_item,out.bytes());
  }
  return result;
}
}
