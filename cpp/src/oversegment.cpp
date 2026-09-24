// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// Reproduces the behaviour of Kimimaro 5.8.1 oversegment (kimimaro/utility.py; cpp/licenses/Kimimaro.txt).
#include "oversegment.hpp"
#include "integer_set.hpp"
#include "trace_primitives.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <set>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace brook {
namespace {
// Index of the lowest set bit of x (x != 0): the GCC/Clang builtin, _BitScanForward on MSVC.
#ifdef _MSC_VER
inline unsigned lowest_bit(uint32_t x) {unsigned long index;_BitScanForward(&index,x);return unsigned(index);}
#define BROOK_CTZ(x) lowest_bit(x)
#else
#define BROOK_CTZ(x) __builtin_ctz(x)
#endif
constexpr int offsets[26][3]={
 {-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},
 {-1,-1,0},{-1,1,0},{1,-1,0},{1,1,0},
 {0,-1,-1},{0,-1,1},{0,1,-1},{0,1,1},
 {-1,0,-1},{-1,0,1},{1,0,-1},{1,0,1},
 {-1,-1,-1},{1,-1,-1},{-1,1,-1},{-1,-1,1},
 {1,1,-1},{1,-1,1},{-1,1,1},{1,1,1}};
struct Entry {float cost;uint32_t voxel,source;};
// libstdc++ heap movement with dijkstra3d's >= comparator, expressed without
// passing a non-strict comparator to a standard-library algorithm.
class Heap {
 std::vector<Entry> entries;
 public:
 bool empty() const {return entries.empty();}
 void push(Entry e) {
  size_t h=entries.size();entries.push_back(e);
  while(h && entries[(h-1)/2].cost>=e.cost) {entries[h]=entries[(h-1)/2];h=(h-1)/2;}
  entries[h]=e;
 }
 Entry pop() {
  Entry first=entries.front(),last=entries.back();entries.pop_back();if(entries.empty()) return first;
  size_t h=0;
  while(2*h+2<entries.size()) {
   size_t l=2*h+1,r=l+1,c=entries[r].cost>=entries[l].cost?l:r;
   entries[h]=entries[c];h=c;
  }
  if(2*h+1<entries.size()) {entries[h]=entries[2*h+1];h=2*h+1;}
  while(h && entries[(h-1)/2].cost>=last.cost) {entries[h]=entries[(h-1)/2];h=(h-1)/2;}
  entries[h]=last;return first;
 }
};
Point voxel(std::array<double,3> v,const OversegmentOptions &o,bool float32_coordinates) {
 Point p;
 for(int a=0;a<3;++a) {
  double x=float32_coordinates?double(float(v[a])/float(o.spacing[a])):v[a]/o.spacing[a];
  // nearbyint follows the default IEEE round-to-nearest-even mode.
  x=std::nearbyint(x);
  if(!std::isfinite(x) || x<double(INT64_MIN) || x>=double(INT64_MAX)) throw std::invalid_argument("invalid skeleton coordinate");
  p[a]=int64_t(x);
 }
 return p;
}
// Graph downsampling follows osteoid 0.7.3; BSD notice: cpp/licenses/Osteoid.txt.
std::vector<std::array<double,3>> source_vertices(const OversegmentSkeleton &s,int factor) {
 if(factor<=1) return s.vertices;
 size_t n=s.vertices.size();std::vector<uint32_t> parent(n),degree(n);std::iota(parent.begin(),parent.end(),0);
 auto root=[&](uint32_t v) {while(parent[v]!=v) {parent[v]=parent[parent[v]];v=parent[v];}return v;};
 for(auto e:s.edges) {
  if(e[0]>=n || e[1]>=n) throw std::invalid_argument("edge index outside skeleton");
  ++degree[e[0]];++degree[e[1]];uint32_t a=root(e[0]),b=root(e[1]);if(a!=b) parent[std::max(a,b)]=std::min(a,b);
 }
 std::map<uint32_t,std::vector<uint32_t>> groups;
 for(uint32_t i=0;i<n;++i) if(degree[i]) groups[root(i)].push_back(i);
 std::set<std::array<double,3>> selected;
 for(auto &group:groups) {
  auto &vertices=group.second;std::unordered_map<uint32_t,uint32_t> local;
  for(uint32_t i=0;i<vertices.size();++i) local[vertices[i]]=i;
  std::vector<IntegerSet> adjacent(vertices.size());std::vector<unsigned> counts(vertices.size());
  for(auto e:s.edges) if(root(e[0])==group.first) {
   auto a=local[e[0]],b=local[e[1]];adjacent[a].add(b);adjacent[b].add(a);++counts[a];++counts[b];
  }
  size_t start=0;while(start<counts.size() && counts[start]!=1) ++start;
  if(start==counts.size()) throw std::invalid_argument("downsample requires a terminal vertex in each component");
  struct State {uint32_t node,root;std::vector<uint32_t> path;};
  std::vector<State> stack{{uint32_t(start),uint32_t(start),{}}};std::vector<bool> visited(vertices.size());
  while(!stack.empty()) {
   auto state=std::move(stack.back());stack.pop_back();auto node=state.node;state.path.push_back(node);visited[node]=true;
   if(node!=state.root && (counts[node]==1 || counts[node]>=3)) {
    std::vector<std::array<double,3>> path;
    for(size_t i=0;i<state.path.size();i+=factor) path.push_back(s.vertices[vertices[state.path[i]]]);
    path.push_back(s.vertices[vertices[state.path.back()]]);
    // Consolidation removes vertices supported only by a zero-length edge.
    for(size_t i=1;i<path.size();++i) if(path[i]!=path[i-1]) {selected.insert(path[i]);selected.insert(path[i-1]);}
    state.path={node};state.root=node;
   }
   for(auto child:adjacent[node].values()) if(!visited[child]) stack.push_back({uint32_t(child),state.root,state.path});
  }
 }
 return {selected.begin(),selected.end()};
}
struct LabelInfo {Box box;uint64_t rank=0;};
template<class T> Oversegmentation run(Context *ctx,const brook_volume &v,const std::vector<OversegmentSkeleton> &skeletons,const OversegmentOptions &o) {
 std::array<int64_t,3> shape={v.shape[0],v.shape[1],v.shape[2]};size_t n=volume_size(shape);
 auto read=[&](Point p) {T value;auto ptr=static_cast<const char*>(v.data)+p[0]*v.strides[0]+p[1]*v.strides[1]+p[2]*v.strides[2];std::memcpy(&value,ptr,sizeof(T));return value;};
 std::unordered_map<T,LabelInfo> labels;T previous_label=0;LabelInfo *previous_info=nullptr;
 auto visit=[&](Point p) {
  T value=read(p);if(!value) return;
  if(value!=previous_label) {previous_label=value;previous_info=&labels[value];}
  auto &box=previous_info->box;++box.count;for(int a=0;a<3;++a) {box.lo[a]=std::min(box.lo[a],p[a]);box.hi[a]=std::max(box.hi[a],p[a]);}
 };
 // Bounding-box min/max does not depend on encounter order. Preserve contiguous
 // incoming C bytes on the host, as the CUDA upload path does.
 if(std::abs(v.strides[2])<std::abs(v.strides[0]))
  for(int64_t x=0;x<shape[0];++x) for(int64_t y=0;y<shape[1];++y) for(int64_t z=0;z<shape[2];++z) visit({x,y,z});
 else for(int64_t z=0;z<shape[2];++z) for(int64_t y=0;y<shape[1];++y) for(int64_t x=0;x<shape[0];++x) visit({x,y,z});
 Oversegmentation out;out.shape=shape;out.features.resize(n);out.processed.resize(skeletons.size());uint64_t next_label=0;
 for(size_t i=0;i<skeletons.size();++i) {
  const auto &skel=skeletons[i];T label=o.binary_labels?T(1):T(skel.label);
  auto it=labels.find(label);if(!label || it==labels.end() || (!o.binary_labels && static_cast<long double>(label)!=static_cast<long double>(skel.label))) continue;
  auto box=it->second.box;std::array<int64_t,3> crop_shape;int64_t box_volume=1;
  for(int a=0;a<3;++a) {box_volume*=box.hi[a]-box.lo[a]+1;box.lo[a]=std::max(int64_t(0),box.lo[a]-1);box.hi[a]=std::min(shape[a],box.hi[a]+2);crop_shape[a]=box.hi[a]-box.lo[a];}
  if(box_volume<=1) continue;
  auto vertices=source_vertices(skel,o.downsample);std::vector<Point> sources;sources.reserve(vertices.size());
  for(auto vtx:vertices) {Point p=voxel(vtx,o,skel.float32_coordinates);for(int a=0;a<3;++a) p[a]-=box.lo[a];sources.push_back(p);}
  std::vector<uint8_t> mask(volume_size(crop_shape));
  // Walk the crop in its existing Fortran order without decoding every
  // linear index through out-of-line division helpers. Host strides remain
  // arbitrary, including negative and permuted views.
  size_t j=0;
  for(int64_t z=box.lo[2];z<box.hi[2];++z) for(int64_t y=box.lo[1];y<box.hi[1];++y) {
   const char *row=static_cast<const char*>(v.data)+box.lo[0]*v.strides[0]+y*v.strides[1]+z*v.strides[2];
   for(int64_t x=0;x<crop_shape[0];++x,++j) {T value;std::memcpy(&value,row+x*v.strides[0],sizeof(T));mask[j]=value==label;}
  }
  if(o.fill_holes) {
   if(!ctx) throw std::invalid_argument("fill_holes requires a CUDA context");
   brook_volume mv{};mv.struct_size=sizeof(mv);mv.abi_version=BROOK_ABI_VERSION;mv.data=mask.data();mv.dtype=BROOK_U8;mv.memory=BROOK_HOST;
   int64_t stride=1;for(int a=0;a<3;++a) {mv.shape[a]=crop_shape[a];mv.strides[a]=stride;stride*=crop_shape[a];}
   auto filled=fill_voids(*ctx,upload(*ctx,mv));filled.first.copy_to_host(mask.data(),mask.size());
  }
  std::array<float,3> spacing;for(int a=0;a<3;++a) spacing[a]=float(o.spacing[a]);
  auto features=feature_map_heap(mask,crop_shape,std::move(sources),spacing);
  j=0;
  for(int64_t z=box.lo[2];z<box.hi[2];++z) for(int64_t y=box.lo[1];y<box.hi[1];++y) {
   size_t destination=size_t(box.lo[0]+shape[0]*(y+shape[1]*z));
   for(int64_t x=0;x<crop_shape[0];++x,++j) {
    uint32_t value=features[j];if(mask[j]) value+=uint32_t(next_label);out.features[destination+size_t(x)]+=value;
   }
  }
  next_label+=vertices.size();out.processed[i]=1;
 }
 std::unordered_map<uint64_t,uint64_t> renumber;uint64_t next=0;
 uint64_t previous_value=0,previous_id=0;
 for(auto &value:out.features) if(value) {
  if(value!=previous_value) {auto [it,added]=renumber.emplace(value,next+1);if(added) ++next;previous_value=value;previous_id=it->second;}
  value=previous_id;
 }
 out.maximum=next;
 for(const auto &skel:skeletons) for(auto vtx:skel.vertices) {
  Point p=voxel(vtx,o,skel.float32_coordinates);for(int a=0;a<3;++a) {if(p[a]<0) p[a]+=shape[a];if(p[a]<0 || p[a]>=shape[a]) throw std::invalid_argument("skeleton coordinate outside label volume");}
  out.segments.push_back(out.features[flatten(p,shape)]);
 }
 if(o.in_place && !o.binary_labels) {
  // fastremap mutates contiguous inputs in storage order; strided inputs
  // are first copied, so their caller-owned storage stays unchanged.
  bool f_order=true;int64_t stride=sizeof(T);for(int a=0;a<3;++a) {if(shape[a]>1 && v.strides[a]!=stride) f_order=false;stride*=shape[a];}
  bool c_order=true;stride=sizeof(T);for(int a=2;a>=0;--a) {if(shape[a]>1 && v.strides[a]!=stride) c_order=false;stride*=shape[a];}
  if(!f_order && !c_order) return out; // Strided input: fastremap renumbers a copy.
  uint64_t rank=0;
  auto update=[&](Point p) {T value=read(p);if(value) {auto &id=labels.at(value).rank;if(!id) id=++rank;value=T(id);}auto ptr=const_cast<char*>(static_cast<const char*>(v.data))+p[0]*v.strides[0]+p[1]*v.strides[1]+p[2]*v.strides[2];std::memcpy(ptr,&value,sizeof(T));};
  if(f_order) for(int64_t z=0;z<shape[2];++z) for(int64_t y=0;y<shape[1];++y) for(int64_t x=0;x<shape[0];++x) update({x,y,z});
  else for(int64_t x=0;x<shape[0];++x) for(int64_t y=0;y<shape[1];++y) for(int64_t z=0;z<shape[2];++z) update({x,y,z});
 }
 return out;
}
}
std::vector<uint32_t> feature_map_heap(const std::vector<uint8_t>& mask,std::array<int64_t,3> shape,
                                      std::vector<Point> sources,std::array<float,3> spacing) {
 size_t n=volume_size(shape);if(n!=mask.size() || n>=size_t{1}<<31) throw std::invalid_argument("invalid feature volume size");
 std::sort(sources.begin(),sources.end());sources.erase(std::unique(sources.begin(),sources.end()),sources.end());
 if(sources.empty()) throw std::invalid_argument("feature map requires at least one source");
 std::vector<float> dist(n,INFINITY);std::vector<uint32_t> features(n);Heap heap;
 for(size_t i=0;i<sources.size();++i) {
  auto p=sources[i];for(int a=0;a<3;++a) if(p[a]<0 || p[a]>=shape[a]) throw std::invalid_argument("source outside feature volume");
  uint32_t index=uint32_t(flatten(p,shape));dist[index]=0;features[index]=uint32_t(i+1);heap.push({0,index,uint32_t(i+1)});
 }
 std::array<int64_t,26> steps;std::array<unsigned,26> needed{};std::array<float,26> weights;
 for(int k=0;k<26;++k) {
  steps[k]=offsets[k][0]+shape[0]*(offsets[k][1]+shape[1]*offsets[k][2]);float squared=0;int count=0,axis=0;
  for(int a=0;a<3;++a) if(offsets[k][a]) {needed[k]|=1u<<(2*a+(offsets[k][a]>0));squared+=spacing[a]*spacing[a];++count;axis=a;}
  weights[k]=count==1?spacing[axis]:std::sqrt(squared);
 }
 // Every index is below 2^31, so 32-bit decode is exact. Power-of-two
 // strides require only shifts/masks, as in dijkstra3d.
 uint32_t sx=uint32_t(shape[0]),sy=uint32_t(shape[1]),sxy=sx*sy;
 bool shift=!(sx&(sx-1)) && !(sy&(sy-1));
 unsigned xbits=BROOK_CTZ(sx),ybits=BROOK_CTZ(sy);
 while(!heap.empty()) {
  auto entry=heap.pop();auto v=entry.voxel;float base=dist[v];if(std::signbit(base)) continue;
  Point p;
  if(shift) p={int64_t(v&(sx-1)),int64_t((v>>xbits)&(sy-1)),int64_t(v>>(xbits+ybits))};
  else {uint32_t z=v/sxy,t=v-z*sxy,y=t/sx;p={int64_t(t-y*sx),int64_t(y),int64_t(z)};}
  unsigned allowed=0;for(int a=0;a<3;++a) {if(p[a]>0) allowed|=1u<<(2*a);if(p[a]+1<shape[a]) allowed|=1u<<(2*a+1);}
  for(int k=0;k<26;++k) {
   if((allowed&needed[k])!=needed[k]) continue;uint32_t u=uint32_t(int64_t(v)+steps[k]);if(!mask[u]) continue;
   float value=base+weights[k];
   if(value<dist[u]) {dist[u]=value;features[u]=entry.source;heap.push({value,u,entry.source});}
   else if(value==dist[u] && entry.source>features[u]) features[u]=entry.source;
  }
  dist[v]=-base;
 }
 return features;
}
Oversegmentation oversegment(Context *ctx,const brook_volume &v,const std::vector<OversegmentSkeleton>& skeletons,const OversegmentOptions &o) {
 if(v.struct_size<sizeof(v) || v.abi_version!=BROOK_ABI_VERSION || v.memory!=BROOK_HOST || !v.data) throw std::invalid_argument("oversegment requires a valid host volume");
 for(double spacing:o.spacing) if(!std::isfinite(spacing) || spacing<=0) throw std::invalid_argument("anisotropy must be finite and positive");
 switch(v.dtype) {
  case BROOK_U8:return run<uint8_t>(ctx,v,skeletons,o);case BROOK_U16:return run<uint16_t>(ctx,v,skeletons,o);
  case BROOK_U32:return run<uint32_t>(ctx,v,skeletons,o);case BROOK_U64:return run<uint64_t>(ctx,v,skeletons,o);
  case BROOK_I8:return run<int8_t>(ctx,v,skeletons,o);case BROOK_I16:return run<int16_t>(ctx,v,skeletons,o);
  case BROOK_I32:return run<int32_t>(ctx,v,skeletons,o);case BROOK_I64:return run<int64_t>(ctx,v,skeletons,o);
  case BROOK_F32:return run<float>(ctx,v,skeletons,o);case BROOK_F64:return run<double>(ctx,v,skeletons,o);
 }
 throw std::invalid_argument("invalid label dtype");
}
}
