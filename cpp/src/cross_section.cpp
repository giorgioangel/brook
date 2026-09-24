// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// Contains a C++ translation of Kimimaro 5.8.1 cross_sectional_area (kimimaro/utility.py)
// (authors: Alex Bae and Will Silversmith, Seung Lab, Princeton Neuroscience Institute;
// GPL-3.0-or-later; cpp/licenses/Kimimaro.txt, Kimimaro-AUTHORS.txt). Translated and modified
// for Brook by Giorgio Angelotti, 2026. Path semantics follow osteoid 0.7.3 (BSD-3-Clause;
// cpp/licenses/Osteoid.txt).
#include "cross_section.hpp"
#include "../third_party/xs3d/bridge.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <deque>
#include <map>
#include <numeric>
#include <set>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace brook {
namespace {
using V=std::array<float,3>;
using D=std::array<double,3>;
using Path=std::vector<D>;
struct ShapeGuard { ~ShapeGuard(){brook_xs3d_clear_shape();} };

std::vector<Path> paths(const Skeleton &input,const std::vector<D> &precise) {
  const size_t n=input.vertices.size();
  std::vector<std::vector<uint32_t>> adjacency(n);
  for(auto e:input.edges) {
    if(e[0]>=n || e[1]>=n) throw std::invalid_argument("edge outside skeleton");
    adjacency[e[0]].push_back(e[1]);adjacency[e[1]].push_back(e[0]);
  }
  std::vector<uint8_t> seen(n);std::vector<Path> result;
  for(size_t first=0;first<n;++first) {
    if(seen[first] || adjacency[first].empty()) continue;
    std::vector<uint32_t> component{uint32_t(first)};seen[first]=1;
    for(size_t i=0;i<component.size();++i) for(auto v:adjacency[component[i]]) if(!seen[v]) {seen[v]=1;component.push_back(v);}
    std::sort(component.begin(),component.end());
    std::vector<D> vertices;std::vector<std::array<uint32_t,2>> edges;
    for(auto v:component) vertices.push_back(precise[v]);
    std::sort(vertices.begin(),vertices.end());vertices.erase(std::unique(vertices.begin(),vertices.end()),vertices.end());
    std::vector<uint32_t> remap(n);
    for(auto v:component) remap[v]=uint32_t(std::lower_bound(vertices.begin(),vertices.end(),precise[v])-vertices.begin());
    for(auto v:component) for(auto w:adjacency[v]) if(v<=w&&remap[v]!=remap[w]) {
      auto a=remap[v],b=remap[w];if(a>b) std::swap(a,b);edges.push_back({a,b});
    }
    std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    if(edges.empty()) continue;
    std::vector<std::vector<uint32_t>> tree(vertices.size());
    for(auto e:edges) {tree[e[0]].push_back(e[1]);tree[e[1]].push_back(e[0]);}
    auto traverse=[&](uint32_t root) {
      struct Item {std::vector<uint32_t> path;std::vector<uint8_t> visited;};
      std::deque<Item> queue;queue.push_back({{root},std::vector<uint8_t>(tree.size())});
      std::vector<std::vector<uint32_t>> leaves;
      while(!queue.empty()) {
        auto item=std::move(queue.front());queue.pop_front();auto v=item.path.back();item.visited[v]=1;bool leaf=true;
        for(auto child:tree[v]) if(!item.visited[child]) {leaf=false;auto next=item;next.path.push_back(child);queue.push_back(std::move(next));}
        if(leaf) leaves.push_back(std::move(item.path));
      }
      return leaves;
    };
    auto initial=traverse(edges[0][0]);size_t longest=0;
    for(size_t i=1;i<initial.size();++i) if(initial[i].size()>initial[longest].size()) longest=i;
    for(auto &indices:traverse(initial[longest].back())) {
      Path path;for(auto it=indices.rbegin();it!=indices.rend();++it) path.push_back(vertices[*it]);result.push_back(std::move(path));
    }
  }
  return result;
}

std::vector<D> moving_average(const std::vector<D>& input,int window) {
  const int64_t length=int64_t(input.size());
  std::vector<D> prefix(size_t(length+int64_t(2)*window));D sum{};
  for(int64_t i=-window;i<length+window;++i) {
    int64_t j=i%(2*length);if(j<0) j+=2*length;if(j>=length) j=2*length-1-j;
    for(int a=0;a<3;++a) sum[a]+=input[j][a];prefix[size_t(i+window)]=sum;
  }
  std::vector<D> out(input.size());
  for(size_t i=0;i<out.size();++i) for(int a=0;a<3;++a) out[i][a]=(prefix[i+window][a]-prefix[i][a])/double(window);
  return out;
}
D voxel(D v,const CrossSectionInput &in,const CrossSectionOptions &o) {
  if(in.physical) for(int a=0;a<3;++a) {
    double spacing=o.float64_spacing?o.position_spacing[a]:double(o.anisotropy[a]);
    v[a]=(in.float64_vertices||o.float64_spacing)?std::nearbyint(v[a]/spacing):double(std::nearbyint(float(v[a])/o.anisotropy[a]));
  }
  return v;
}
std::vector<V> normals(const Path &path,int window) {
  std::vector<D> raw(path.size());
  for(size_t i=0;i+1<path.size();++i) for(int a=0;a<3;++a) raw[i][a]=float(path[i+1][a]-path[i][a]);
  raw.back()=raw[raw.size()-2];
  if(window>1) {raw=moving_average(raw,window);std::reverse(raw.begin(),raw.end());raw=moving_average(raw,window);std::reverse(raw.begin(),raw.end());}
  std::vector<V> out(path.size());
  for(size_t i=0;i<raw.size();++i) {
    if(window==1) {
      float x=float(raw[i][0]),y=float(raw[i][1]),z=float(raw[i][2]);float norm=std::sqrt((x*x+y*y)+z*z);
      out[i]={x/norm,y/norm,z/norm};
    } else {
      auto v=raw[i];double norm=std::sqrt((v[0]*v[0]+v[1]*v[1])+v[2]*v[2]);
      for(int a=0;a<3;++a) out[i][a]=float(v[a]/norm);
    }
  }
  return out;
}
struct BinaryMask {
  std::unique_ptr<bool[]> values;
  size_t length;
  explicit BinaryMask(size_t n):values(std::make_unique<bool[]>(n)),length(n) {}
  size_t size() const {return length;}
  bool *data() {return values.get();}
  bool &operator[](size_t i) {return values[i];}
};
void fill(BinaryMask &mask,Point s) {
  std::vector<uint8_t> visited(mask.size());std::vector<size_t> queue;
  auto add=[&](size_t i){if(!mask[i]&&!visited[i]) {visited[i]=1;queue.push_back(i);}};
  for(int64_t z=0;z<s[2];++z) for(int64_t y=0;y<s[1];++y) {add(flatten({0,y,z},s));add(flatten({s[0]-1,y,z},s));}
  for(int64_t z=0;z<s[2];++z) for(int64_t x=0;x<s[0];++x) {add(flatten({x,0,z},s));add(flatten({x,s[1]-1,z},s));}
  for(int64_t y=0;y<s[1];++y) for(int64_t x=0;x<s[0];++x) {add(flatten({x,y,0},s));add(flatten({x,y,s[2]-1},s));}
  for(size_t k=0;k<queue.size();++k) {auto p=unravel(queue[k],s);for(int a=0;a<3;++a) for(int d:{-1,1}) {auto q=p;q[a]+=d;if(q[a]>=0&&q[a]<s[a]) add(flatten(q,s));}}
  for(size_t i=0;i<mask.size();++i) mask[i]=mask[i]||!visited[i];
}
template<class T> T read(const brook_volume &v,Point p) {
  T value;const char *ptr=static_cast<const char*>(v.data)+p[0]*v.strides[0]+p[1]*v.strides[1]+p[2]*v.strides[2];std::memcpy(&value,ptr,sizeof(value));return value;
}
template<class T> void renumber_in_place(const brook_volume &volume) {
  bool c=true,f=true;int64_t cs=sizeof(T),fs=sizeof(T);
  for(int a=0;a<3;++a) {
    if(volume.shape[a]>1&&volume.strides[a]!=fs) f=false;
    if(volume.shape[2-a]>1&&volume.strides[2-a]!=cs) c=false;
    fs*=volume.shape[a];cs*=volume.shape[2-a];
  }
  if(!c&&!f) return; // fastremap copies noncontiguous views, even with in_place.
  std::unordered_map<T,T> remap;remap.emplace(T(0),T(0));uint64_t next=1;
  size_t size=volume_size({volume.shape[0],volume.shape[1],volume.shape[2]});
  auto *data=static_cast<char*>(const_cast<void*>(volume.data));
  // Either contiguous layout scans in its physical order (F wins if both).
  for(size_t i=0;i<size;++i) {
    T old;std::memcpy(&old,data+i*sizeof(T),sizeof(T));
    auto [it,inserted]=remap.emplace(old,T(next));if(inserted) ++next;
    std::memcpy(data+i*sizeof(T),&it->second,sizeof(T));
  }
}
template<class T> std::vector<CrossSectionResult> compute(const brook_volume &volume,
    const std::vector<CrossSectionInput> &inputs,const CrossSectionOptions &o) {
  Point shape={volume.shape[0],volume.shape[1],volume.shape[2]};
  struct Bounds {Point lo,hi;};std::map<T,Bounds> boxes;
  auto label_value=[](const CrossSectionInput &in) {return in.unsigned_label?T(*in.unsigned_label):T(in.skeleton.label);};
  auto nonzero=[](const CrossSectionInput &in) {return in.unsigned_label?*in.unsigned_label!=0:in.skeleton.label!=0;};
  auto fits=[](const CrossSectionInput &in) {
    if(in.unsigned_label) return *in.unsigned_label<=uint64_t(std::numeric_limits<T>::max());
    int64_t label=in.skeleton.label;
    if constexpr(std::is_unsigned_v<T>) return label>=0&&uint64_t(label)<=uint64_t(std::numeric_limits<T>::max());
    else return label>=int64_t(std::numeric_limits<T>::lowest())&&label<=int64_t(std::numeric_limits<T>::max());
  };
  for(auto &in:inputs) if(nonzero(in)&&fits(in)) boxes.emplace(label_value(in),Bounds{shape,{-1,-1,-1}});
  const char *scan_mode=std::getenv("BROOK_AREA_SCAN");
  if(o.shape_is_crop) {
    // Crackle already supplied the exact binary crop and its origin.
  } else if(scan_mode&&std::string(scan_mode)=="0") {
    for(int64_t z=0;z<shape[2];++z) for(int64_t y=0;y<shape[1];++y) for(int64_t x=0;x<shape[0];++x) {
      Point p{x,y,z};auto it=boxes.find(read<T>(volume,p));if(it==boxes.end()) continue;
      for(int a=0;a<3;++a) {it->second.lo[a]=std::min(it->second.lo[a],p[a]);it->second.hi[a]=std::max(it->second.hi[a],p[a]);}
    }
  } else if(!boxes.empty()) {
    // Bounding boxes commute across visits. Scan the smallest byte stride first
    // and update bounds once per equal-label row run, never once per voxel.
    std::array<int,3> axes={0,1,2};
    auto magnitude=[](int64_t x){return x<0?uint64_t(-(x+1))+1:uint64_t(x);};
    std::stable_sort(axes.begin(),axes.end(),[&](int a,int b){return magnitude(volume.strides[a])<magnitude(volume.strides[b]);});
    int inner=axes[0],middle=axes[1],outer=axes[2];Point p{};
    for(p[outer]=0;p[outer]<shape[outer];++p[outer]) for(p[middle]=0;p[middle]<shape[middle];++p[middle]) {
      p[inner]=0;
      const char *source=static_cast<const char*>(volume.data)+p[outer]*volume.strides[outer]+p[middle]*volume.strides[middle];
      const int64_t inner_stride=volume.strides[inner];
      T previous;std::memcpy(&previous,source,sizeof(T));int64_t begin=0;
      auto flush=[&](int64_t end) {
        auto it=boxes.find(previous);if(it==boxes.end()) return;
        for(int a=0;a<3;++a) {
          int64_t low=a==inner?begin:p[a],high=a==inner?end-1:p[a];
          it->second.lo[a]=std::min(it->second.lo[a],low);it->second.hi[a]=std::max(it->second.hi[a],high);
        }
      };
      for(p[inner]=1;p[inner]<shape[inner];++p[inner]) {
        source+=inner_stride;T value;std::memcpy(&value,source,sizeof(T));
        if(value!=previous) {flush(p[inner]);begin=p[inner];previous=value;}
      }
      flush(shape[inner]);
    }
  }
  std::vector<CrossSectionResult> results;results.reserve(inputs.size());
  for(auto &in:inputs) {
    size_t count=in.skeleton.vertices.size();CrossSectionResult result;
    auto it=boxes.find(label_value(in));bool present=nonzero(in)&&fits(in)&&it!=boxes.end()&&it->second.hi[0]>=0;
    Point origin{},crop{};size_t voxels=1;
    if(present) for(int a=0;a<3;++a) {voxels*=size_t(it->second.hi[a]-it->second.lo[a]+1);origin[a]=std::max(int64_t(0),it->second.lo[a]-1);crop[a]=std::min(shape[a],it->second.hi[a]+2)-origin[a];}
    present=present&&voxels>1;
    if(o.shape_is_crop) {present=true;origin={0,0,0};crop=shape;}
    if(!present) {
      result.areas=in.areas.empty()?std::vector<float>(count,-1):in.areas;
      result.contacts=in.contacts.empty()?std::vector<uint8_t>(count):in.contacts;
      results.push_back(std::move(result));continue;
    }
    result.processed=true;
    if(o.repair_contacts||(o.multipass&&!in.areas.empty())) {
      if(in.areas.size()!=count||in.contacts.size()!=count) throw std::invalid_argument("repair/multipass requires matching areas and contacts");
      result.areas=in.areas;result.contacts=in.contacts;
    } else {result.areas.resize(count);result.contacts.resize(count);}
    BinaryMask mask(volume_size(crop));
    size_t mask_index=0;
    for(int64_t z=0;z<crop[2];++z) for(int64_t y=0;y<crop[1];++y) for(int64_t x=0;x<crop[0];++x)
      mask[mask_index++]=read<T>(volume,{x+origin[0],y+origin[1],z+origin[2]})==label_value(in);
    if(o.fill_holes) fill(mask,crop);
    Point global_origin=origin;for(int a=0;a<3;++a) global_origin[a]+=o.origin[a];
    if(o.visualize) {result.sections.resize(mask.size());result.section_shape=crop;result.section_origin=global_origin;}
    brook_xs3d_set_shape(crop[0],crop[1],crop[2]);ShapeGuard guard;
    const char *u8=std::getenv("BROOK_AREA_U8");
    auto area_fn=u8&&std::string(u8)=="1"?brook_xs3d_area_u8:brook_xs3d_area;
    std::vector<D> precise=in.precise_vertices;
    if(precise.empty()) for(auto v:in.skeleton.vertices) precise.push_back({v[0],v[1],v[2]});
    std::map<D,size_t> mapping;std::vector<size_t> degree(count);std::vector<uint8_t> visited(count);
    for(size_t i=0;i<count;++i) {auto v=voxel(precise[i],in,o);for(int a=0;a<3;++a) v[a]-=double(global_origin[a]);mapping[v]=i;}
    for(auto e:in.skeleton.edges) {if(e[0]>=count||e[1]>=count) throw std::invalid_argument("edge outside skeleton");++degree[e[0]];++degree[e[1]];}
    std::map<size_t,std::vector<float>> branch_values;
    for(auto path:paths(in.skeleton,precise)) {
      for(auto &v:path) {v=voxel(v,in,o);for(int a=0;a<3;++a) v[a]-=double(global_origin[a]);}
      auto normal=normals(path,o.smoothing_window);int ct=0;
      for(size_t i=0;i<path.size();++i) {
        ++ct;if(ct<o.step&&i!=0&&i+1!=path.size()) continue;else if(ct==o.step) ct=0;
        auto p=path[i];if(p[0]<0||p[1]<0||p[2]<0||p[0]>=crop[0]||p[1]>=crop[1]||p[2]>=crop[2]) continue;
        size_t idx=mapping.at(p);bool branch=degree[idx]>=3;
        if(result.areas[idx]!=0&&!branch&&!(o.repair_contacts&&result.contacts[idx]>0&&!visited[idx])) continue;
        visited[idx]=1;auto n=normal[i];
        if(!std::isfinite(n[0])||!std::isfinite(n[1])||!std::isfinite(n[2])) throw std::invalid_argument("section normal is not finite after voxel rounding");
        uint64_t dims[3]={uint64_t(crop[0]),uint64_t(crop[1]),uint64_t(crop[2])};uint8_t contact=0;
        V position={float(p[0]),float(p[1]),float(p[2])};
        float area=area_fn(mask.data(),dims,position.data(),n.data(),o.anisotropy.data(),&contact);
        result.areas[idx]=area;if(o.repair_contacts) result.contacts[idx]=contact;else result.contacts[idx]|=contact;
        if(branch) branch_values[idx].push_back(area);
        if(o.visualize) {
          std::vector<float> plane(mask.size());brook_xs3d_section(mask.data(),dims,position.data(),n.data(),o.anisotropy.data(),plane.data());
          for(size_t j=0;j<plane.size();++j) if(plane[j]>0) result.sections[j]=uint32_t(idx);
        }
      }
    }
    for(auto &[idx,values]:branch_values) {float sum=0;for(float value:values) sum+=value;result.areas[idx]=sum/float(values.size());}
    results.push_back(std::move(result));
  }
  if(o.in_place) renumber_in_place<T>(volume);
  return results;
}
}
std::vector<CrossSectionResult> cross_sectional_area_host(const brook_volume &volume,
    const std::vector<CrossSectionInput> &inputs,const CrossSectionOptions &o) {
  if(volume.struct_size<sizeof(volume)||volume.abi_version!=BROOK_ABI_VERSION||volume.memory!=BROOK_HOST) throw std::invalid_argument("cross sections require a valid host volume");
  size_t size=volume_size({volume.shape[0],volume.shape[1],volume.shape[2]});
  if(!size) throw std::invalid_argument("cross sections require a nonempty volume");
  if(size&&!volume.data) throw std::invalid_argument("cross sections require valid volume storage");
  if(o.step<1||o.smoothing_window<1) throw std::invalid_argument("step and smoothing_window must be positive");
  for(float a:o.anisotropy) if(!std::isfinite(a)||a<=0) throw std::invalid_argument("anisotropy must be finite and positive");
  for(auto &in:inputs) {
    if(!in.precise_vertices.empty()&&in.precise_vertices.size()!=in.skeleton.vertices.size()) throw std::invalid_argument("vertex counts differ");
    for(auto v:in.skeleton.vertices) for(float x:v) if(!std::isfinite(x)) throw std::invalid_argument("vertices must be finite");
    for(auto v:in.precise_vertices) for(double x:v) if(!std::isfinite(x)) throw std::invalid_argument("vertices must be finite");
  }
  switch(volume.dtype) {
    case BROOK_U8:return compute<uint8_t>(volume,inputs,o);case BROOK_U16:return compute<uint16_t>(volume,inputs,o);
    case BROOK_U32:return compute<uint32_t>(volume,inputs,o);case BROOK_U64:return compute<uint64_t>(volume,inputs,o);
    case BROOK_I8:return compute<int8_t>(volume,inputs,o);case BROOK_I16:return compute<int16_t>(volume,inputs,o);
    case BROOK_I32:return compute<int32_t>(volume,inputs,o);case BROOK_I64:return compute<int64_t>(volume,inputs,o);
    default:throw std::invalid_argument("cross sections require integer labels");
  }
}
}
