// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// Reproduces the behaviour of Kimimaro 5.8.1 engage_avocado_protection (kimimaro/intake.py; cpp/licenses/Kimimaro.txt).
#include "skeleton.hpp"
#include "integer_set.hpp"
#include "voxel_graph.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>

namespace brook {
namespace {
size_t index(Point p,const std::array<int64_t,3> &s) {return size_t(p[0]+s[0]*(p[1]+s[1]*p[2]));}
template<class Fn> void visit(std::array<int64_t,3> s,Point origin,std::array<int64_t,3> whole,Fn fn) {
  size_t i=0;
  for(int64_t z=0;z<s[2];++z) for(int64_t y=0;y<s[1];++y) {
    size_t global=index({origin[0],origin[1]+y,origin[2]+z},whole);
    for(int64_t x=0;x<s[0];++x) fn(i++,global++,Point{x,y,z});
  }
}
// Boolean 4/6-connected background filling. State 2 denotes reached exterior.
void fill(std::vector<uint8_t> &mask,std::array<int64_t,3> s,int dimensions) {
  if(mask.size()>=size_t{1}<<31) throw std::invalid_argument("avocado crop exceeds int32 capacity");
  std::vector<int> queue;queue.reserve(mask.size());
  if(dimensions==3) {
    // fill_voids 2.1.2 seeds the first background pixel of each flattened
    // face run, retaining run state across row boundaries. Seeding every
    // boundary pixel changes its result for disconnected runs at face edges.
    const int axes[3][3]={{0,1,2},{0,2,1},{1,2,0}};
    for(auto &axis:axes) for(int side=0;side<2;++side) {
      int a=axis[0],b=axis[1],c=axis[2];bool background=false;
      for(int64_t y=0;y<s[b];++y) for(int64_t x=0;x<s[a];++x) {
        Point p{};p[a]=x;p[b]=y;p[c]=side?s[c]-1:0;size_t i=index(p,s);
        if(!mask[i] && !background) queue.push_back(int(i));
        background=!mask[i];
      }
    }
    for(int i:queue) mask[i]=2;
  } else for(int64_t y=0;y<s[1];++y) for(int64_t x=0;x<s[0];++x) {
    size_t i=index({x,y,0},s);
    if(!mask[i] && (x==0 || x==s[0]-1 || y==0 || y==s[1]-1)) {mask[i]=2;queue.push_back(int(i));}
  }
  for(size_t head=0;head<queue.size();++head) {
    auto p=unravel(queue[head],s);
    for(int axis=0;axis<dimensions;++axis) for(int direction:{-1,1}) {
      auto q=p;q[axis]+=direction;
      if(q[axis]<0 || q[axis]>=s[axis]) continue;
      size_t i=index(q,s);if(!mask[i]) {mask[i]=2;queue.push_back(int(i));}
    }
  }
  for(auto &value:mask) value=value==2?0:1;
}
void paint_walls(std::vector<uint8_t> &mask,std::array<int64_t,3> s) {
  const int axes[3][3]={{0,1,2},{0,2,1},{1,2,0}};
  for(auto &axis:axes) for(int side=0;side<2;++side) {
    int a=axis[0],b=axis[1],c=axis[2];int64_t fixed=side?s[c]-1:0;
    std::array<int64_t,3> plane_shape={s[a],s[b],1};std::vector<uint8_t> plane(volume_size(plane_shape));
    for(int64_t y=0;y<s[b];++y) for(int64_t x=0;x<s[a];++x) {
      Point p{};p[a]=x;p[b]=y;p[c]=fixed;plane[x+s[a]*y]=mask[index(p,s)];
    }
    fill(plane,plane_shape,2);
    for(int64_t y=0;y<s[b];++y) for(int64_t x=0;x<s[a];++x) {
      Point p{};p[a]=x;p[b]=y;p[c]=fixed;mask[index(p,s)]=plane[x+s[a]*y];
    }
  }
}
template<class Label> std::pair<size_t,size_t> fruit(const std::vector<Label> &labels,Point center,std::array<int64_t,3> shape) {
  size_t pit=labels[index(center,shape)];std::map<size_t,int> hits;int rays=0;
  for(int axis=0;axis<3;++axis) for(int direction:{1,-1}) {
    // Kimimaro's negative rays deliberately omit coordinate zero.
    for(Point p=center;direction>0?p[axis]<shape[axis]:p[axis]>0;p[axis]+=direction) {
      size_t value=labels[index(p,shape)];if(!value) break;
      if(value!=pit) {++hits[value];++rays;break;}
    }
  }
  if(rays<3) return {pit,pit};
  size_t result=pit;int most=0;
  for(auto [label,count]:hits) if(count>most) {result=label;most=count;}
  return rays-most>(rays>3?1:0)?std::pair<size_t,size_t>{pit,pit}:std::pair<size_t,size_t>{pit,result};
}
// The correction of one volume on host copies: `labels` holds compact ids below `table` (0 is
// background) and `distance` its distance field; both are updated in place. `refresh(labels,
// distance)` recomputes the field for the current labels. `keys`: the id Kimimaro's detector holds
// for each compact id when they differ (voxel graphs), empty otherwise: its candidate and change sets
// are Python sets of those ids, whose order the sets below reproduce. Returns the number of passes.
template<class Label,class Refresh> int correct(std::vector<Label> &labels,std::vector<float> &distance,std::array<int64_t,3> shape,
    size_t table,double threshold,const std::vector<int64_t> &keys,Refresh refresh) {
  auto key=[&](size_t id) {return keys.empty()?id:size_t(keys[id]);};
  auto id_of=[&](size_t k) {return keys.empty()?k:size_t(std::lower_bound(keys.begin(),keys.end(),int64_t(k))-keys.begin());};
  IntegerSet unchanged;int passes=0;
  for(;passes<20;++passes) {
    std::vector<uint8_t> present(table,0);float cutoff=float(threshold/2.5);
    for(size_t i=0;i<labels.size();++i) present[distance[i]>cutoff?size_t(labels[i]):0]=1;
    IntegerSet candidates;for(size_t id=0;id<table;++id) if(present[id]) candidates.add(key(id));
    candidates.subtract(unchanged);candidates.discard(0);
    if(!candidates.size()) break;
    std::vector<Box> boxes(table);
    visit(shape,{},shape,[&](size_t i,size_t,Point p) {if(labels[i]) {
      auto &box=boxes[labels[i]];++box.count;
      for(int a=0;a<3;++a) {box.lo[a]=std::min(box.lo[a],p[a]);box.hi[a]=std::max(box.hi[a],p[a]);}
    }});
    IntegerSet unchanged_cycle,changed;
    for(size_t candidate:candidates.values()) {
      const size_t label=id_of(candidate);auto box=boxes[label];std::array<int64_t,3> s;
      for(int a=0;a<3;++a) s[a]=box.hi[a]-box.lo[a]+1;
      std::vector<uint8_t> mask(volume_size(s));
      visit(s,box.lo,shape,[&](size_t i,size_t g,Point) {mask[i]=labels[g]==Label(label);});
      paint_walls(mask,s);
      size_t best=0;float maximum=float(mask[0])*distance[index(box.lo,shape)];
      // np.argmax on Kimimaro's Fortran-ordered crop: the first maximum (a NaN wins) in x-fastest
      // order, the order visit() walks the crop in.
      visit(s,box.lo,shape,[&](size_t i,size_t g,Point) {
        float value=float(mask[i])*distance[g];
        if(!(value<=maximum) && !std::isnan(maximum)) {maximum=value;best=i;}
      });
      auto center=unravel(int64_t(best),s);for(int a=0;a<3;++a) center[a]+=box.lo[a];
      auto [pit,owner]=fruit(labels,center,shape);
      if(pit==owner && !changed.contains(key(pit))) unchanged_cycle.add(key(pit));
      else {
        unchanged_cycle.discard(key(pit));unchanged_cycle.discard(key(owner));changed.add(key(pit));changed.add(key(owner));
        visit(s,box.lo,shape,[&](size_t i,size_t g,Point) {mask[i]|=labels[g]==Label(owner);});
      }
      fill(mask,s,3);
      visit(s,box.lo,shape,[&](size_t i,size_t g,Point) {if(mask[i]) labels[g]=Label(owner);});
    }
    unchanged.merge(unchanged_cycle);
    if(!changed.size()) {++passes;break;}
    refresh(labels,distance);
  }
  return passes;
}
// fastremap.renumber on Kimimaro's Fortran-ordered labels: compact ids in the order of their first
// voxel in index order. `mapping` follows the ids of `original` (whose table is `through`) at each run
// start, as get_mapping does; `roots` is the first voxel of each new id. Returns the number of ids.
template<class Label> size_t renumber(std::vector<Label> &labels,const std::vector<Label> &original,const std::vector<int64_t> &through,
    size_t table,std::vector<int64_t> &mapping,std::vector<int64_t> &roots) {
  std::vector<size_t> compact(table,0);mapping.assign(1,0);roots.clear();size_t previous=SIZE_MAX;
  for(size_t i=0;i<labels.size();++i) {
    size_t &label=compact[labels[i]];
    if(labels[i] && !label) {label=roots.size()+1;roots.push_back(int64_t(i));mapping.push_back(0);}
    labels[i]=Label(label);
    if(label!=previous) {if(label) mapping[label]=original[i]?through[original[i]]:0;previous=label;}
  }
  return roots.size();
}
// A stack of `segment`-deep samples along z is corrected sample by sample on the sample's own ids
// and shape: rays, enclosure tests, candidate order and the refreshed distance field never see a
// neighbouring sample. Components are numbered by their first voxel and never cross samples, so
// sample s owns the contiguous id range (base[s], base[s+1]].
template<class Label> void execute(Context &ctx,Components &cc,Array &dbf,double threshold,std::array<float,3> spacing,bool black_border,int64_t segment,
    const Array *voxel_graph) {
  const auto shape=cc.labels.shape;const brook_dtype dtype=cc.labels.dtype;
  const size_t n=cc.labels.size(),table=cc.mapping.size();
  const int64_t depth=segment>0?segment:shape[2];
  const size_t samples=size_t(shape[2]/depth),slab=size_t(shape[0]*shape[1]*depth);
  const std::array<int64_t,3> slab_shape={shape[0],shape[1],depth};
  if((voxel_graph || !cc.keys.empty()) && samples>1) throw std::invalid_argument("stacked avocado correction takes no voxel graph");
  if(!cc.keys.empty() && (cc.keys.size()!=table || std::adjacent_find(cc.keys.begin(),cc.keys.end(),std::greater_equal<int64_t>())!=cc.keys.end()))
    throw std::invalid_argument("avocado keys do not match the component table");
  std::vector<Label> labels(n);cc.labels.copy_to_host(labels.data(),cc.labels.bytes());
  for(auto label:labels) if(size_t(label)>=table) throw std::invalid_argument("avocado mapping does not cover component labels");
  std::vector<float> distance(n);dbf.copy_to_host(distance.data(),dbf.bytes());
  cc.labels={};   // the host copy is the working state; the device volume is rebuilt at the end
  std::vector<size_t> base(samples+1,table-1);base[0]=0;
  if(samples>1) {
    if(cc.roots.size()+1!=table) throw std::invalid_argument("avocado roots do not match the component table");
    // A root lies in its component's sample (an id a fill emptied keeps its root there), so the ids
    // before sample s are those whose root lies before it; the roots need not be sorted.
    std::vector<size_t> owned(samples,0);
    for(auto r:cc.roots) {
      if(r<0 || size_t(r)/slab>=samples) throw std::invalid_argument("avocado roots do not match the component table");
      ++owned[size_t(r)/slab];
    }
    for(size_t s=1;s<samples;++s) base[s]=base[s-1]+owned[s-1];
  }
  Array scratch=allocate(ctx,slab_shape,dtype);std::vector<Array> refreshed(samples);
  std::vector<Label> compact(n);std::vector<int64_t> mapping={0},roots;size_t total=0;int passes=0;
  for(size_t s=0;s<samples;++s) {
    const size_t lo=s*slab,local_table=base[s+1]-base[s]+1;
    std::vector<Label> local(labels.begin()+lo,labels.begin()+lo+slab);
    for(auto &value:local) if(value) {
      if(size_t(value)<=base[s] || size_t(value)>base[s+1]) throw std::logic_error("stacked component ids are not contiguous per sample");
      value=Label(size_t(value)-base[s]);
    }
    const std::vector<Label> original=local;
    std::vector<float> local_distance(distance.begin()+lo,distance.begin()+lo+slab);
    passes=std::max(passes,correct<Label>(local,local_distance,slab_shape,local_table,threshold,cc.keys,[&](const std::vector<Label> &current,std::vector<float> &field) {
      BROOK_CUDA(cudaMemcpyAsync(scratch.data(),current.data(),scratch.bytes(),cudaMemcpyHostToDevice,ctx.stream));ctx.synchronize();
      // Kimimaro's edtfn on its Fortran-ordered labels: the EDT of the initial field, with its voxel graph.
      refreshed[s]=voxel_graph?graph_edt(ctx,scratch,*voxel_graph,spacing,black_border):edt(ctx,scratch,spacing,black_border);
      refreshed[s].copy_to_host(field.data(),refreshed[s].bytes());
    }));
    std::vector<int64_t> through(local_table,0);
    for(size_t id=1;id<local_table;++id) through[id]=cc.mapping[base[s]+id];
    std::vector<int64_t> local_mapping,local_roots;
    const size_t count=renumber<Label>(local,original,through,local_table,local_mapping,local_roots);
    for(size_t i=0;i<slab;++i) compact[lo+i]=local[i]?Label(size_t(local[i])+total):Label(0);
    for(size_t id=1;id<=count;++id) mapping.push_back(local_mapping[id]);
    for(auto root:local_roots) roots.push_back(root+int64_t(lo));
    total+=count;
  }
  scratch={};
  brook_dtype out_dtype=total<=UINT8_MAX?BROOK_U8:(total<=UINT16_MAX?BROOK_U16:(total<=UINT32_MAX?BROOK_U32:BROOK_U64));
  auto output=allocate(ctx,shape,out_dtype);
  auto store=[&](auto value) {using T=decltype(value);std::vector<T> converted(compact.begin(),compact.end());
    BROOK_CUDA(cudaMemcpyAsync(output.data(),converted.data(),output.bytes(),cudaMemcpyHostToDevice,ctx.stream));ctx.synchronize();};
  switch(out_dtype) {case BROOK_U8:store(uint8_t{});break;case BROOK_U16:store(uint16_t{});break;
    case BROOK_U32:store(uint32_t{});break;default:store(uint64_t{});}
  cc.labels=std::move(output);cc.mapping=std::move(mapping);cc.roots=std::move(roots);cc.keys.clear();ctx.stats["avocado_passes"]=passes;
  // Samples whose labels changed carry their refreshed field; the others keep the original one.
  const bool any=std::any_of(refreshed.begin(),refreshed.end(),[](const Array &a){return bool(a.storage);});
  if(!any) return;
  if(samples==1) {dbf=std::move(refreshed[0]);return;}
  auto field=allocate(ctx,shape,BROOK_F32);
  BROOK_CUDA(cudaMemcpyAsync(field.data(),dbf.data(),dbf.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
  for(size_t s=0;s<samples;++s) if(refreshed[s].storage)
    BROOK_CUDA(cudaMemcpyAsync(static_cast<float*>(field.data())+s*slab,refreshed[s].data(),refreshed[s].bytes(),cudaMemcpyDeviceToDevice,ctx.stream));
  ctx.synchronize();dbf=std::move(field);
}
}
void fix_avocados(Context &ctx,Components &cc,Array &dbf,double threshold,std::array<float,3> spacing,bool black_border,int64_t segment,
    const Array *voxel_graph) {
  ctx.activate();
  if(dbf.dtype!=BROOK_F32 || dbf.shape!=cc.labels.shape || cc.mapping.empty()) throw std::invalid_argument("invalid avocado distance field or mapping");
  if(segment<0 || (segment>0 && cc.labels.shape[2]%segment)) throw std::invalid_argument("avocado segment must divide the z extent");
  if(voxel_graph && (voxel_graph->dtype!=BROOK_U32 || voxel_graph->shape!=cc.labels.shape || (segment>0 && segment!=cc.labels.shape[2])))
    throw std::invalid_argument("an avocado voxel graph must be uint32 and match the labels of one sample");
  if(!cc.labels.size()) return;
  switch(cc.labels.dtype) {
    case BROOK_U8:execute<uint8_t>(ctx,cc,dbf,threshold,spacing,black_border,segment,voxel_graph);break;
    case BROOK_U16:execute<uint16_t>(ctx,cc,dbf,threshold,spacing,black_border,segment,voxel_graph);break;
    case BROOK_U32:execute<uint32_t>(ctx,cc,dbf,threshold,spacing,black_border,segment,voxel_graph);break;
    case BROOK_U64:execute<uint64_t>(ctx,cc,dbf,threshold,spacing,black_border,segment,voxel_graph);break;
    default:throw std::invalid_argument("invalid avocado component dtype");
  }
}
}
