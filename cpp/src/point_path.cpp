// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "point_path.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cstring>

namespace brook {
namespace {
struct Entry {float cost;uint32_t voxel;};
// Explicit heap rules preserve the equal-key behavior of dijkstra3d's >= comparator
// without giving std::priority_queue a non-strict comparator.
class PathHeap {
  std::vector<Entry> nodes;
 public:
  bool empty() const {return nodes.empty();}
  void push(Entry value) {
    size_t hole=nodes.size();nodes.push_back(value);
    while(hole && nodes[(hole-1)/2].cost>=value.cost) {nodes[hole]=nodes[(hole-1)/2];hole=(hole-1)/2;}
    nodes[hole]=value;
  }
  Entry pop() {
    auto result=nodes.front(),last=nodes.back();nodes.pop_back();if(nodes.empty()) return result;
    size_t hole=0;
    while(2*hole+2<nodes.size()) {
      size_t left=2*hole+1,right=left+1;
      size_t child=nodes[right].cost>=nodes[left].cost?left:right;
      nodes[hole]=nodes[child];hole=child;
    }
    if(2*hole+1<nodes.size()) {nodes[hole]=nodes[2*hole+1];hole=2*hole+1;}
    while(hole && nodes[(hole-1)/2].cost>=last.cost) {nodes[hole]=nodes[(hole-1)/2];hole=(hole-1)/2;}
    nodes[hole]=last;return result;
  }
};
constexpr int offsets[26][3]={
  {-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},
  {-1,-1,0},{-1,1,0},{1,-1,0},{1,1,0},
  {0,-1,-1},{0,-1,1},{0,1,-1},{0,1,1},
  {-1,0,-1},{-1,0,1},{1,0,-1},{1,0,1},
  {-1,-1,-1},{1,-1,-1},{-1,1,-1},{-1,-1,1},
  {1,1,-1},{1,-1,1},{-1,1,1},{1,1,1}};
struct FreeWords {void operator()(uint32_t *p) const {std::free(p);}};
std::unique_ptr<uint32_t,FreeWords> zero_words(size_t n) {
  auto p=static_cast<uint32_t*>(std::calloc(n,sizeof(uint32_t)));
  if(!p) throw std::bad_alloc();return std::unique_ptr<uint32_t,FreeWords>(p);
}
// Untouched zero-filled pages represent +Inf without an eager full-volume fill.
// Bit conversion preserves every float value and its visited sign bit exactly.
float decode(uint32_t word) {word^=0x7f800000u;float value;std::memcpy(&value,&word,sizeof(value));return value;}
uint32_t encode(float value) {uint32_t word;std::memcpy(&word,&value,sizeof(value));return word^0x7f800000u;}
}
std::vector<int64_t> point_path_heap(const std::vector<float> &field,std::array<int64_t,3> shape,
                                    int64_t source,int64_t target) {
  size_t n=volume_size(shape);
  if(n!=field.size() || n>=size_t{1}<<31 || source<0 || target<0 || size_t(source)>=n || size_t(target)>=n)
    throw std::invalid_argument("invalid point-path volume or endpoints");
  if(source==target) return {source};
  std::array<int64_t,26> steps{};std::array<unsigned,26> needed{};
  for(int k=0;k<26;++k) {
    steps[k]=offsets[k][0]+shape[0]*(offsets[k][1]+shape[1]*offsets[k][2]);
    for(int a=0;a<3;++a) if(offsets[k][a]) needed[k]|=1u<<(2*a+(offsets[k][a]>0));
  }
  auto distance=zero_words(n),parents=zero_words(n);
  distance.get()[source]=encode(0);PathHeap heap;heap.push({0,uint32_t(source)});bool found=false;
  while(!heap.empty() && !found) {
    uint32_t v=heap.pop().voxel;float base=decode(distance.get()[v]);if(std::signbit(base)) continue;
    auto p=unravel(v,shape);
    unsigned allowed=0;for(int a=0;a<3;++a) {if(p[a]>0) allowed|=1u<<(2*a);if(p[a]+1<shape[a]) allowed|=1u<<(2*a+1);}
    for(int k=0;k<26;++k) {
      if((allowed&needed[k])!=needed[k]) continue;
      uint32_t u=uint32_t(int64_t(v)+steps[k]);float next=base+field[u];
      if(next<decode(distance.get()[u])) {
        distance.get()[u]=encode(next);parents.get()[u]=v+1;
        if(u==target) {found=true;break;}
        heap.push({next,u});
      }
    }
    distance.get()[v]=encode(-base);
  }
  if(!found) return {};
  std::vector<int64_t> path;
  for(uint32_t v=uint32_t(target);;) {
    if(path.size()>=n) throw std::runtime_error("point path contains a cycle; weights must be nonnegative");
    path.push_back(v);
    if(!parents.get()[v]) break;
    v=parents.get()[v]-1;
  }
  std::reverse(path.begin(),path.end());return path;
}
}
