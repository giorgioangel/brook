// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "skeleton.hpp"
#include <algorithm>
#include <numeric>

namespace brook {
Point unravel(int64_t i,const std::array<int64_t,3> &s) {
  auto z=i/(s[0]*s[1]); i-=z*s[0]*s[1]; auto y=i/s[0];
  return {i-y*s[0],y,z};
}
int64_t flatten(Point p,const std::array<int64_t,3> &s) { return p[0]+s[0]*(p[1]+s[1]*p[2]); }
template<class S> void consolidate_impl(S &s) {
  if(s.vertices.empty() || s.edges.empty()) { s.vertices.clear();s.edges.clear();s.radii.clear();return; }
  if(s.vertices.size()>UINT32_MAX) throw std::invalid_argument("skeleton exceeds uint32 vertex capacity");
  std::vector<uint32_t> order(s.vertices.size()),inverse(s.vertices.size());
  std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return s.vertices[a]<s.vertices[b];});
  decltype(s.vertices) vertices;
  std::vector<float> radii;
  for(auto old:order) {
    if(vertices.empty() || vertices.back()!=s.vertices[old]) { vertices.push_back(s.vertices[old]);radii.push_back(s.radii[old]); }
    inverse[old]=static_cast<uint32_t>(vertices.size()-1);
  }
  std::vector<std::array<uint32_t,2>> edges;
  for(auto e:s.edges) {
    if(e[0]>=inverse.size() || e[1]>=inverse.size()) throw std::invalid_argument("edge outside skeleton");
    e={inverse[e[0]],inverse[e[1]]};if(e[0]>e[1]) std::swap(e[0],e[1]);
    if(e[0]!=e[1]) edges.push_back(e);
  }
  std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
  std::vector<uint8_t> used(vertices.size(),0);
  for(auto e:edges) used[e[0]]=used[e[1]]=1;
  s.vertices.clear();s.radii.clear();
  for(size_t i=0;i<vertices.size();++i) if(used[i]) {
    inverse[i]=static_cast<uint32_t>(s.vertices.size());s.vertices.push_back(vertices[i]);s.radii.push_back(radii[i]);
  }
  for(auto &e:edges) e={inverse[e[0]],inverse[e[1]]};
  s.edges=std::move(edges);
}
void consolidate(Skeleton &s) {consolidate_impl(s);}
void consolidate(VoxelSkeleton &s) {consolidate_impl(s);}
}
