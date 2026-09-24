// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// Reproduces the behaviour of Kimimaro 5.8.1 oversegment (kimimaro/utility.py; cpp/licenses/Kimimaro.txt).
#pragma once
#include "skeleton.hpp"
#include <cstdlib>

namespace brook {
struct OversegmentOptions {
  std::array<double,3> spacing={1,1,1};
  bool float32_coordinates=true,binary_labels=false,fill_holes=false,in_place=false;
  int downsample=0;
};
// Metadata objects may contain float64 vertices even when the packed C SDK
// uses float32. Carry exact source coordinates and arithmetic precision per item.
struct OversegmentSkeleton {
  int64_t label=0;
  std::vector<std::array<double,3>> vertices;
  std::vector<std::array<uint32_t,2>> edges;
  bool float32_coordinates=true;
};
class FeatureStorage {
  struct Free {void operator()(uint64_t *p) const {std::free(p);}};
  std::unique_ptr<uint64_t,Free> values;
  size_t count=0;
 public:
  void resize(size_t n) {
    if(n>SIZE_MAX/sizeof(uint64_t)) throw std::invalid_argument("feature byte size overflow");
    auto p=static_cast<uint64_t*>(std::calloc(n?n:1,sizeof(uint64_t)));
    if(!p) throw std::bad_alloc();values.reset(p);count=n;
  }
  size_t size() const {return count;}
  uint64_t *begin() {return values.get();}
  uint64_t *end() {return count?values.get()+count:values.get();}
  const uint64_t *begin() const {return values.get();}
  const uint64_t *end() const {return count?values.get()+count:values.get();}
  uint64_t &operator[](size_t i) {return values.get()[i];}
  uint64_t operator[](size_t i) const {return values.get()[i];}
};
struct Oversegmentation {
  std::array<int64_t,3> shape{};
  FeatureStorage features;
  std::vector<uint64_t> segments;
  std::vector<uint8_t> processed;
  uint64_t maximum=0;
};
// Exact heap source ownership, as in Kimimaro's oversegment; source points are sorted and deduplicated.
std::vector<uint32_t> feature_map_heap(const std::vector<uint8_t>& mask,std::array<int64_t,3> shape,
                                     std::vector<Point> sources,std::array<float,3> spacing);
Oversegmentation oversegment(Context *ctx,const brook_volume &labels,const std::vector<OversegmentSkeleton>& skeletons,
                            const OversegmentOptions &options);
}
