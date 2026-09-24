// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "skeleton.hpp"

namespace brook {
// Compatibility with the insertion-only integer-tuple sets Kimimaro uses to collect border
// targets, with CPython 3.12.3's hashing and set order. No Python runtime is involved.
// Hash/probe algorithms: CPython Objects/{tupleobject,setobject}.c (PSF license;
// the notice and license are in cpp/licenses/CPython.txt).
class PointSet {
  struct Slot { Point point{};uint64_t hash=0;bool live=false; };
  std::vector<Slot> table_=std::vector<Slot>(8);
  size_t used_=0;
  static uint64_t hash(Point point) {
    constexpr uint64_t p1=11400714785074694791ull,p2=14029467366897019727ull,p5=2870177450012600261ull;
    uint64_t acc=p5;
    for(int64_t v:point) {
      uint64_t magnitude=v<0?uint64_t(-(v+1))+1:uint64_t(v);
      int64_t integer_hash=int64_t(magnitude%((uint64_t{1}<<61)-1));
      if(v<0) integer_hash=-integer_hash;
      if(integer_hash==-1) integer_hash=-2;
      acc+=uint64_t(integer_hash)*p2;acc=(acc<<31)|(acc>>33);acc*=p1;
    }
    acc+=uint64_t{3}^(p5^3527539ull);
    return acc==UINT64_MAX?1546275796ull:acc;
  }
  bool insert(Point point,uint64_t h) {
    size_t mask=table_.size()-1,i=h&mask;uint64_t perturb=h;
    for(;;) {
      int probes=i+9<=mask?9:0;
      for(int j=0;j<=probes;++j) {
        auto &slot=table_[i+j];
        if(!slot.live) { slot={point,h,true};return true; }
        if(slot.hash==h && slot.point==point) return false;
      }
      perturb>>=5;i=(i*5+1+perturb)&mask;
    }
  }
 public:
  void add(Point point) {
    if(!insert(point,hash(point))) return;
    ++used_;
    if(used_*5<(table_.size()-1)*3) return;
    size_t target=used_*(used_>50000?2:4),size=8;
    while(size<=target) size*=2;
    auto old=std::move(table_);table_=std::vector<Slot>(size);
    for(const auto &slot:old) if(slot.live) insert(slot.point,slot.hash);
  }
  std::vector<Point> values() const {
    std::vector<Point> result;result.reserve(used_);
    for(const auto &slot:table_) if(slot.live) result.push_back(slot.point);
    return result;
  }
};
}
