// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace brook {
// CPython 3.12.3's set probe/resize/merge rules for non-negative integer keys, hashed as CPython
// hashes integers below 2**61 (hash(k) == k).
// Needed for the processing order of Kimimaro's avocado detector (a Python set). PSF license:
// cpp/licenses/CPython.txt. No Python runtime is used.
class IntegerSet {
  struct Slot {size_t key=0;uint8_t state=0;}; // empty, live, dummy
  std::vector<Slot> table_=std::vector<Slot>(8);
  size_t used_=0,fill_=0;
  size_t locate(size_t key,bool clean=false) const {
    size_t mask=table_.size()-1,i=key&mask,perturb=key,dummy=SIZE_MAX;
    for(;;) {
      int probes=i+9<=mask?9:0;
      for(int j=0;j<=probes;++j) {
        auto &slot=table_[i+j];
        if(!slot.state) return dummy==SIZE_MAX?i+j:dummy;
        if(slot.state==1 && slot.key==key && !clean) return i+j;
        if(slot.state==2 && !clean) dummy=i+j;
      }
      perturb>>=5;i=(i*5+1+perturb)&mask;
    }
  }
  void resize(size_t minimum) {
    size_t n=8;while(n<=minimum) n*=2;
    if(n==8 && table_.size()==8 && used_==fill_) return;
    auto old=std::move(table_);table_=std::vector<Slot>(n);fill_=used_;
    for(auto slot:old) if(slot.state==1) table_[locate(slot.key,true)]=slot;
  }
 public:
  bool contains(size_t key) const {auto &slot=table_[locate(key)];return slot.state==1 && slot.key==key;}
  void add(size_t key) {
    auto &slot=table_[locate(key)];if(slot.state==1) return;
    bool virgin=slot.state==0;slot={key,1};++used_;
    if(virgin) {++fill_;if(fill_*5>=(table_.size()-1)*3) resize(used_*(used_>50000?2:4));}
  }
  void discard(size_t key) {auto &slot=table_[locate(key)];if(slot.state==1 && slot.key==key) {slot.state=2;--used_;}}
  void subtract(const IntegerSet &other) {
    if(this==&other) {*this=IntegerSet{};return;}
    for(auto slot:other.table_) if(slot.state==1) discard(slot.key);
    if(fill_-used_>(table_.size()-1)/4) resize(used_*(used_>50000?2:4));
  }
  void merge(const IntegerSet &other) {
    if(this==&other || !other.used_) return;
    if((fill_+other.used_)*5>=(table_.size()-1)*3) resize((used_+other.used_)*2);
    if(!fill_ && table_.size()==other.table_.size() && other.fill_==other.used_) {*this=other;return;}
    if(!fill_) {
      used_=fill_=other.used_;
      for(auto slot:other.table_) if(slot.state==1) table_[locate(slot.key,true)]=slot;
      return;
    }
    for(auto slot:other.table_) if(slot.state==1) add(slot.key);
  }
  size_t size() const {return used_;}
  std::vector<size_t> values() const {
    std::vector<size_t> out;out.reserve(used_);
    for(auto slot:table_) if(slot.state==1) out.push_back(slot.key);
    return out;
  }
};
}
