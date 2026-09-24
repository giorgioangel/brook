// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include <array>
#include <cstdint>
#include <vector>
namespace brook {
// CPython 3.12 tuple hashing and mutable set order; PSF license in licenses/CPython.txt.
class CpuPairSet {
public:
  using Pair = std::array<uint32_t, 2>;

private:
  struct Slot {
    Pair value{};
    uint64_t hash = 0;
    int state = 0;
  };
  std::vector<Slot> table_ = std::vector<Slot>(8);
  size_t used_ = 0, fill_ = 0;
  static uint64_t hash(Pair p) {
    constexpr uint64_t p1 = 11400714785074694791ull, p2 = 14029467366897019727ull,
                       p5 = 2870177450012600261ull;
    uint64_t h = p5;
    for (auto v : p) {
      h += uint64_t(v) * p2;
      h = (h << 31) | (h >> 33);
      h *= p1;
    }
    h += 2 ^ (p5 ^ 3527539ull);
    return h == UINT64_MAX ? 1546275796ull : h;
  }
  size_t find(Pair p, uint64_t h) const {
    size_t mask = table_.size() - 1, i = h & mask, dummy = SIZE_MAX;
    uint64_t perturb = h;
    for (;;) {
      int probes = i + 9 <= mask ? 9 : 0;
      for (int j = 0; j <= probes; ++j) {
        const auto &s = table_[i + j];
        if (!s.state)
          return dummy == SIZE_MAX ? i + j : dummy;
        if (s.state == 1 && s.hash == h && s.value == p)
          return i + j;
        if (s.state == 2)
          dummy = i + j;
      }
      perturb >>= 5;
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

public:
  void add(Pair p) {
    auto h = hash(p);
    auto &s = table_[find(p, h)];
    if (s.state == 1)
      return;
    bool virgin = !s.state;
    s = {p, h, 1};
    ++used_;
    if (!virgin)
      return;
    ++fill_;
    if (fill_ * 5 < (table_.size() - 1) * 3)
      return;
    size_t n = 8, minimum = used_ * (used_ > 50000 ? 2 : 4);
    while (n <= minimum)
      n *= 2;
    auto old = std::move(table_);
    table_ = std::vector<Slot>(n);
    fill_ = used_;
    for (auto v : old)
      if (v.state == 1)
        table_[find(v.value, v.hash)] = v;
  }
  void discard(Pair p) {
    auto &s = table_[find(p, hash(p))];
    if (s.state == 1) {
      s.state = 2;
      --used_;
    }
  }
  std::vector<Pair> values() const {
    std::vector<Pair> out;
    for (auto s : table_)
      if (s.state == 1)
        out.push_back(s.value);
    return out;
  }
};
} // namespace brook
