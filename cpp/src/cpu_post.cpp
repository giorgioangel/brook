// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// Contains a C++ translation of Kimimaro 5.8.1 kimimaro/post.py and skeletontricks
// (authors: Alex Bae and Will Silversmith, Seung Lab, Princeton Neuroscience Institute;
// GPL-3.0-or-later; cpp/licenses/Kimimaro.txt, Kimimaro-AUTHORS.txt). Translated and modified
// for Brook by Giorgio Angelotti, 2026. Graph semantics follow osteoid 0.7.3 (BSD-3-Clause;
// cpp/licenses/Osteoid.txt).
// Float operation order and insertion order are observable compatibility rules.
#include "cpu_post.hpp"
#include "integer_set.hpp"
#include "cpu_pair_set.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <unordered_map>

extern "C" {
void *brook_cpu_tree_create(float *, uint32_t);
void brook_cpu_tree_delete(void *);
void brook_cpu_tree_query(void *, float *, float *, uint32_t, float, uint32_t *, float *);
void *brook_cpu_tree64_create(double *, uint32_t);
void brook_cpu_tree64_delete(void *);
void brook_cpu_tree64_query(void *, double *, double *, uint32_t, double, uint32_t *, double *);
}
namespace brook {
namespace {
using G = CpuSkeleton;
using E = std::array<uint32_t, 2>;
using V = std::array<double, 3>;
constexpr float inf = std::numeric_limits<float>::infinity();
bool empty(const G &g) { return g.skeleton.vertices.empty() || g.skeleton.edges.empty(); }
G input(CpuGeometry s, uint64_t offset = 0, CpuMetadata metadata = {}) {
  if (s.radii.size() != s.vertices.size())
    throw std::invalid_argument("one radius per vertex is required");
  if (s.vertices.size() > UINT32_MAX)
    throw std::invalid_argument("graph exceeds uint32 capacity");
  for (auto e : s.edges)
    for (auto i : e)
      if (i >= s.vertices.size())
        throw std::invalid_argument("edge outside skeleton");
  for (auto v : s.vertices)
    for (auto x : v)
      if (!std::isfinite(x))
        throw std::invalid_argument("coordinates must be finite");
  G g{std::move(s), {}, metadata};
  g.sources.resize(g.skeleton.vertices.size());
  std::iota(g.sources.begin(), g.sources.end(), offset);
  return g;
}
G select(const G &g, const std::vector<uint32_t> &ids, std::vector<E> edges) {
  G out;
  out.metadata = g.metadata;
  out.skeleton.vertex64 = g.skeleton.vertex64;
  out.skeleton.radius64 = g.skeleton.radius64;
  out.skeleton.label = g.skeleton.label;
  out.skeleton.anisotropy = g.skeleton.anisotropy;
  std::vector<uint32_t> remap(g.skeleton.vertices.size());
  for (auto i : ids) {
    remap[i] = out.sources.size();
    out.sources.push_back(g.sources[i]);
    out.skeleton.vertices.push_back(g.skeleton.vertices[i]);
    out.skeleton.radii.push_back(g.skeleton.radii[i]);
  }
  for (auto &e : edges) {
    e = {remap[e[0]], remap[e[1]]};
  }
  out.skeleton.edges = std::move(edges);
  return out;
}
G merge(const std::vector<G> &gs) {
  G out;
  bool first = true;
  for (const auto &g : gs)
    if (!empty(g)) {
      out.skeleton.vertex64 |= g.skeleton.vertex64;
      out.skeleton.radius64 |= g.skeleton.radius64;
    }
  for (const auto &g : gs)
    if (!empty(g)) {
      if (first) {
        out.skeleton.label = g.skeleton.label;
        out.skeleton.anisotropy = g.skeleton.anisotropy;
        out.metadata = g.metadata;
        first = false;
      }
      auto n = out.sources.size();
      if (n + g.sources.size() > UINT32_MAX)
        throw std::invalid_argument("merged graph exceeds uint32 capacity");
      for (auto e : g.skeleton.edges)
        out.skeleton.edges.push_back({uint32_t(e[0] + n), uint32_t(e[1] + n)});
      if (g.metadata.physical == out.metadata.physical)
        out.skeleton.vertices.insert(out.skeleton.vertices.end(), g.skeleton.vertices.begin(),
                                     g.skeleton.vertices.end());
      else {
        auto t =
            out.metadata.physical ? g.metadata.transform : cpu_inverse(g.metadata.transform);
        for (auto v : g.skeleton.vertices)
          out.skeleton.vertices.push_back(cpu_transform_point(v, t, g.skeleton.vertex64));
      }
      out.skeleton.radii.insert(out.skeleton.radii.end(), g.skeleton.radii.begin(),
                                g.skeleton.radii.end());
      out.sources.insert(out.sources.end(), g.sources.begin(), g.sources.end());
    }
  return out;
}
G clean(const G &g, bool prune = true) {
  if (empty(g)) {
    G o;
    o.metadata = g.metadata;
    o.skeleton.label = g.skeleton.label;
    return o;
  }
  std::vector<uint32_t> order(g.sources.size()), inverse(g.sources.size()), unique;
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return g.skeleton.vertices[a] < g.skeleton.vertices[b]; });
  for (auto i : order) {
    if (unique.empty() || g.skeleton.vertices[unique.back()] != g.skeleton.vertices[i])
      unique.push_back(i);
    inverse[i] = unique.size() - 1;
  }
  auto out = select(g, unique, {});
  for (auto e : g.skeleton.edges) {
    e = {inverse[e[0]], inverse[e[1]]};
    if (e[0] > e[1])
      std::swap(e[0], e[1]);
    if (e[0] != e[1])
      out.skeleton.edges.push_back(e);
  }
  auto &es = out.skeleton.edges;
  std::sort(es.begin(), es.end());
  es.erase(std::unique(es.begin(), es.end()), es.end());
  if (!prune)
    return out;
  std::vector<bool> used(unique.size());
  for (auto e : es)
    used[e[0]] = used[e[1]] = true;
  order.clear();
  for (uint32_t i = 0; i < used.size(); ++i)
    if (used[i])
      order.push_back(i);
  return select(out, order, es);
}
std::vector<G> components(const G &g) {
  if (empty(g))
    return {};
  size_t n = g.sources.size();
  std::vector<uint32_t> p(n);
  std::iota(p.begin(), p.end(), 0);
  auto root = [&](uint32_t a) {
    while (p[a] != a) {
      p[a] = p[p[a]];
      a = p[a];
    }
    return a;
  };
  for (auto e : g.skeleton.edges) {
    auto a = root(e[0]), b = root(e[1]);
    p[std::max(a, b)] = std::min(a, b);
  }
  // Osteoid lists components in the order of their first edge.
  std::unordered_map<uint32_t, std::vector<E>> groups;
  std::vector<uint32_t> order;
  for (auto e : g.skeleton.edges) {
    if (e[0] > e[1])
      std::swap(e[0], e[1]);
    auto &es = groups[root(e[0])];
    if (es.empty())
      order.push_back(root(e[0]));
    es.push_back(e);
  }
  if (groups.size() == 1)
    return {g};
  std::vector<G> out;
  for (auto key : order) {
    auto &es = groups[key];
    std::sort(es.begin(), es.end());
    es.erase(std::unique(es.begin(), es.end()), es.end());
    std::vector<uint32_t> ids;
    for (auto e : es) {
      ids.push_back(e[0]);
      ids.push_back(e[1]);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    auto c = select(g, ids, es);
    c.metadata.physical = false;
    c.metadata.transform = cpu_identity;
    out.push_back(std::move(c));
  }
  return out;
}
double squared(V a, V b, bool wide = false) {
  if (wide) {
    double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return (x * x + y * y) + z * z;
  }
  float x = float(a[0]) - float(b[0]), y = float(a[1]) - float(b[1]), z = float(a[2]) - float(b[2]);
  return (x * x + y * y) + z * z;
}
double length(double square, bool wide) {
  return wide ? std::sqrt(square) : double(std::sqrt(float(square)));
}
// NumPy 2.4.6 pairwise reduction (cpp/licenses/NumPy.txt).
template <class T> T pairwise(const T *v, size_t n) {
  if (n < 8) {
    T o = -T(0);
    for (size_t i = 0; i < n; ++i)
      o += v[i];
    return o;
  }
  if (n <= 128) {
    T r[8];
    std::copy(v, v + 8, r);
    size_t i = 8;
    for (; i < n - n % 8; i += 8)
      for (int j = 0; j < 8; ++j)
        r[j] += v[i + j];
    T o = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i)
      o += v[i];
    return o;
  }
  size_t m = n / 2;
  m -= m % 8;
  return pairwise(v, m) + pairwise(v + m, n - m);
}
G dust(G g, double threshold) {
  if (empty(g) || threshold == 0)
    return g;
  std::vector<G> keep;
  for (auto &c : components(g)) {
    if (!c.metadata.physical) {
      for (auto &v : c.skeleton.vertices)
        v = cpu_transform_point(v, c.metadata.transform, c.skeleton.vertex64);
      c.metadata.physical = true;
    }
    double total;
    if (c.skeleton.vertex64) {
      std::vector<double> d;
      for (auto e : c.skeleton.edges)
        d.push_back(
            length(squared(c.skeleton.vertices[e[0]], c.skeleton.vertices[e[1]], true), true));
      total = pairwise(d.data(), d.size());
    } else {
      std::vector<float> d;
      for (auto e : c.skeleton.edges)
        d.push_back(
            float(length(squared(c.skeleton.vertices[e[0]], c.skeleton.vertices[e[1]]), false)));
      total = pairwise(d.data(), d.size());
    }
    if (total > threshold)
      keep.push_back(std::move(c));
  }
  return merge(keep);
}
std::vector<std::vector<uint32_t>> adjacency(const G &g) {
  std::vector<std::vector<uint32_t>> a(g.sources.size());
  for (auto e : g.skeleton.edges) {
    for (int k = 0; k < 2; ++k)
      if (std::find(a[e[k]].begin(), a[e[k]].end(), e[1 - k]) == a[e[k]].end())
        a[e[k]].push_back(e[1 - k]);
  }
  return a;
}
std::vector<uint32_t> cycle(const G &g) {
  if (empty(g))
    return {};
  auto a = adjacency(g);
  struct Item {
    uint32_t n, p, d;
  };
  std::vector<Item> stack{{g.skeleton.edges[0][0], UINT32_MAX, 0}};
  std::vector<uint32_t> path;
  std::vector<bool> visited(a.size());
  uint32_t node = 0;
  while (!stack.empty()) {
    auto x = stack.back();
    stack.pop_back();
    node = x.n;
    path.resize(x.d);
    path.push_back(x.n);
    if (visited[x.n])
      break;
    visited[x.n] = true;
    for (auto c : a[x.n])
      if (c != x.p)
        stack.push_back({c, x.n, x.d + 1});
  }
  if (path.size() <= 1)
    return {};
  auto pos = std::find(path.begin(), path.end() - 1, node);
  if (path.end() - pos < 3)
    return {};
  return {pos, path.end()};
}
std::vector<E> path_edges(const std::vector<uint32_t> &p) {
  std::vector<E> es;
  for (size_t i = 1; i < p.size(); ++i) {
    E e{p[i - 1], p[i]};
    if (e[0] > e[1])
      std::swap(e[0], e[1]);
    es.push_back(e);
  }
  return es;
}
void remove_edges(G &g, const std::vector<E> &remove) {
  auto &es = g.skeleton.edges;
  for (auto &e : es)
    if (e[0] > e[1])
      std::swap(e[0], e[1]);
  es.erase(std::remove_if(
               es.begin(), es.end(),
               [&](auto e) { return std::find(remove.begin(), remove.end(), e) != remove.end(); }),
           es.end());
}
G loops_one(G g) {
  auto &vs = g.skeleton.vertices;
  bool wide = g.skeleton.vertex64;
  for (;;) {
    auto path = cycle(g);
    if (path.empty())
      break;
    auto ce = path_edges(path);
    std::vector<uint32_t> nodes = path;
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    std::vector<int> degree(vs.size());
    for (auto e : g.skeleton.edges) {
      ++degree[e[0]];
      ++degree[e[1]];
    }
    std::vector<uint32_t> branches;
    for (auto n : nodes)
      if (degree[n] >= 3)
        branches.push_back(n);
    if (branches.size() == 0) {
      remove_edges(g, ce);
    } else if (branches.size() == 1) {
      auto b = branches[0], far = nodes[0];
      double d = -1;
      for (auto n : nodes) {
        auto nd = squared(vs[n], vs[b], wide);
        if (nd > d) {
          far = n;
          d = nd;
        }
      }
      remove_edges(g, ce);
      g.skeleton.edges.push_back({b, far});
    } else if (branches.size() == 2) {
      path.erase(path.begin());
      std::vector<size_t> pos;
      for (size_t i = 0; i < path.size(); ++i)
        if (path[i] == branches[0] || path[i] == branches[1])
          pos.push_back(i);
      std::vector<uint32_t> shortpath;
      if (pos[1] - pos[0] < double(path.size()) / 2)
        shortpath = {path.begin() + pos[0], path.begin() + pos[1] + 1};
      else {
        shortpath = {path.begin() + pos[1], path.end()};
        shortpath.insert(shortpath.end(), path.begin(), path.begin() + pos[0] + 1);
      }
      auto keep = path_edges(shortpath);
      std::vector<E> drop;
      for (auto e : ce)
        if (std::find(keep.begin(), keep.end(), e) == keep.end())
          drop.push_back(e);
      remove_edges(g, drop);
    } else {
      V centroid{};
      for (auto b : branches)
        for (int j = 0; j < 3; ++j)
          centroid[j] =
              wide ? centroid[j] + vs[b][j] : double(float(centroid[j]) + float(vs[b][j]));
      for (auto &x : centroid)
        x = wide ? x / double(branches.size()) : double(float(x) / float(branches.size()));
      uint32_t closest = 0;
      double d = inf;
      for (uint32_t i = 0; i < vs.size(); ++i) {
        double nd = squared(vs[i], centroid, wide);
        if (nd < d) {
          d = nd;
          closest = i;
        }
      }
      d = 0;
      for (auto b : branches)
        d = std::max(d, squared(vs[b], vs[closest], wide));
      if (length(d, wide) > g.skeleton.radii[closest])
        remove_edges(g, {ce[0]});
      else {
        remove_edges(g, ce);
        for (auto b : branches)
          if (b != closest)
            g.skeleton.edges.push_back({b, closest});
      }
    }
  }
  return g;
}
G loops(G g) {
  if (empty(g))
    return g;
  std::vector<G> out;
  for (auto c : components(g))
    out.push_back(loops_one(std::move(c)));
  return clean(merge(out), false);
}
G join(std::vector<G> inputs, double radius, bool restricted, bool radius_float32 = false,
       bool radius_double_compare = false, CpuJoinTrace *trace = nullptr) {
  if (radius <= 0)
    throw std::invalid_argument("radius must be greater than zero");
  std::vector<G> gs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto parts = components(inputs[i]);
    for (auto &c : parts)
      if (!empty(c)) {
        gs.push_back(clean(c));
        if (trace)
          trace->pieces.push_back(parts.size() > 1 ? -1 : int64_t(i));
      }
  }
  if (gs.size() < 2)
    return gs.empty() ? G{} : gs[0];
  for (const auto &g : gs)
    if (empty(g))
      throw std::invalid_argument("cannot build nearest tree for an empty component");
  bool radius_wide = radius_double_compare;
  if (restricted) {
    bool radii_wide = false;
    double r = -inf;
    for (const auto &g : gs) {
      radii_wide |= g.skeleton.radius64;
      for (auto x : g.skeleton.radii) {
        if (std::isnan(x)) {
          r = x;
          break;
        }
        r = std::max(r, x);
      }
    }
    radius = std::max(radii_wide ? 2 * r : double(float(2 * float(r))), 0.0);
    radius_float32 = !radii_wide;
    radius_wide = radii_wide;
  }
  auto compute = [&](size_t i, std::vector<float> &ds, std::vector<E> &pairs) {
    auto &a = gs[i].skeleton;
    std::vector<float> af;
    if (!a.vertex64) {
      af.reserve(3 * a.vertices.size());
      for (auto v : a.vertices)
        for (auto x : v)
          af.push_back(float(x));
    }
    auto destroy = a.vertex64 ? brook_cpu_tree64_delete : brook_cpu_tree_delete;
    std::unique_ptr<void, decltype(destroy)> tree(
        a.vertex64 ? brook_cpu_tree64_create(a.vertices[0].data(), a.vertices.size())
                   : brook_cpu_tree_create(af.data(), a.vertices.size()),
        destroy);
    if (!tree)
      throw std::bad_alloc();
    double bound =
        radius_float32 ? double(float(float(radius) + float(0.000001))) : radius + 0.000001;
    for (size_t j = i + 1; j < gs.size(); ++j) {
      auto &b = gs[j].skeleton;
      std::vector<uint32_t> idx(b.vertices.size());
      size_t k;
      double r;
      if (a.vertex64) {
        std::vector<double> d(b.vertices.size());
        brook_cpu_tree64_query(tree.get(), a.vertices[0].data(), b.vertices[0].data(), d.size(),
                               bound * bound, idx.data(), d.data());
        k = std::min_element(d.begin(), d.end()) - d.begin();
        r = d[k];
      } else {
        if (b.vertex64)
          throw CpuQueryDtypeError("Type mismatch. query points must be of type float32 when "
                                   "data points are of type float32");
        std::vector<float> bf;
        bf.reserve(3 * b.vertices.size());
        for (auto v : b.vertices)
          for (auto x : v)
            bf.push_back(float(x));
        std::vector<float> d(b.vertices.size());
        brook_cpu_tree_query(tree.get(), af.data(), bf.data(), d.size(), float(bound * bound),
                             idx.data(), d.data());
        k = std::min_element(d.begin(), d.end()) - d.begin();
        r = d[k];
      }
      if (!std::isinf(r) && idx[k] >= a.vertices.size())
        throw std::invalid_argument("nearest distance overflow");
      if (restricted && !std::isinf(r)) {
        double limit = a.radii[idx[k]] + b.radii[k];
        if (!a.radius64 && !b.radius64)
          limit = float(limit);
        if (r > limit)
          r = inf;
      }
      ds[i * gs.size() + j] = ds[j * gs.size() + i] = float(r);
      pairs[i * gs.size() + j] = {idx[k], uint32_t(k)};
    }
  };
  size_t n = gs.size();
  std::vector<float> ds(n * n, inf);
  std::vector<E> pairs(n * n);
  for (size_t i = 0; i < n; ++i)
    compute(i, ds, pairs);
  while (gs.size() > 1) {
    size_t ix = std::min_element(ds.begin(), ds.end()) - ds.begin();
    float r = ds[ix];
    if (std::isinf(r) || (radius_wide ? double(r) > radius : r > float(radius)))
      break;
    size_t i = ix / n, j = ix % n;
    G fused = merge({gs[i], gs[j]});
    auto p = pairs[ix];
    fused.skeleton.edges.push_back({p[0], uint32_t(p[1] + gs[i].sources.size())});
    if (trace)
      trace->joined = true;
    std::vector<G> next;
    next.push_back(std::move(fused));
    std::vector<size_t> ids;
    for (size_t k = 0; k < n; ++k)
      if (k != i && k != j) {
        ids.push_back(k);
        next.push_back(std::move(gs[k]));
      }
    gs = std::move(next);
    size_t nn = gs.size();
    std::vector<float> nd(nn * nn, inf);
    std::vector<E> np(nn * nn);
    for (size_t a = 0; a < ids.size(); ++a)
      for (size_t b = 0; b < ids.size(); ++b) {
        nd[(a + 1) * nn + b + 1] = ds[ids[a] * n + ids[b]];
        np[(a + 1) * nn + b + 1] = pairs[ids[a] * n + ids[b]];
      }
    ds = std::move(nd);
    pairs = std::move(np);
    n = nn;
    compute(0, ds, pairs);
  }
  return clean(merge(gs));
}
struct DE {
  E e;
  double d;
  bool live = true;
};
G ticks_one(G g, double threshold, bool threshold_float32) {
  if (g.skeleton.vertex64)
    throw std::invalid_argument("Buffer dtype mismatch, expected 'float' but got 'double'");
  auto adj = adjacency(g);
  std::vector<int> degree(adj.size()), counts(adj.size());
  uint32_t start = UINT32_MAX;
  for (uint32_t i = 0; i < adj.size(); ++i) {
    degree[i] = adj[i].size();
    if (degree[i] == 1 && start == UINT32_MAX)
      start = i;
    if (degree[i] >= 3)
      counts[i] = degree[i];
  }
  if (start == UINT32_MAX)
    throw std::invalid_argument("tick graph has no terminal vertex");
  struct Walk {
    uint32_t n, p, root;
    float d;
  };
  std::vector<Walk> stack{{start, UINT32_MAX, start, 0}};
  std::vector<bool> visited(adj.size());
  // Kimimaro's Cython wrapper (skeletontricks) materializes Python dict entries in this map's
  // iteration order. Sorting these keys would change equal-length tick choices.
  std::unordered_map<uint64_t, float> dg;
  while (!stack.empty()) {
    auto x = stack.back();
    stack.pop_back();
    if (visited[x.n])
      throw std::invalid_argument("cycle detected in tick graph");
    visited[x.n] = true;
    if (degree[x.n] != 2 && x.n != x.root) {
      auto lo = std::min(x.n, x.root), hi = std::max(x.n, x.root);
      dg[uint64_t(lo) | (uint64_t(hi) << 32)] = x.d;
      x.d = 0;
      x.root = x.n;
    }
    for (auto c : adj[x.n])
      if (c != x.p) {
        float d = squared(g.skeleton.vertices[x.n], g.skeleton.vertices[c]);
        stack.push_back({c, x.n, x.root, float(double(x.d) + std::sqrt(double(d)))});
      }
  }
  std::vector<DE> edges;
  CpuPairSet terminal;
  for (auto [k, d] : dg) {
    E e{uint32_t(k >> 32), uint32_t(k)};
    edges.push_back({e, d});
    if (degree[e[0]] == 1 || degree[e[1]] == 1)
      terminal.add(e);
  }
  auto lookup = [&](E e) -> DE & {
    for (auto &d : edges)
      if (d.live && d.e == e)
        return d;
    throw std::runtime_error("invalid tick superedge");
  };
  auto fuse = [&](uint32_t v) {
    IntegerSet ends;
    double d = 0;
    for (auto &e : edges)
      if (e.live && (e.e[0] == v || e.e[1] == v)) {
        terminal.discard(e.e);
        d += e.d;
        e.live = false;
        ends.add(e.e[0]);
        ends.add(e.e[1]);
      }
    ends.discard(v);
    auto ns = ends.values();
    if (ns.size() != 2)
      throw std::runtime_error("invalid tick fusion");
    E e{uint32_t(ns[0]), uint32_t(ns[1])};
    edges.push_back({e, d});
    terminal.add(e);
    counts[v] = 0;
  };
  for (;;) {
    size_t live = 0;
    for (auto &e : edges)
      live += e.live;
    if (live <= 1)
      break;
    auto ts = terminal.values();
    if (ts.empty())
      throw std::runtime_error("no terminal tick");
    E best = ts[0];
    for (auto e : ts)
      if (lookup(e).d < lookup(best).d)
        best = e;
    auto a = best[0], b = best[1];
    if ((counts[a] == 1 && counts[b] == 1) ||
        (threshold_float32 ? float(lookup(best).d) >= float(threshold)
                           : lookup(best).d >= threshold))
      break;
    // Unique path in this tree. Adjacency keeps NetworkX insertion order.
    std::vector<uint32_t> parent(adj.size(), UINT32_MAX), queue{a};
    parent[a] = a;
    for (size_t q = 0; q < queue.size() && parent[b] == UINT32_MAX; ++q)
      for (auto c : adj[queue[q]])
        if (parent[c] == UINT32_MAX) {
          parent[c] = queue[q];
          queue.push_back(c);
        }
    if (parent[b] == UINT32_MAX)
      throw std::runtime_error("disconnected tick superedge");
    for (auto v = b; v != a; v = parent[v]) {
      auto p = parent[v];
      auto &av = adj[v], &ap = adj[p];
      av.erase(std::find(av.begin(), av.end(), p));
      ap.erase(std::find(ap.begin(), ap.end(), v));
    }
    lookup(best).live = false;
    terminal.discard(best);
    --counts[a];
    --counts[b];
    if (counts[a] == 2)
      fuse(a);
    if (counts[b] == 2)
      fuse(b);
  }
  // NetworkX nodes retain first edge-appearance order, including removed nodes.
  std::vector<uint32_t> nodeorder;
  std::vector<bool> seen(adj.size());
  for (auto e : g.skeleton.edges)
    for (auto v : e)
      if (!seen[v]) {
        seen[v] = true;
        nodeorder.push_back(v);
      }
  std::fill(seen.begin(), seen.end(), false);
  g.skeleton.edges.clear();
  for (auto v : nodeorder) {
    for (auto c : adj[v])
      if (!seen[c])
        g.skeleton.edges.push_back({v, c});
    seen[v] = true;
  }
  return g;
}
G ticks(G g, double threshold, bool threshold_float32) {
  if (empty(g) || threshold == 0)
    return g;
  std::vector<G> out;
  for (auto c : components(g))
    out.push_back(ticks_one(std::move(c), threshold, threshold_float32));
  return clean(merge(out), false);
}
} // namespace
CpuSkeleton cpu_postprocess(CpuGeometry s, double d, double t,
                            const CpuMetadata &metadata, bool tick_float32) {
  auto label = s.label;
  auto g = clean(input(std::move(s), 0, metadata));
  g = dust(std::move(g), d);
  g = loops(std::move(g));
  g = join({g}, inf, true);
  g = ticks(std::move(g), t, tick_float32);
  g = clean(g);
  g.skeleton.label = label;
  return g;
}
CpuSkeleton cpu_join_precise(std::vector<CpuGeometry> ss, double r, bool restrict,
                             const std::vector<CpuMetadata> &metadata, bool radius_float32,
                             bool radius_double_compare, CpuJoinTrace *trace) {
  if (!metadata.empty() && metadata.size() != ss.size())
    throw std::invalid_argument("metadata/input count mismatch");
  std::vector<G> gs;
  uint64_t offset = 0;
  for (size_t i = 0; i < ss.size(); ++i) {
    auto &s = ss[i];
    auto n = s.vertices.size();
    gs.push_back(input(std::move(s), offset, metadata.empty() ? CpuMetadata{} : metadata[i]));
    offset += n;
  }
  return join(std::move(gs), r, restrict, radius_float32, radius_double_compare, trace);
}
CpuSkeleton cpu_join(std::vector<Skeleton> inputs, double radius, bool restricted) {
  std::vector<CpuGeometry> gs;
  for (auto &s : inputs)
    gs.emplace_back(std::move(s));
  return cpu_join_precise(std::move(gs), radius, restricted);
}
CpuSkeleton cpu_fuse(std::vector<CpuGeometry> inputs,
                     const std::vector<CpuMetadata> &metadata,
                     const std::vector<CpuCrop> &boxes,std::vector<size_t> *attribute_inputs) {
  if(metadata.size()!=inputs.size()||(!boxes.empty()&&boxes.size()!=inputs.size()))
    throw std::invalid_argument("fusion metadata/crop count mismatch");
  std::vector<G> fragments;uint64_t offset=0;
  for(size_t i=0;i<inputs.size();++i) {
    size_t count=inputs[i].vertices.size();auto g=input(std::move(inputs[i]),offset,metadata[i]);offset+=count;
    if(!boxes.empty()&&!empty(g)) {
      auto b=boxes[i].bounds;
      // Igneous leaves a fragment unchanged when the shrunken box has no volume.
      if(boxes[i].apply) {
        std::vector<uint8_t> valid(count);size_t first=count;
        for(size_t j=0;j<count;++j) {
          auto v=g.skeleton.vertices[j];valid[j]=v[0]>=b[0]&&v[1]>=b[1]&&v[2]>=b[2]&&v[0]<=b[3]&&v[1]<=b[4]&&v[2]<=b[5];
          if(valid[j]&&first==count) first=j;
        }
        if(first==count) {g.skeleton.vertices.clear();g.skeleton.edges.clear();g.skeleton.radii.clear();g.sources.clear();}
        else {
          // Osteoid replaces invalid coordinates before consolidating, so an
          // earlier invalid vertex may supply attributes at the first valid one.
          for(size_t j=0;j<count;++j) if(!valid[j]) g.skeleton.vertices[j]=g.skeleton.vertices[first];
          auto &edges=g.skeleton.edges;
          edges.erase(std::remove_if(edges.begin(),edges.end(),[&](auto e){return !valid[e[0]]||!valid[e[1]];}),edges.end());
          g=clean(g);
        }
      }
    }
    if(attribute_inputs&&!empty(g)) attribute_inputs->push_back(i);
    fragments.push_back(std::move(g));
  }
  return clean(merge(fragments));
}
std::pair<CpuSkeleton,double> cpu_physical_length(CpuGeometry s,const CpuMetadata &metadata) {
  auto g=input(std::move(s),0,metadata);
  if(!g.metadata.physical) {
    for(auto &v:g.skeleton.vertices) v=cpu_transform_point(v,g.metadata.transform,g.skeleton.vertex64);
    g.metadata.physical=true;
  }
  double total=0;
  if(g.skeleton.vertex64) {
    std::vector<double> distances;distances.reserve(g.skeleton.edges.size());
    for(auto e:g.skeleton.edges) distances.push_back(length(squared(g.skeleton.vertices[e[0]],g.skeleton.vertices[e[1]],true),true));
    if(!distances.empty()) total=pairwise(distances.data(),distances.size());
  } else {
    std::vector<float> distances;distances.reserve(g.skeleton.edges.size());
    for(auto e:g.skeleton.edges) distances.push_back(float(length(squared(g.skeleton.vertices[e[0]],g.skeleton.vertices[e[1]]),false)));
    if(!distances.empty()) total=pairwise(distances.data(),distances.size());
  }
  return {std::move(g),total};
}
} // namespace brook
