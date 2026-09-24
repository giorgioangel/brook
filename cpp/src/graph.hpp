// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include <functional>

namespace brook {
// Brook's loops run as CUDA graphs. BROOK_GRAPHS=0 (or off/false/no) selects the host loops, for
// debugging only (e.g. racecheck/synccheck); a graph that fails raises, it never falls back.
bool graphs_enabled();
class WhileGraph {
  Context &context_;
  cudaGraph_t graph_=nullptr,body_=nullptr;
  cudaGraphExec_t executable_=nullptr;
  cudaStream_t stream_=nullptr;
  cudaEvent_t begin_=nullptr,end_=nullptr;
  cudaGraphConditionalHandle handle_=0;
 public:
  explicit WhileGraph(Context &ctx);
  WhileGraph(Context &ctx,bool conditional);
  ~WhileGraph();
  WhileGraph(const WhileGraph &)=delete;
  WhileGraph &operator=(const WhileGraph &)=delete;
  cudaGraphConditionalHandle condition() const { return handle_; }
  std::vector<cudaGraphNode_t> capture_segment(const std::function<void()> &body,
                                               const std::vector<cudaGraphNode_t> &dependencies={});
  void finish();
  void capture(const std::function<void()> &body);
  void launch();
  float wait();
  void run() { launch();wait(); }
};
class FixedGraph : public WhileGraph {
 public:
  explicit FixedGraph(Context &ctx):WhileGraph(ctx,false) {}
};
}
