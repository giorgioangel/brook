// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "graph.hpp"
#include "device_check.hpp"
#include <type_traits>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace brook {
bool graphs_enabled() {
  const char *value=std::getenv("BROOK_GRAPHS");if(!value) return true;
  std::string text=value;
  auto first=text.find_first_not_of(" \t\r\n"),last=text.find_last_not_of(" \t\r\n");
  text=first==std::string::npos?"":text.substr(first,last-first+1);
  for(char &c:text) c=char(std::tolower(static_cast<unsigned char>(c)));
  return text!="0" && text!="off" && text!="false" && text!="no";
}
namespace {
// Every CUDA error of a graph step is reported with what Brook was doing; there is no host-loop
// fallback to hide it behind.
[[noreturn]] void graph_failure(const Context &ctx,const char *step,const CudaError &error) {
  cudaGetLastError();   // the error travels in the exception; do not leave it for the next launch check
  int driver=0,runtime=0;cudaDriverGetVersion(&driver);cudaRuntimeGetVersion(&runtime);
  std::string device="CUDA device "+std::to_string(ctx.device);
  cudaDeviceProp properties{};
  if(cudaGetDeviceProperties(&properties,ctx.device)==cudaSuccess)
    device+=" ("+std::string(properties.name)+", compute capability "+std::to_string(properties.major)+"."+std::to_string(properties.minor)+")";
  else cudaGetLastError();
  // Launching and waiting also return an error that earlier GPU work on the stream or context left
  // behind, so those two do not blame the graph alone.
  const std::string what=step;
  std::string lead=what=="run"?"Brook's CUDA graph failed while it ran (":"Brook could not "+what+" its CUDA graph (";
  lead+=error.what();
  lead+=what=="run" || what=="launch"?"); the error may also come from GPU work queued before the graph":")";
  throw CudaError(CudaError::Message{},error.code,lead+
    ". Brook runs its GPU loops as CUDA graphs and has no host-loop fallback; this is a Brook bug "
    "or an unsupported CUDA environment ("+device+", driver CUDA "+cuda_version_string(driver)+", runtime CUDA "+cuda_version_string(runtime)+
    "). BROOK_GRAPHS=0 runs the host loops, for debugging only.");
}
template<class Step> void graph_step(const Context &ctx,const char *step,Step &&body) {
  try { body(); }
  catch(const CudaError &error) { graph_failure(ctx,step,error); }
}
// CUDA 12.3's entry point has no edge-data argument; later headers add one.
// Select by the actual declaration so both builds retain the same graph body.
template<class Function> cudaError_t add_node(Function fn,cudaGraphNode_t *node,cudaGraph_t graph,cudaGraphNodeParams *parameters) {
  if constexpr(std::is_invocable_v<Function,cudaGraphNode_t*,cudaGraph_t,const cudaGraphNode_t*,
                                  const cudaGraphEdgeData*,size_t,cudaGraphNodeParams*>)
    return fn(node,graph,nullptr,nullptr,0,parameters);
  else return fn(node,graph,nullptr,0,parameters);
}
}
WhileGraph::WhileGraph(Context &ctx):WhileGraph(ctx,true) {}
WhileGraph::WhileGraph(Context &ctx,bool conditional):context_(ctx) {
  ctx.activate();
  try {
    graph_step(ctx,"create",[&] {
      BROOK_CUDA(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
      BROOK_CUDA(cudaEventCreate(&begin_));BROOK_CUDA(cudaEventCreate(&end_));
      BROOK_CUDA(cudaGraphCreate(&graph_,0));
      if(!conditional) { body_=graph_;return; }
      BROOK_CUDA(cudaGraphConditionalHandleCreate(&handle_,graph_,1,cudaGraphCondAssignDefault));
      cudaGraphNodeParams parameters{};
      parameters.type=cudaGraphNodeTypeConditional;parameters.conditional.handle=handle_;
      parameters.conditional.type=cudaGraphCondTypeWhile;parameters.conditional.size=1;
      cudaGraphNode_t node;
      BROOK_CUDA(add_node(cudaGraphAddNode,&node,graph_,&parameters));
      body_=parameters.conditional.phGraph_out[0];
    });
  } catch(...) {
    if(graph_) cudaGraphDestroy(graph_);
    if(begin_) cudaEventDestroy(begin_);
    if(end_) cudaEventDestroy(end_);
    if(stream_) cudaStreamDestroy(stream_);
    throw;
  }
}
WhileGraph::~WhileGraph() {
  DestructorDeviceGuard device(context_.device);
  if(stream_) cudaStreamSynchronize(stream_);
  if(executable_) cudaGraphExecDestroy(executable_);
  if(graph_) cudaGraphDestroy(graph_);
  if(begin_) cudaEventDestroy(begin_);
  if(end_) cudaEventDestroy(end_);
  if(stream_) cudaStreamDestroy(stream_);
}
std::vector<cudaGraphNode_t> WhileGraph::capture_segment(const std::function<void()> &body,const std::vector<cudaGraphNode_t> &dependencies) {
  context_.activate();context_.synchronize();
  if(executable_) throw std::logic_error("graph is already captured");
  if(!graph_) throw std::logic_error("graph capture failed earlier");
  auto nodes=[&] {
    size_t count=0;BROOK_CUDA(cudaGraphGetNodes(body_,nullptr,&count));
    std::vector<cudaGraphNode_t> result(count);BROOK_CUDA(cudaGraphGetNodes(body_,result.data(),&count));return result;
  };
  std::vector<cudaGraphNode_t> after;
  graph_step(context_,"capture",[&] {
    auto before=nodes();
    cudaStream_t previous=context_.stream;
    BROOK_CUDA(cudaStreamBeginCaptureToGraph(stream_,body_,dependencies.empty()?nullptr:dependencies.data(),nullptr,dependencies.size(),cudaStreamCaptureModeThreadLocal));
    context_.stream=stream_;
    // An invalidated capture (a launch in the body failed) makes the driver destroy the graph it
    // captured into: the whole graph for a fixed graph, the conditional node's body graph for a
    // WHILE graph, whose parent then cannot be destroyed either (it crashes). Neither is touched
    // again; the parent of a WHILE body leaks, on this error path only.
    auto end_capture=[&] {
      context_.stream=previous;
      cudaGraph_t captured=nullptr;cudaError_t status=cudaStreamEndCapture(stream_,&captured);
      if(status==cudaSuccess && !captured) status=cudaErrorStreamCaptureInvalidated;
      if(status!=cudaSuccess) { graph_=nullptr;body_=nullptr;cudaGetLastError(); }
      return status;
    };
    try { body(); }
    catch(...) { end_capture();throw; }
    check(end_capture(),"cudaStreamEndCapture(stream_,&captured)");
    after=nodes();
    after.erase(std::remove_if(after.begin(),after.end(),[&](auto node){return std::find(before.begin(),before.end(),node)!=before.end();}),after.end());
  });
  return after;
}
void WhileGraph::finish() {
  if(executable_) throw std::logic_error("graph is already instantiated");
  if(!graph_) throw std::logic_error("graph capture failed earlier");
  graph_step(context_,"instantiate",[&]{ BROOK_CUDA(cudaGraphInstantiate(&executable_,graph_,0)); });
}
void WhileGraph::capture(const std::function<void()> &body) { capture_segment(body);finish(); }
void WhileGraph::launch() {
  context_.activate();
  if(!executable_) throw std::logic_error("graph has not been captured");
  graph_step(context_,"launch",[&] {
    BROOK_CUDA(cudaEventRecord(begin_,context_.stream));
    BROOK_CUDA(cudaStreamWaitEvent(stream_,begin_,0));
    BROOK_CUDA(cudaEventRecord(begin_,stream_));
    BROOK_CUDA(cudaGraphLaunch(executable_,stream_));
    BROOK_CUDA(cudaEventRecord(end_,stream_));
  });
}
float WhileGraph::wait() {
  context_.activate();
  // An error here comes from a kernel inside the graph, or from GPU work queued before it.
  float ms=0;
  graph_step(context_,"run",[&] { BROOK_CUDA(cudaEventSynchronize(end_));BROOK_CUDA(cudaEventElapsedTime(&ms,begin_,end_)); });
  return ms*0.001f;
}
}
