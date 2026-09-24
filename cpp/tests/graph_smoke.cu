#include "graph.hpp"
#include "cooperative.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

// setenv/unsetenv are POSIX; the Microsoft C runtime has _putenv_s (an empty value unsets).
static void set_env(const char *name,const char *value) {
#ifdef _WIN32
  _putenv_s(name,value?value:"");
#else
  if(value) setenv(name,value,1); else unsetenv(name);
#endif
}

__global__ void body(int *counter,cudaGraphConditionalHandle condition) {
  auto grid=cooperative_groups::this_grid();
  __threadfence();grid.sync();
  if(blockIdx.x==0 && threadIdx.x==0) {
    ++*counter;cudaGraphSetConditional(condition,*counter<7);
  }
  __threadfence();grid.sync();
}
__global__ void increment(int *counter) {
  auto grid=cooperative_groups::this_grid();
  __threadfence();grid.sync();
  if(blockIdx.x==0 && threadIdx.x==0) ++*counter;
  __threadfence();grid.sync();
}
__global__ void join(const int *a,const int *b,cudaGraphConditionalHandle condition) {
  cudaGraphSetConditional(condition,*a<7 || *b<7);
}
int main() {
  brook::Context ctx(0);
  brook::Buffer<int> counter(ctx,1);counter.clear(ctx);
  // Compute Sanitizer sets up its device-side state for a module on that module's first host-side
  // kernel launch; a kernel whose first launch happens inside a conditional node (issued by the
  // device-side graph executor) faults under initcheck and synccheck otherwise. The engine always
  // host-launches its modules before capturing a graph; do the same here so memcheck and initcheck
  // pass. synccheck and racecheck still abort on the cooperative kernels followed by other nodes
  // below: their barrier tracking breaks when such a kernel shares a conditional body with other nodes.
  brook::cooperative_launch(ctx,increment,1,counter.data());ctx.synchronize();counter.clear(ctx);
  brook::WhileGraph loop(ctx);
  loop.capture([&]{brook::cooperative_launch(ctx,body,2,counter.data(),loop.condition());});
  loop.run();
  if(counter.get(ctx,1)[0]!=7) return 1;
  counter.clear(ctx);loop.run();
  if(counter.get(ctx,1)[0]!=7) return 2;
  counter.clear(ctx);brook::Buffer<int> other(ctx,1);other.clear(ctx);
  brook::WhileGraph dag(ctx);
  auto a=dag.capture_segment([&]{brook::cooperative_launch(ctx,increment,2,counter.data());});
  auto b=dag.capture_segment([&]{brook::cooperative_launch(ctx,increment,2,other.data());});
  a.insert(a.end(),b.begin(),b.end());
  dag.capture_segment([&]{join<<<1,1,0,ctx.stream>>>(counter.data(),other.data(),dag.condition());},a);
  dag.finish();dag.run();
  if(counter.get(ctx,1)[0]!=7 || other.get(ctx,1)[0]!=7) return 3;
  // A graph that cannot be built raises a CudaError that says so; nothing falls back to a host loop.
  int per_sm=0,sms=0;
  BROOK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,body,128,0));
  BROOK_CUDA(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,ctx.device));
  // The invalidated capture destroys the graph captured into (the WHILE body, or the whole fixed
  // graph); WhileGraph must not touch it again.
  for(int fixed=0;fixed<2;++fixed) {
    counter.clear(ctx);
    bool raised=false;
    try {
      std::unique_ptr<brook::WhileGraph> broken=fixed?std::make_unique<brook::FixedGraph>(ctx):std::make_unique<brook::WhileGraph>(ctx);
      broken->capture([&]{brook::cooperative_launch(ctx,body,per_sm*sms+1,counter.data(),broken->condition());});
      broken->run();
    } catch(const brook::CudaError &error) {
      std::string text=error.what();raised=true;
      std::printf("expected graph failure: %s\n",error.what());
      if(text.find("Brook could not capture its CUDA graph")==std::string::npos || text.find("no host-loop fallback")==std::string::npos ||
         error.code!=cudaErrorCooperativeLaunchTooLarge) return 4;
    }
    if(!raised || counter.get(ctx,1)[0]!=0) return 5;
    counter.clear(ctx);loop.run();   // the failure left nothing behind: the earlier graph still runs
    if(counter.get(ctx,1)[0]!=7) return 6;
  }
  // BROOK_GRAPHS is the explicit debugging switch, read the same way everywhere.
  const char *off[]={"0","off"," False ","no"},*on[]={"1","on",""};
  for(auto value:off) { set_env("BROOK_GRAPHS",value);if(brook::graphs_enabled()) return 7; }
  for(auto value:on) { set_env("BROOK_GRAPHS",value);if(!brook::graphs_enabled()) return 8; }
  set_env("BROOK_GRAPHS",nullptr);if(!brook::graphs_enabled()) return 9;
  return 0;
}
