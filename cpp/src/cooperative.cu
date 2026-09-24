// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "cooperative.cuh"
#include <cooperative_groups.h>
#include <chrono>
#include <map>
#include <mutex>

namespace brook {
namespace {
__global__ __launch_bounds__(128,8) void barrier_probe(int rounds) {
  auto grid=cooperative_groups::this_grid();
  for(int i=0;i<rounds;++i) { __threadfence(); grid.sync(); }
}
std::mutex cache_mutex;
std::map<int,int> cache;
}
int barrier_grid(Context &ctx) {
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto found=cache.find(ctx.device);
  if(found!=cache.end()) return found->second;
  int per_sm=0,sms=0;
  BROOK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,barrier_probe,128,0));
  BROOK_CUDA(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,ctx.device));
  int top=per_sm*sms;
  std::vector<int> grids;
  for(int b=16;b<top;b*=2) grids.push_back(b);
  grids.push_back(top);
  std::vector<double> times;
  for(int blocks:grids) {
    double best=1e30;
    for(int sample=0;sample<4;++sample) {
      ctx.synchronize();
      auto start=std::chrono::steady_clock::now();
      cooperative_launch(ctx,barrier_probe,blocks,sample?1000:0);
      ctx.synchronize();
      double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      if(sample) best=std::min(best,elapsed);
    }
    times.push_back(best);
  }
  double threshold=*std::min_element(times.begin(),times.end())*1.25;
  int chosen=grids.front();
  for(size_t i=0;i<grids.size();++i) if(times[i]<=threshold) chosen=grids[i];
  cache[ctx.device]=chosen;
  return chosen;
}
}
