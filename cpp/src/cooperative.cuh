// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace brook {
int barrier_grid(Context &ctx);
// Records a cooperative grid in the call's stats (the largest one used under that key).
inline void note_grid(Context &ctx,const char *key,int64_t blocks) {
  auto &value=ctx.stats[key];value=std::max(value,blocks);
}
// The stats key of a grid override variable: grid_override for BROOK_GRID, grid_override_lockstep
// for BROOK_LOCKSTEP_GRID, grid_override_flood for BROOK_FLOOD_GRID, and so on.
inline std::string grid_override_key(const char *variable) {
  std::string name=variable;
  if(name.rfind("BROOK_",0)==0) name.erase(0,6);
  if(name=="GRID") name.clear();
  else if(name.size()>5 && name.compare(name.size()-5,5,"_GRID")==0) name.erase(name.size()-5);
  for(char &c:name) c=char(std::tolower(static_cast<unsigned char>(c)));
  return name.empty()?std::string("grid_override"):"grid_override_"+name;
}
template<class Function> int cooperative_grid(Context &ctx, Function kernel, size_t n, const char *override_name) {
  int per_sm=0, sms=0;
  BROOK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,kernel,128,0));
  BROOK_CUDA(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,ctx.device));
  int cap=per_sm*sms;
  const char *variable=override_name,*setting=std::getenv(variable);
  if(!setting) { variable="BROOK_GRID";setting=std::getenv(variable); }
  if(setting) {
    int requested=std::stoi(setting);
    if(requested<0) throw std::invalid_argument("negative cooperative grid size");
    if(requested) cap=std::min(cap,requested);
    // One key per variable; 0 means no cap (the occupancy limit, without the barrier probe).
    ctx.stats[grid_override_key(variable)]=requested;
  } else {
    int probed=barrier_grid(ctx);
    ctx.stats["grid_probe"]=probed;   // the grid the barrier probe picked for this device
    cap=std::min(cap,probed);
  }
  return std::max(1,std::min(cap,static_cast<int>((n+127)/128)));
}
template<class Function, class... Args> void cooperative_launch(Context &ctx, Function fn, int blocks, Args... args) {
  void *arguments[]={static_cast<void*>(&args)...};
  BROOK_CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(fn),dim3(blocks),dim3(128),arguments,0,ctx.stream));
}
}
