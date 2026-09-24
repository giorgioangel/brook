// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "draft_plan.hpp"
#include <algorithm>
#include <cmath>
#include <math_constants.h>

namespace brook {
namespace {
__global__ void remaining_paths(const int *active,int count,const long long *cursor,const long long *ends,
    const int *order,const uint8_t *alive,const int *paths,const long long *voxels,double *remaining) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  int label=active[i];long long first=cursor[label],end=ends[label],left=end-first;
  if(left<=0) { remaining[i]=0;return; }
  int valid=0;
  for(int sample=0;sample<64;++sample) {
    long long at=first+static_cast<long long>((double(sample)+0.5)*(double(left)/64.0));
    at=min(at,end-1);valid+=alive[order[at]]!=0;
  }
  double alive_left=double(left)*(double(valid)/64.0);
  double per_path=fmax(double(voxels[label])-alive_left,1.0)/(double(paths[label])+1.0);
  remaining[i]=alive_left/per_path;
}
__global__ void path_reach(const int *starts,const int *lengths,const int *store,const float *dbf,
    int first,int count,long long sx,long long sy,long long volume,double ax,double ay,double az,
    double scale,double constant,double *output) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count) return;
  int start=starts[first+i],length=lengths[first+i];
  if(length<=0) { for(int a=0;a<3;++a) output[3*i+a]=-1;return; }
  long long target=store[static_cast<long long>(start)+length-1];
  double best[3]={0,0,0};
  if(target<volume) {
    for(int a=0;a<3;++a) best[a]=-CUDART_INF;
    long long t[3]={target%sx,(target/sx)%sy,target/(sx*sy)};
    const double pitch[3]={ax,ay,az};
    for(int j=0;j<length;++j) {
      long long v=store[static_cast<long long>(start)+j];
      if(v>=volume) { for(int a=0;a<3;++a) best[a]=fmax(best[a],0.0);continue; }
      long long c[3]={v%sx,(v/sx)%sy,v/(sx*sy)};
      double radius=scale*double(dbf[v])+constant;
      for(int a=0;a<3;++a) best[a]=fmax(best[a],double(llabs(c[a]-t[a]))+ceil(radius/pitch[a])+1.0);
    }
  }
  for(int a=0;a<3;++a) output[3*i+a]=best[a];
}
int side_log(double reach) {
  if(!std::isfinite(reach)) return 30;
  return std::max(2,std::min(30,int(std::ceil(std::log2(std::max(reach,4.0))))));
}
}
DraftEstimate estimate_draft_paths(Context &ctx,const int *active,int count,const long long *cursor,
    const long long *ends,const int *order,const uint8_t *alive,const int *paths,const long long *voxels) {
  if(count<0) throw std::invalid_argument("negative draft label count");
  DraftEstimate result;result.labels.resize(count);result.remaining.resize(count);if(!count) return result;
  Buffer<double> output(ctx,count);
  remaining_paths<<<(count+255LL)/256,256,0,ctx.stream>>>(active,count,cursor,ends,order,alive,paths,voxels,output.data());
  BROOK_CUDA(cudaGetLastError());
  BROOK_CUDA(cudaMemcpyAsync(result.labels.data(),active,size_t(count)*sizeof(int),cudaMemcpyDeviceToHost,ctx.stream));
  BROOK_CUDA(cudaMemcpyAsync(result.remaining.data(),output.data(),size_t(count)*sizeof(double),cudaMemcpyDeviceToHost,ctx.stream));
  ctx.synchronize();return result;
}
std::array<int,3> plan_draft_window(Context &ctx,const int *starts,const int *lengths,const int *store,
    const float *dbf,int records,std::array<int64_t,3> original_shape,std::array<float,3> anisotropy,
    double scale,double constant,int forced_side) {
  if(forced_side>0) { int log=side_log(forced_side);return {log,log,log}; }
  if(records<=0) return {4,4,4};
  int count=std::min(records,1024);Buffer<double> output(ctx,3*count);
  path_reach<<<(count+255)/256,256,0,ctx.stream>>>(starts,lengths,store,dbf,records-count,count,
    original_shape[0],original_shape[1],volume_size(original_shape),double(anisotropy[0]),double(anisotropy[1]),double(anisotropy[2]),scale,constant,output.data());
  BROOK_CUDA(cudaGetLastError());std::vector<double> values(3*count);std::vector<int> valid_lengths(count);
  BROOK_CUDA(cudaMemcpyAsync(values.data(),output.data(),values.size()*sizeof(double),cudaMemcpyDeviceToHost,ctx.stream));
  BROOK_CUDA(cudaMemcpyAsync(valid_lengths.data(),lengths+records-count,count*sizeof(int),cudaMemcpyDeviceToHost,ctx.stream));ctx.synchronize();
  int maximum=2;
  for(int axis=0;axis<3;++axis) {
    std::vector<double> need;need.reserve(count);
    for(int i=0;i<count;++i) if(valid_lengths[i]>0) need.push_back(values[3*i+axis]);
    if(need.empty()) continue;
    std::sort(need.begin(),need.end());size_t mid=need.size()/2;
    double median=need.size()%2?need[mid]:(need[mid-1]+need[mid])*0.5;
    maximum=std::max(maximum,side_log(2*std::max(median,2.0)));
  }
  return {maximum,maximum,maximum};
}
std::optional<DraftWindowPlan> fit_draft_window(std::array<int,3> requested,int reserve,
    size_t table,size_t label_bytes,int labels,int drafts) {
  if(reserve<=0 || labels<=0 || drafts<2 || drafts>8 || table>INT32_MAX) return {};
  if(label_bytes!=1 && label_bytes!=2 && label_bytes!=4 && label_bytes!=8) throw std::invalid_argument("invalid draft label width");
  for(int &log:requested) log=std::max(2,std::min(log,30));
  uint64_t room=label_bytes<8?(uint64_t{1}<<(8*label_bytes)):uint64_t(INT32_MAX);
  room=room>table?room-table:0;
  room=std::min(room,uint64_t(INT32_MAX)-table);
  for(;;) {
    int sum=requested[0]+requested[1]+requested[2];
    int voxels=sum<=30?int(uint32_t{1}<<sum):0;
    int fit=voxels?int(std::min(uint64_t(reserve/voxels),room)/unsigned(drafts-1)):0;
    fit=std::min(fit,labels);
    if(fit>0) return DraftWindowPlan{requested,fit,fit*(drafts-1),voxels};
    auto largest=std::max_element(requested.begin(),requested.end());
    if(*largest<=2) return {};--*largest;
  }
}
int draft_reserve_budget(size_t free_bytes,int dense,int foreground,size_t label_bytes,
    bool retained_inputs,int64_t wanted) {
  if(dense<0 || foreground<0 || wanted<=0) return 0;
  double room=0.85*double(free_bytes)-(21.0*dense+32.0*foreground);
  if(retained_inputs) room-=double(dense)*(label_bytes+8);
  if(room<=0) return 0;
  double voxels=std::floor((room/8.0)/double(label_bytes+61));
  return int(std::max(0.0,std::min({double(foreground),voxels,double(INT32_MAX-dense),double(wanted)})));
}
}
