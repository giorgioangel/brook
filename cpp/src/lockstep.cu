// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "lockstep.hpp"
#include "lockstep_order.hpp"
#include "graph.hpp"
#include "assembly.hpp"
#include "draft_plan.hpp"
#include "draft_state.hpp"
#include "draft_batch.hpp"
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cctype>

namespace brook {
namespace {
unsigned blocks(size_t n) { return static_cast<unsigned>((n+255)/256); }
template<class T> Buffer<T> upload(Context &ctx,const std::vector<T> &values) {
  Buffer<T> out(ctx,values.size());out.set(ctx,values.data(),values.size());ctx.synchronize();return out;
}
template<class T> Buffer<T> zeros(Context &ctx,size_t count) { Buffer<T> out(ctx,count);out.clear(ctx);return out; }
template<class T> void grow(Context &ctx,Buffer<T> &buffer,size_t size,size_t used) {
  if(size>INT32_MAX) throw std::runtime_error("lockstep store exceeds int32 capacity");
  Buffer<T> next(ctx,size);
  if(used) BROOK_CUDA(cudaMemcpyAsync(next.data(),buffer.data(),used*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));
  ctx.synchronize();buffer=std::move(next);
}
__global__ void fill_infinity(float *values,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) values[i]=__int_as_float(0x7f800000);
}
__global__ void set_roots(float *field,const int *root,const int *labels,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) field[root[labels[i]]]=0;
}
__global__ void radii(const float *dbf,const int *path,float *out,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) out[i]=dbf[path[i]];
}
std::string normalized(const char *value) {
  std::string out=value?value:"";
  auto first=out.find_first_not_of(" \t\r\n"),last=out.find_last_not_of(" \t\r\n");
  out=first==std::string::npos?"":out.substr(first,last-first+1);
  for(char &c:out) c=char(std::tolower(static_cast<unsigned char>(c)));
  return out;
}
bool setting(const char *name,bool fallback) {
  const char *v=std::getenv(name);if(!v) return fallback;
  auto s=normalized(v);return s!="0" && s!="off" && s!="false" && s!="no";
}
int integer_setting(const char *name,int fallback) {
  const char *v=std::getenv(name);return v?std::stoi(v):fallback;
}
__global__ void shifted_records(const int *input,int *output,int count,int shift) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count) output[i]=input[i]+shift;
}
template<class Label> __global__ void count_zero_distances(const Label *labels,const float *dbf,int n,int *count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && labels[i] && dbf[i]==0.0f) atomicAdd(count,1);
}
template<class Label> __global__ void zero_distances_to_infinity(const Label *labels,float *dbf,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && labels[i] && dbf[i]==0.0f) dbf[i]=__int_as_float(0x7f800000);
}
// A foreground voxel at distance zero (background painted into a label by the avocado correction,
// whose field is not refreshed) is +inf to the tracer, as prepare_component's normalization makes
// it: infinite penalty, invalidation ball and radius. The raw field stays with its other holders
// (the component fallbacks summarize it), so shared storage is copied before the change.
void normalize_zero_distances(Context &ctx,const Array &labels,Array &dbf) {
  const int n=int(dbf.size());if(!n) return;
  Buffer<int> count(ctx,1);count.clear(ctx);
  auto run=[&](auto value) {
    using Label=decltype(value);
    count_zero_distances<Label><<<blocks(n),256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),static_cast<const float*>(dbf.data()),n,count.data());
    BROOK_CUDA(cudaGetLastError());
    const int zeros=count.get(ctx,1)[0];if(!zeros) return;
    ctx.stats["lockstep_zero_distances"]=zeros;
    if(dbf.storage.use_count()>1) {
      auto copy=allocate(ctx,dbf.shape,dbf.dtype);
      BROOK_CUDA(cudaMemcpyAsync(copy.data(),dbf.data(),dbf.bytes(),cudaMemcpyDeviceToDevice,ctx.stream));dbf=std::move(copy);
    }
    zero_distances_to_infinity<Label><<<blocks(n),256,0,ctx.stream>>>(static_cast<const Label*>(labels.data()),static_cast<float*>(dbf.data()),n);
    BROOK_CUDA(cudaGetLastError());ctx.synchronize();
  };
  switch(labels.dtype) {
    case BROOK_U8:run(uint8_t{});break;
    case BROOK_U16:run(uint16_t{});break;
    case BROOK_U32:run(uint32_t{});break;
    case BROOK_U64:run(uint64_t{});break;
    default:throw std::invalid_argument("invalid lockstep label dtype");
  }
}
}

LockstepResult trace_lockstep(Context &ctx,Components cc,Array dbf,BatchedPreparation prep,const ArenaLayout &layout,
    const std::vector<Box> &boxes,const std::vector<std::vector<Point>> &borders,
    const std::vector<std::vector<Point>> &before,const std::vector<std::vector<Point>> &after,
    const SkeletonizeOptions &options,bool keep_device) {
  ctx.activate();
  const auto setup_begin=std::chrono::steady_clock::now();
  int n=int(cc.labels.size());const int dense=n;const size_t table=cc.mapping.size();
  const int na=int(layout.dimensions.size());
  const bool fix=options.fix_branching;const auto aniso=options.anisotropy;
  const float wx=aniso[0],wy=aniso[1],wz=aniso[2],scale=float(options.teasar.scale),constant=float(options.teasar.constant);
  const float inv_delta=std::min({wx,wy,wz});
  LockstepResult result;result.components.resize(table);result.handled.resize(table,0);
  if((fix && !prep.pdrf.storage) || (!fix && !prep.parents.storage)) return result;
  // The voxel graph (when there is one) gates every neighbour step of the floods, backtraces and
  // invalidations below exactly as it gates the single-component tracer. It is grown with the
  // draft reserve, so its pointer is taken at every launch.
  const int has_vg=prep.graph.storage?1:0;
  if(has_vg && (prep.graph.dtype!=BROOK_U32 || prep.graph.size()!=size_t(n))) throw std::invalid_argument("lockstep voxel graph must be uint32 over the arenas");
  auto vg=[&]()->void* { return has_vg?prep.graph.data():nullptr; };
  std::vector<uint8_t> eligible=prep.active;
  std::vector<int> labels,manual_start(table,0),manual_count(table,0),manual_flat,maximum_paths(table,0),roots(table,-1);
  std::vector<int64_t> counts_host(table,0);
  std::vector<double> soma_radius(table,-1);
  std::vector<long long> soma_root(table*3,0);
  std::vector<int> soma_seeds;
  auto global_flat=[&](Point point,size_t label) {
    int arena=layout.label_arena[label];
    for(int a=0;a<3;++a) point[a]-=layout.origins[arena][a];
    return layout.offsets[arena]+flatten(point,layout.dimensions[arena]);
  };
  int64_t foreground=0;
  for(size_t id=1;id<table;++id) {
    if(!after[id].empty()) eligible[id]=0;
    counts_host[id]=boxes[id].count;
    if(!eligible[id]) continue;
    labels.push_back(int(id));foreground+=boxes[id].count;roots[id]=int(prep.root[id]);
    std::vector<Point> manual=borders[id];
    if(!manual.empty()) manual.pop_back();
    if(prep.soma[id] && !borders[id].empty()) manual.insert(manual.begin(),borders[id].back());
    manual.insert(manual.end(),before[id].begin(),before[id].end());
    manual_start[id]=int(manual_flat.size());
    if(manual.empty() && !prep.soma[id]) manual_flat.push_back(int(prep.target[id]));
    else for(auto it=manual.rbegin();it!=manual.rend();++it) manual_flat.push_back(int(global_flat(*it,id)));
    manual_count[id]=int(manual_flat.size())-manual_start[id];
    int64_t budget=options.teasar.max_paths.value_or(boxes[id].count);
    maximum_paths[id]=manual_count[id]>=budget?0:int(std::min(int64_t(INT32_MAX),budget));
    if(prep.soma[id]) {
      soma_seeds.push_back(roots[id]);
      soma_radius[id]=double(prep.dbf_max[id])*options.teasar.soma_scale+options.teasar.soma_constant;
      int arena=layout.label_arena[id];
      auto point=unravel(roots[id]-layout.offsets[arena],layout.dimensions[arena]);for(int k=0;k<3;++k) soma_root[3*id+k]=point[k];
    }
  }
  if(labels.empty()) return result;
  size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
  double needed=double(n)*(13+(fix?4:0))+double(foreground)*32;
  if(needed>0.85*double(free)) { ctx.stats["lockstep_declined_memory"]=1;return result; }   // per-component tracer instead
  normalize_zero_distances(ctx,cc.labels,dbf);
  ctx.stats["lockstep_labels"]=int64_t(labels.size());ctx.stats["arenas"]=na-1;
  const int cap=int(labels.size());
  auto module=lockstep_module(cc.labels.dtype);
  auto order=build_target_order(ctx,cc.labels,prep.daf,eligible,counts_host,fix);prep.daf={};
  auto cursor=upload(ctx,order.host_start);
  auto alive=Buffer<uint8_t>(ctx,n);initialize_alive(ctx,cc.labels,eligible,alive);
  auto aoff=upload(ctx,std::vector<long long>(layout.offsets.begin(),layout.offsets.end()));
  std::vector<long long> dimensions;for(auto shape:layout.dimensions) dimensions.insert(dimensions.end(),shape.begin(),shape.end());
  auto adim=upload(ctx,dimensions);
  auto sr=upload(ctx,soma_radius);auto sroot=upload(ctx,soma_root);auto root=upload(ctx,roots);
  auto mb_start=upload(ctx,manual_start),mb_count=upload(ctx,manual_count);
  manual_flat.push_back(0);auto mb_flat=upload(ctx,manual_flat);
  Buffer<unsigned long long> key(ctx,n);BROOK_CUDA(cudaMemsetAsync(key.data(),0xff,n*sizeof(unsigned long long),ctx.stream));
  auto flags=zeros<int>(ctx,n);
  Buffer<int> a(ctx,foreground,true),b(ctx,foreground,true),touched(ctx,foreground,true),snap(ctx,foreground,true),snapm(ctx,foreground,true);
  Buffer<int> far_a(ctx,foreground,true),far_b(ctx,foreground,true);
  auto hdr=zeros<int>(ctx,8),counts=zeros<int>(ctx,8);
  int lanes=std::min(n,1<<18);
  Buffer<float> sdu(ctx,lanes),stb(ctx,lanes);
  Buffer<long long> sxyz(ctx,2*lanes);Buffer<int> sar(ctx,lanes);
  Buffer<unsigned long long> pba(ctx,lanes),pbb(ctx,lanes);
  Buffer<int> route_a,route_b,route_far_a,route_far_b,route_touched,route_sar;
  Buffer<float> route_sdu,route_stb;Buffer<long long> route_sxyz;
  int *ra=a.data(),*rb=b.data(),*rfa=far_a.data(),*rfb=far_b.data(),*rtouched=touched.data(),*rsar=sar.data();
  float *rsdu=sdu.data(),*rstb=stb.data();long long *rsxyz=sxyz.data();
  Buffer<float> distance,threshold,delta;
  Buffer<unsigned long long> rail_best,minfar;
  Buffer<int> done;
  if(fix) {
    distance=Buffer<float>(ctx,n);fill_infinity<<<blocks(n),256,0,ctx.stream>>>(distance.data(),n);
    threshold=zeros<float>(ctx,table);delta=zeros<float>(ctx,table);
    rail_best=Buffer<unsigned long long>(ctx,table);BROOK_CUDA(cudaMemsetAsync(rail_best.data(),0xff,table*sizeof(unsigned long long),ctx.stream));
    minfar=zeros<unsigned long long>(ctx,table);done=zeros<int>(ctx,table);
    auto selected=upload(ctx,labels);
    set_roots<<<blocks(cap),256,0,ctx.stream>>>(static_cast<float*>(prep.pdrf.data()),root.data(),selected.data(),cap);
    BROOK_CUDA(cudaGetLastError());ctx.synchronize();
  }
  int target_grid=module.get("target_lab").grid(ctx,n,"BROOK_LOCKSTEP_GRID");
  int invalidation_grid=module.get("inval_lab").grid(ctx,n,"BROOK_LOCKSTEP_GRID");
  int railroad_grid=fix?module.get("dstep_lab").grid(ctx,n,"BROOK_LOCKSTEP_GRID"):1;
  // share 1: the sequential body (and the pipelined join); share 2: the two pipelined branches.
  auto grid_key=[](int share,const char *family) { return std::string(share==1?"grid_lockstep_":"grid_pipe_")+family; };
  auto invalidate_paths=[&](const int *seeds,const int *pending,float s,float c,int share=1) {
    note_grid(ctx,grid_key(share,"invalidation").c_str(),std::max(1,invalidation_grid/share));
    module.get("inval_lab").launch(ctx,std::max(1,invalidation_grid/share),128,alive.data(),dbf.data(),cc.labels.data(),vg(),has_vg,key.data(),snap.data(),snapm.data(),
      aoff.data(),adim.data(),na,wx,wy,wz,s,c,inv_delta,seeds,pending,a.data(),b.data(),far_a.data(),far_b.data(),flags.data(),touched.data(),
      hdr.data(),n,lanes,sdu.data(),sxyz.data(),sar.data(),pba.data(),pbb.data());
  };
  if(!soma_seeds.empty()) {
    auto seeds=upload(ctx,soma_seeds),pending=upload(ctx,std::vector<int>{0,int(soma_seeds.size()),0,0});
    invalidate_paths(seeds.data(),pending.data(),float(options.teasar.soma_scale),float(options.teasar.soma_constant));
    auto header=hdr.get(ctx,8);if(header[4]) throw std::runtime_error("soma invalidation did not converge");
    auto removed=removed_per_label(ctx,cc.labels,touched,header[3],table);
    for(auto id:labels) if(prep.soma[id]) {
      int64_t budget=options.teasar.max_paths.value_or(boxes[id].count-removed[id]);
      maximum_paths[id]=manual_count[id]>=budget?0:int(std::min(int64_t(INT32_MAX),budget));
    }
  }
  auto maxp=upload(ctx,maximum_paths),active=upload(ctx,labels);
  auto target=zeros<int>(ctx,cap),length=zeros<int>(ctx,cap),offset=zeros<int>(ctx,cap),status=zeros<int>(ctx,cap);
  auto true_target=zeros<int>(ctx,cap);
  auto wcur=zeros<long long>(ctx,cap);auto un_a=zeros<int>(ctx,cap),un_b=zeros<int>(ctx,cap);
  auto best=zeros<unsigned long long>(ctx,cap);auto thdr=zeros<int>(ctx,2);
  auto ejected=zeros<int>(ctx,cap),pcount=zeros<int>(ctx,table),last_offset=zeros<int>(ctx,table),last_length=zeros<int>(ctx,table);
  auto zero_fill=zeros<int>(ctx,cap),lane_stop=zeros<int>(ctx,1);
  int initial_store=std::max(1,integer_setting("BROOK_STORE_START",1<<22));
  int initial_records=std::max(1,integer_setting("BROOK_RECORDS_START",1<<16));
  Buffer<int> store(ctx,initial_store);
  size_t record_capacity=std::max(size_t(4*cap),size_t(initial_records));
  Buffer<int> rec_label(ctx,record_capacity),rec_length(ctx,record_capacity),rec_start(ctx,record_capacity);
  std::vector<int> h(32,0);h[0]=cap;h[9]=int(store.size());h[10]=int(record_capacity);h[11]=INT32_MAX;
  const char *force_setting=std::getenv("BROOK_BODY");std::string forced=force_setting?force_setting:"";
  const char *pipeline_setting=std::getenv("BROOK_PIPELINE");
  bool use_graph=graphs_enabled(),can_pipeline=use_graph && fix && forced!="seq" && (!pipeline_setting || std::string(pipeline_setting)!="0");
  bool pipe_open=false,force_pipe=forced=="pipe";
  if(can_pipeline) h[20]=std::max(1,railroad_grid/2)*128;
  auto ih=upload(ctx,h);
  const char *preflight_setting=std::getenv("BROOK_DRAFT_PREFLIGHT");
  bool draft_preflight=fix && preflight_setting && std::string(preflight_setting)=="1";
  Buffer<long long> draft_voxels;
  int64_t next_draft_check=16;
  if(draft_preflight) draft_voxels=upload(ctx,std::vector<long long>(counts_host.begin(),counts_host.end()));
  auto backtrace=[&](int pass) {
    if(fix) module.get("backtrace_dist_lab").launch(ctx,cap,32,distance.data(),prep.pdrf.data(),cc.labels.data(),vg(),has_vg,rail_best.data(),done.data(),
      active.data(),target.data(),ih.data(),aoff.data(),adim.data(),na,sr.data(),sroot.data(),wx,wy,wz,pass,offset.data(),length.data(),store.data(),n,
      ra,order.start.data(),order.end.data(),zero_fill.data(),INT32_MAX,static_cast<int*>(nullptr),static_cast<int*>(nullptr));
    else module.get("backtrace_parent_lab").launch(ctx,blocks(cap),256,prep.parents.data(),root.data(),active.data(),target.data(),ih.data(),
      aoff.data(),adim.data(),na,sr.data(),sroot.data(),wx,wy,wz,pass,offset.data(),length.data(),store.data(),n,ra,order.start.data(),order.end.data(),zero_fill.data(),1);
  };
  auto tail=[&](unsigned long long condition) {
    backtrace(1);
    if(fix) {
      module.get("reset_touched").launch(ctx,1024,256,distance.data(),flags.data(),flags.data(),rtouched,counts.data(),ih.data());
      module.get("lock_rails").launch(ctx,1024,256,prep.pdrf.data(),store.data(),ih.data());
    }
    module.get("lock_advance").launch(ctx,1,1,active.data(),status.data(),length.data(),offset.data(),rec_label.data(),rec_length.data(),rec_start.data(),
      ejected.data(),pcount.data(),last_offset.data(),last_length.data(),counts.data(),int(fix),ih.data(),condition,lane_stop.data());
  };
  auto find_target=[&](int *out,int spec,int share=1) {
    note_grid(ctx,grid_key(share,"target").c_str(),std::max(1,target_grid/share));
    module.get("target_lab").launch(ctx,std::max(1,target_grid/share),128,alive.data(),order.order.data(),order.end.data(),cursor.data(),active.data(),ih.data(),pcount.data(),
      mb_start.data(),mb_count.data(),mb_flat.data(),maxp.data(),out,wcur.data(),un_a.data(),un_b.data(),best.data(),thdr.data(),64,spec,
      last_offset.data(),last_length.data(),store.data(),dbf.data(),aoff.data(),adim.data(),na,wx,wy,wz,scale,constant);
  };
  auto branch_b=[&](int spec,int share=1) {
    find_target(target.data(),spec,share);
    if(fix) note_grid(ctx,grid_key(share,"railroad").c_str(),std::max(1,railroad_grid/share));
    if(fix) module.get("dstep_lab").launch(ctx,std::max(1,railroad_grid/share),128,distance.data(),prep.pdrf.data(),cc.labels.data(),vg(),has_vg,aoff.data(),adim.data(),na,
      active.data(),target.data(),ih.data(),threshold.data(),delta.data(),rail_best.data(),minfar.data(),done.data(),ra,rb,rfa,rfb,
      flags.data(),flags.data(),counts.data(),rtouched,INT32_MAX,n,lanes,rsdu,rstb,rsxyz,rsar,2.0f,5e-4f);
    backtrace(0);
  };
  auto join=[&](bool spec,unsigned long long condition) {
    if(spec) find_target(true_target.data(),0);
    module.get("lock_offsets").launch(ctx,1,1,spec?true_target.data():target.data(),target.data(),length.data(),offset.data(),status.data(),ih.data());
    tail(condition);
  };
  auto body=[&](unsigned long long condition) {
    invalidate_paths(store.data(),ih.data()+14,scale,constant);branch_b(0);join(false,condition);
  };
  std::unique_ptr<WhileGraph> loops[3];
  auto close_loops=[&] { for(auto &loop:loops) loop.reset(); };
  auto open_pipeline=[&] {
    close_loops();h[19]=h[20]=0;
    size_t available=0,total=0;BROOK_CUDA(cudaMemGetInfo(&available,&total));
    if(can_pipeline && double(foreground)*20<=0.5*double(available)) {
      route_a=Buffer<int>(ctx,foreground,true);route_b=Buffer<int>(ctx,foreground,true);
      route_far_a=Buffer<int>(ctx,foreground,true);route_far_b=Buffer<int>(ctx,foreground,true);route_touched=Buffer<int>(ctx,foreground,true);
      route_sdu=Buffer<float>(ctx,lanes);route_stb=Buffer<float>(ctx,lanes);route_sxyz=Buffer<long long>(ctx,2*lanes);route_sar=Buffer<int>(ctx,lanes);
      ra=route_a.data();rb=route_b.data();rfa=route_far_a.data();rfb=route_far_b.data();rtouched=route_touched.data();
      rsdu=route_sdu.data();rstb=route_stb.data();rsxyz=route_sxyz.data();rsar=route_sar.data();pipe_open=true;
      ctx.stats["pipelined"]=1;   // the pipelined body can run (its buffers exist); chunks_pipe counts where it did
    }
    can_pipeline=false;ih.set(ctx,h.data(),h.size());ctx.synchronize();
  };
  if(force_pipe && can_pipeline) open_pipeline();
  const int draft_count=std::clamp(integer_setting("BROOK_DRAFTS",8),1,8);
  const int draft_rounds=std::max(0,integer_setting("BROOK_DRAFT_ROUNDS",3));
  const int draft_pool=int(setting("BROOK_KPOOL",true)),draft_adaptive=int(setting("BROOK_HORIZON",true));
  const int critical_share=std::max(1,integer_setting("BROOK_CRIT_SHARE",4));
  const bool force_drafts=forced=="k";
  const char *draft_env=std::getenv("BROOK_KDRAFT");
  bool k_pending=fix && !draft_preflight && draft_count>=2 && (use_graph || force_drafts) && (forced.empty() || force_drafts) &&
    (force_drafts || setting("BROOK_KDRAFT",true));
  bool k_eager=force_drafts || normalized(draft_env)=="eager";
  int64_t k_check_at=k_eager?0:16;
  if(k_pending && !draft_voxels.size()) draft_voxels=upload(ctx,std::vector<long long>(counts_host.begin(),counts_host.end()));
  std::unique_ptr<DraftBatch> kb;std::unique_ptr<DraftLane> critical;
  int reserve=0,draft_grid=1,nc=0,nc_max=1,ranked_at=0,planned_at=0;
  int64_t k_chunks=0,off_until=0,backoff=1;
  bool split=false;double lane_ratio=draft_count,wide_velocity=-1;
  const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_begin).count();
  const auto loop_begin=std::chrono::steady_clock::now();
  auto draft_shared=[&] {
    DraftShared s{};s.labels=cc.labels.data();s.graph=vg();s.has_graph=has_vg;s.alive=alive.data();s.dbf=static_cast<float*>(dbf.data());
    s.penalty=static_cast<float*>(prep.pdrf.data());s.distance=distance.data();s.key=key.data();s.flags=flags.data();
    s.path_count=pcount.data();s.last_offset=last_offset.data();s.last_length=last_length.data();
    s.manual_start=mb_start.data();s.manual_count=mb_count.data();s.manual=mb_flat.data();s.max_paths=maxp.data();
    s.wide_store=store.data();s.lane_stop=lane_stop.data();s.cursor=cursor.data();s.order_end=order.end.data();s.order=order.order.data();
    s.total=n;s.lanes=lanes;s.target_grid=target_grid;s.draft_grid=draft_grid;s.railroad_grid=railroad_grid;s.invalidation_grid=invalidation_grid;
    s.wx=wx;s.wy=wy;s.wz=wz;s.scale=scale;s.constant=constant;return s;
  };
  auto remaining=[&] { return estimate_draft_paths(ctx,active.data(),h[0],cursor.data(),order.end.data(),order.order.data(),alive.data(),pcount.data(),draft_voxels.data()); };
  auto window_plan=[&] {
    const char *window=std::getenv("BROOK_KWINDOW");int forced_side=window && *window?std::max(4,std::stoi(window)):0;
    return plan_draft_window(ctx,rec_start.data(),rec_length.data(),store.data(),static_cast<const float*>(dbf.data()),h[4],layout.dimensions.front(),aniso,
      options.teasar.scale,options.teasar.constant,forced_side);
  };
  auto allocate_batch=[&](const DraftWindowPlan &plan) {
    close_loops();
    Buffer<int> old_stats,old_depth,old_walk,old_sources;
    if(kb) { old_stats=std::move(kb->stats);old_depth=std::move(kb->depth);old_walk=std::move(kb->walk);old_sources=std::move(kb->sources);kb.reset(); }
    kb=std::make_unique<DraftBatch>(ctx,draft_count,cap,int(table),dense,int(foreground),reserve,plan,layout,order,soma_radius,soma_root);
    if(old_stats.size()) { kb->stats=std::move(old_stats);kb->depth=std::move(old_depth);kb->walk=std::move(old_walk);kb->sources=std::move(old_sources); }
    if(!critical) critical=std::make_unique<DraftLane>(ctx,cap,lanes,initial_store,initial_records);
    ctx.stats["drafts"]=draft_count;ctx.stats["draft_windows"]=plan.windows;
    for(int a=0;a<3;++a) ctx.stats[std::string("draft_window_")+"xyz"[a]]=int64_t{1}<<plan.log2[a];
    ctx.synchronize();
  };
  auto open_drafts=[&] {
    if(cap>INT32_MAX/draft_count || table>INT32_MAX || na>=65536) return false;
    close_loops();auto proposed=window_plan();auto estimates=remaining();
    double longest=*std::max_element(estimates.remaining.begin(),estimates.remaining.end());
    int count=int(std::count_if(estimates.remaining.begin(),estimates.remaining.end(),[&](double value){return value*draft_count>=longest;}));
    int bits=proposed[0]+proposed[1]+proposed[2];
    int64_t want=bits>=31?INT32_MAX:int64_t(std::min(double(INT32_MAX),2.0*std::max(count,1)*(draft_count-1)*double(int64_t{1}<<bits)));
    size_t available=0,total_bytes=0;BROOK_CUDA(cudaMemGetInfo(&available,&total_bytes));
    reserve=draft_reserve_budget(available,dense,int(foreground),dtype_size(cc.labels.dtype),cc.labels.storage.use_count()>1 || dbf.storage.use_count()>1,want);
    const char *reserve_limit=std::getenv("BROOK_DRAFT_RESERVE");
    if(reserve_limit) reserve=std::min(reserve,std::max(0,std::stoi(reserve_limit)));
    auto plan=fit_draft_window(proposed,reserve,table,dtype_size(cc.labels.dtype),cap,draft_count);
    if(!plan) { reserve=0;return false; }
    bool shared_route=ra==a.data();
    append_draft_reserve(ctx,{cc.labels,dbf,prep.pdrf,alive,key,flags,distance,{&a,&b,&touched,&snap,&snapm,&far_a,&far_b},has_vg?&prep.graph:nullptr},dense,int(foreground),reserve);
    n=dense+reserve;
    if(shared_route) { ra=a.data();rb=b.data();rfa=far_a.data();rfb=far_b.data();rtouched=touched.data(); }
    target_grid=module.get("target_lab").grid(ctx,n,"BROOK_LOCKSTEP_GRID");
    invalidation_grid=module.get("inval_lab").grid(ctx,n,"BROOK_LOCKSTEP_GRID");
    railroad_grid=module.get("dstep_lab").grid(ctx,n,"BROOK_LOCKSTEP_GRID");
    draft_grid=module.get("draft_lab").grid(ctx,n,"BROOK_LOCKSTEP_GRID");
    BROOK_CUDA(cudaMemGetInfo(&available,&total_bytes));
    ctx.stats["draft_reserve"]=reserve;
    if(8.0*(foreground+reserve)+64.0*(cap+reserve/64)+96.0*draft_count*cap>0.5*double(available)) { ctx.stats["draft_allocation_declined"]=1;return false; }
    allocate_batch(*plan);ctx.stats["draft_reserve"]=reserve;return true;
  };
  auto join_lanes=[&] {
    if(!split) return;
    if(nc) BROOK_CUDA(cudaMemcpyAsync(active.data()+h[0],critical->active.data(),size_t(nc)*4,cudaMemcpyDeviceToDevice,ctx.stream));
    h[0]+=nc;ih.set(ctx,h.data(),h.size());ctx.synchronize();split=false;nc=0;
  };
  auto enter_lanes=[&] {
    if(split && nc>0 && h[1]-ranked_at<256) {
      std::vector<int> ch(32,0);ch[0]=nc;ch[9]=int(critical->store.size());ch[10]=int(critical->rec_label.size());
      critical->header.set(ctx,ch.data(),ch.size());ctx.synchronize();return true;
    }
    join_lanes();
    if(h[15]) { invalidate_paths(store.data(),ih.data()+14,scale,constant);h[15]=0;ih.set(ctx,h.data(),h.size());ctx.synchronize(); }
    auto estimate=remaining();std::vector<size_t> rank(estimate.labels.size());for(size_t i=0;i<rank.size();++i) rank[i]=i;
    std::stable_sort(rank.begin(),rank.end(),[&](size_t a,size_t b){return estimate.remaining[a]>estimate.remaining[b];});
    if(rank.empty()) return false;
    int drafting=0;for(size_t i:rank) if(estimate.remaining[i]*lane_ratio>=estimate.remaining[rank.front()]) ++drafting;
    if(h[1]-planned_at>=256) {
      auto plan=fit_draft_window(window_plan(),reserve,table,dtype_size(cc.labels.dtype),cap,draft_count);planned_at=h[1];
      if(plan && plan->log2!=kb->plan.log2) { allocate_batch(*plan);++ctx.stats["draft_replans"]; }
    }
    drafting=std::min({drafting,kb->cap_windows,nc_max});
    size_t need=size_t(kb->windows)*kb->plan.voxels;
    for(int i=0;i<drafting;++i) need+=size_t(counts_host[estimate.labels[rank[i]]]);
    if(need>critical->work_capacity) {
      size_t available=0,total_bytes=0;BROOK_CUDA(cudaMemGetInfo(&available,&total_bytes));
      if(double(need)*28>0.5*double(available)) { ++ctx.stats["draft_lane_declined"];return false; }
      close_loops();critical->reserve_worklists(ctx,need);
    }
    std::vector<int> wide,crit;for(size_t i=0;i<rank.size();++i) (int(i)<drafting?crit:wide).push_back(estimate.labels[rank[i]]);
    active.set(ctx,wide.data(),wide.size());critical->active.set(ctx,crit.data(),crit.size());h[0]=int(wide.size());
    ih.set(ctx,h.data(),h.size());std::vector<int> ch(32,0);ch[0]=drafting;ch[9]=int(critical->store.size());ch[10]=int(critical->rec_label.size());
    critical->header.set(ctx,ch.data(),ch.size());ctx.synchronize();split=true;nc=drafting;ranked_at=h[1];return true;
  };
  auto unpark_wide=[&] {
    if(!h[5]) return;close_loops();
    if(int64_t(h[2])+h[3]>int64_t(store.size())) { grow(ctx,store,std::max(store.size()*2,size_t(h[2])+h[3]),h[2]);h[9]=int(store.size()); }
    if(int64_t(h[4])+h[0]>int64_t(rec_label.size())) {
      size_t size=std::max(rec_label.size()*2,size_t(h[4])+h[0]);grow(ctx,rec_label,size,h[4]);grow(ctx,rec_length,size,h[4]);grow(ctx,rec_start,size,h[4]);h[10]=int(size);
    }
    h[5]=0;ih.set(ctx,h.data(),h.size());tail(0);h=ih.get(ctx,32);
  };
  auto leave_lanes=[&] {
    auto ch=critical->header.get(ctx,32);
    if(ch[5]) {
      close_loops();
      if(int64_t(ch[2])+ch[3]>int64_t(critical->store.size())) grow(ctx,critical->store,std::max(critical->store.size()*2,size_t(ch[2])+ch[3]),ch[2]);
      if(int64_t(ch[4])+draft_count*ch[0]>int64_t(critical->rec_label.size())) {
        size_t size=std::max(critical->rec_label.size()*2,size_t(ch[4])+size_t(draft_count)*ch[0]);
        grow(ctx,critical->rec_label,size,ch[4]);grow(ctx,critical->rec_length,size,ch[4]);grow(ctx,critical->rec_start,size,ch[4]);
      }
      ch[5]=0;ch[9]=int(critical->store.size());ch[10]=int(critical->rec_label.size());critical->header.set(ctx,ch.data(),ch.size());
      kb->tail(ctx,module,draft_shared(),*critical,draft_adaptive);ch=critical->header.get(ctx,32);
    }
    if(int64_t(h[2])+ch[2]>int64_t(store.size()) || int64_t(h[4])+ch[4]>int64_t(rec_label.size())) {
      close_loops();
      if(int64_t(h[2])+ch[2]>int64_t(store.size())) grow(ctx,store,std::max(store.size()*2,size_t(h[2])+ch[2]),h[2]);
      if(int64_t(h[4])+ch[4]>int64_t(rec_label.size())) {
        size_t size=std::max(rec_label.size()*2,size_t(h[4])+ch[4]);grow(ctx,rec_label,size,h[4]);grow(ctx,rec_length,size,h[4]);grow(ctx,rec_start,size,h[4]);
      }
    }
    if(ch[2]) BROOK_CUDA(cudaMemcpyAsync(store.data()+h[2],critical->store.data(),size_t(ch[2])*4,cudaMemcpyDeviceToDevice,ctx.stream));
    if(ch[4]) {
      BROOK_CUDA(cudaMemcpyAsync(rec_label.data()+h[4],critical->rec_label.data(),size_t(ch[4])*4,cudaMemcpyDeviceToDevice,ctx.stream));
      BROOK_CUDA(cudaMemcpyAsync(rec_length.data()+h[4],critical->rec_length.data(),size_t(ch[4])*4,cudaMemcpyDeviceToDevice,ctx.stream));
      shifted_records<<<blocks(ch[4]),256,0,ctx.stream>>>(critical->rec_start.data(),rec_start.data()+h[4],ch[4],h[2]);
    }
    if(ch[8]) BROOK_CUDA(cudaMemcpyAsync(ejected.data()+h[8],critical->ejected.data(),size_t(ch[8])*4,cudaMemcpyDeviceToDevice,ctx.stream));
    nc=ch[0];h[2]+=ch[2];h[4]+=ch[4];h[9]=int(store.size());h[10]=int(rec_label.size());
    for(int f:{6,7,8,12,13}) h[f]+=ch[f];
    // Without wide-lane labels the draft lane's batches are the lockstep iterations.
    if(!h[0]) { h[1]+=ch[1];ctx.stats["iterations_k"]+=ch[1]; }
    ctx.stats["draft_batches"]+=ch[1];ih.set(ctx,h.data(),h.size());
    BROOK_CUDA(cudaGetLastError());ctx.synchronize();
    if(critical->work_header.get(ctx,8)[4]) throw std::runtime_error("critical invalidation did not converge");
    return std::pair<int,int>{ch[1],ch[7]};
  };
  const bool balanced_schedule=setting("BROOK_BALANCED_SCHEDULE",true);
  int last_trial_iterations[2]={0,0};
  double rates[2]={-1,-1},iteration_time[2]={0,0};int chunk=16,challenger=-1;int64_t epoch=4,since_explore=0;
  while(h[0]>0 || (split && nc>0)) {
    if(k_pending && h[0]>0 && h[1]>=k_check_at) {
      k_check_at=std::max<int64_t>(16,2LL*h[1]);
      auto estimate=remaining();double longest=*std::max_element(estimate.remaining.begin(),estimate.remaining.end());
      double pace=std::chrono::duration<double>(std::chrono::steady_clock::now()-loop_begin).count()/std::max(h[1],1);
      if(k_eager || (h[4]>0 && longest*pace>=setup_seconds)) { k_pending=false;open_drafts(); }
    }
    if(h[19]) open_pipeline();
    if(draft_preflight && h[0]>0 && h[1]>=next_draft_check) {
      auto estimates=estimate_draft_paths(ctx,active.data(),h[0],cursor.data(),order.end.data(),order.order.data(),alive.data(),pcount.data(),draft_voxels.data());
      auto window=plan_draft_window(ctx,rec_start.data(),rec_length.data(),store.data(),static_cast<const float*>(dbf.data()),h[4],
        layout.dimensions.front(),aniso,options.teasar.scale,options.teasar.constant);
      double longest=*std::max_element(estimates.remaining.begin(),estimates.remaining.end());
      int drafting=int(std::count_if(estimates.remaining.begin(),estimates.remaining.end(),[&](double value){return value*8>=longest;}));
      int sum=window[0]+window[1]+window[2];
      int64_t wanted=sum>=31?INT32_MAX:int64_t(std::min(double(INT32_MAX),2.0*std::max(drafting,1)*7.0*double(int64_t{1}<<sum)));
      size_t available=0,total_bytes=0;BROOK_CUDA(cudaMemGetInfo(&available,&total_bytes));
      bool retained=cc.labels.storage.use_count()>1 || dbf.storage.use_count()>1;
      int reserve=draft_reserve_budget(available,n,int(foreground),dtype_size(cc.labels.dtype),retained,wanted);
      auto fitted=fit_draft_window(window,reserve,table,dtype_size(cc.labels.dtype),cap,8);
      if(!ctx.stats["draft_planning_checks"]) ctx.stats["draft_first_check"]=h[1];
      ++ctx.stats["draft_planning_checks"];
      ctx.stats["draft_estimated_paths"]=int64_t(std::llround(longest));
      ctx.stats["draft_proposed_reserve"]=reserve;
      ctx.stats["draft_retained_inputs"]=retained;
      ctx.stats["draft_plan_labels"]=fitted?fitted->drafting_labels:0;
      if(fitted) for(int axis=0;axis<3;++axis) ctx.stats[std::string("draft_window_")+"xyz"[axis]]=int64_t{1}<<fitted->log2[axis];
      next_draft_check=std::max<int64_t>(16,2LL*h[1]);
    }
    int mode=pipe_open && force_pipe?1:0;bool measuring=pipe_open && !force_pipe;
    if(measuring) {
      if(rates[0]<0) mode=0;
      else if(rates[1]<0) mode=1;
      else {
        int winner=rates[1]>rates[0]?1:0;
        if(since_explore>=epoch) { mode=1-winner;challenger=mode;since_explore=0; }
        else { mode=winner;++since_explore; }
      }
    }
    bool run_critical=false;
    if(kb) {
      ++k_chunks;
      if(k_chunks>=off_until) {
        run_critical=enter_lanes();
        if(!run_critical) off_until=INT64_MAX;
      } else join_lanes();
    }
    const bool fresh_trial=measuring && rates[mode]<0 && last_trial_iterations[1-mode]>0;
    h[11]=INT32_MAX;
    if(measuring) {
      h[11]=chunk;
      if(iteration_time[mode]>0) {
        double quick=iteration_time[0]>0?iteration_time[0]:iteration_time[1];
        if(iteration_time[1]>0) quick=std::min(quick,iteration_time[1]);
        h[11]=std::max(2,int(chunk*quick/iteration_time[mode]));
      }
      // A draft checkpoint may truncate the baseline sample. Do not spend a
      // full nominal chunk on the untried body when its baseline got less.
      if(balanced_schedule && fresh_trial)
        h[11]=std::min(h[11],std::max(2,last_trial_iterations[1-mode]));
    }
    if(kb && !measuring) h[11]=256;
    if(k_pending) h[11]=std::min(h[11],int(std::min<int64_t>(INT32_MAX,std::max<int64_t>(k_check_at-h[1],1))));
    if(draft_preflight) h[11]=std::min(h[11],int(std::min<int64_t>(INT32_MAX,std::max<int64_t>(next_draft_check-h[1],1))));
    if(fresh_trial)ctx.stats["initial_trial_iterations"]=h[11];
    ih.set(ctx,h.data(),h.size());
    lane_stop.clear(ctx);
    auto begin=std::chrono::steady_clock::now();int paths_before=h[7],iterations_before=h[1];
    int wide_labels=h[0],critical_labels=nc,critical_paths=0;
    if(run_critical) {
      auto ch=critical->header.get(ctx,32);ch[11]=wide_labels?(1<<30):256;critical->header.set(ctx,ch.data(),ch.size());
      BROOK_CUDA(cudaMemcpyAsync(kb->header.data()+3,&critical_labels,4,cudaMemcpyHostToDevice,ctx.stream));ctx.synchronize();
    }
    auto capture_wide=[&] {
      auto &loop=loops[mode];
      if(!loop) {
        loop=std::make_unique<WhileGraph>(ctx);
        if(mode==0) loop->capture([&]{body(loop->condition());});
        else {
          auto dependencies=loop->capture_segment([&]{invalidate_paths(store.data(),ih.data()+14,scale,constant,2);});
          auto second=loop->capture_segment([&]{branch_b(1,2);});
          dependencies.insert(dependencies.end(),second.begin(),second.end());
          loop->capture_segment([&]{join(true,loop->condition());},dependencies);loop->finish();
        }
      }
    };
    if(run_critical) {
      double dw=0,dc=0;
      if(use_graph) {
        if(wide_labels) capture_wide();
        if(critical_labels && !loops[2]) {
          loops[2]=std::make_unique<WhileGraph>(ctx);
          loops[2]->capture([&]{kb->body(ctx,module,draft_shared(),*critical,draft_rounds,draft_pool,draft_adaptive,critical_share,loops[2]->condition());});
        }
        if(wide_labels) loops[mode]->launch();
        if(critical_labels) loops[2]->launch();
        if(wide_labels) dw=loops[mode]->wait();
        if(critical_labels) dc=loops[2]->wait();
      } else {
        if(wide_labels) body(0);
        if(critical_labels) kb->body(ctx,module,draft_shared(),*critical,draft_rounds,draft_pool,draft_adaptive,1);
      }
      h=ih.get(ctx,32);unpark_wide();
      int iw=h[1]-iterations_before,pw=h[7]-paths_before;
      if(wide_labels) ctx.stats[mode?"iterations_pipe":"iterations_seq"]+=iw;
      auto done=leave_lanes();critical_paths=done.second;
      ++ctx.stats["chunks_k"];
      if(use_graph && done.first>0 && dc>0) {
        if(wide_labels && iw>0 && dw>0) {
          double wide_iteration=dw/iw,batch_time=dc/done.first;
          wide_velocity=pw/(wide_labels*dw);
          if(pw>0 && critical_paths>0) lane_ratio=std::max(std::sqrt(lane_ratio*(critical_paths/(critical_labels*dc))/wide_velocity),1.0);
          nc_max=batch_time<=wide_iteration?std::min(2*nc_max,kb->cap_windows):std::max(nc_max/2,1);
        }
        if(wide_velocity>=0 && !force_drafts && critical_paths/(critical_labels*dc)<wide_velocity) {
          off_until=std::min<int64_t>(INT64_MAX/2,k_chunks+backoff);backoff=std::min<int64_t>(INT64_MAX/2,backoff*2);
          ++ctx.stats["k_lane_off"];   // the draft lane was slower than the wide lane: off for a while
        } else backoff=1;
      }
    } else {
      if(use_graph) { capture_wide();loops[mode]->run(); } else body(0);
      h=ih.get(ctx,32);
      ctx.stats[mode?"iterations_pipe":"iterations_seq"]+=h[1]-iterations_before;
    }
    if(!run_critical || wide_labels) ++ctx.stats[mode?"chunks_pipe":"chunks_seq"];   // the wide lane ran this body
    if(measuring && (!run_critical || wide_labels)) {
      double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
      last_trial_iterations[mode]=std::max(h[1]-iterations_before,0);
      rates[mode]=(h[7]-paths_before-critical_paths)/elapsed;iteration_time[mode]=elapsed/std::max(h[1]-iterations_before,1);
      if(challenger==mode) {
        if(rates[mode]>rates[1-mode]) { epoch=4;chunk=16; }
        else { epoch=std::min(epoch*2,int64_t{1}<<60);chunk=std::min(chunk*2,256); }
        challenger=-1;
      }
    }
    const int parked_from=h[1];unpark_wide();   // the tail of a parked iteration still belongs to this body
    if(h[1]!=parked_from) ctx.stats[mode?"iterations_pipe":"iterations_seq"]+=h[1]-parked_from;
  }
  close_loops();
  join_lanes();
  if(kb) {
    auto counters=kb->stats.get(ctx,8);
    const char *names[]={"drafted","accepted","draft_void","draft_route","draft_contact","draft_wrong_target","draft_skipped","draft_repicked"};
    for(int i=0;i<8;++i) ctx.stats[names[i]]=counters[i];
    kb.reset();critical.reset();
  }
  if(hdr.get(ctx,8)[4]) throw std::runtime_error("lockstep invalidation did not converge");
  result.iterations=h[1];result.paths=h[7];result.ejected=h[8];
  ctx.stats["iterations"]=h[1];ctx.stats["paths"]=h[7];ctx.stats["ejected"]=h[8];ctx.stats["pipe_rejected"]=h[18];
  for(auto id:labels) result.handled[id]=1;
  for(auto id:ejected.get(ctx,h[8])) result.handled[id]=0;
  const char *assembly=std::getenv("BROOK_ASSEMBLY");
  auto assembly_begin=std::chrono::steady_clock::now();
  if(!assembly || std::string(assembly)!="cpu") {
    size_t available=0,total=0;BROOK_CUDA(cudaMemGetInfo(&available,&total));
    size_t original_size=volume_size(layout.dimensions.front());
    if(double(h[2])*80<0.6*double(available) && table<=INT32_MAX && original_size<=UINT64_MAX/table) {
      auto packed=assemble_paths(ctx,cc,dbf,layout,result.handled,store.data(),h[2],rec_start.data(),rec_length.data(),h[4],aniso);
      if(keep_device) result.device=std::move(packed);
      else download_components(packed,cc,result.handled,aniso,result.components);
      ctx.stats["assembly_device"]=1;
      ctx.stats["assembly_us"]=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-assembly_begin).count();
      return result;
    }
  }
  auto lab=rec_label.get(ctx,h[4]),len=rec_length.get(ctx,h[4]),start=rec_start.get(ctx,h[4]);
  auto flat=store.get(ctx,h[2]);Buffer<float> path_radii(ctx,h[2]);
  if(h[2]) radii<<<blocks(h[2]),256,0,ctx.stream>>>(static_cast<const float*>(dbf.data()),store.data(),path_radii.data(),h[2]);
  BROOK_CUDA(cudaGetLastError());auto rad=path_radii.get(ctx,h[2]);
  for(auto id:labels) if(result.handled[id]) { result.components[id]=Skeleton{};result.components[id]->label=cc.mapping[id];result.components[id]->anisotropy=aniso; }
  for(size_t i=0;i<lab.size();++i) {
    int id=lab[i];if(!result.handled[id] || len[i]<2) continue;
    auto &sk=*result.components[id];size_t base=sk.vertices.size();
    const int64_t z0=size_t(id)<layout.label_z.size()?layout.label_z[id]:0;   // the sample's frame in a stack
    for(int j=0;j<len[i];++j) {
      int64_t voxel=flat[start[i]+j];
      int arena=int(std::upper_bound(layout.offsets.begin(),layout.offsets.end(),voxel)-layout.offsets.begin())-1;
      auto point=unravel(voxel-layout.offsets[arena],layout.dimensions[arena]);
      for(int axis=0;axis<3;++axis) point[axis]+=layout.origins[arena][axis];
      point[2]-=z0;
      sk.vertices.push_back({float(point[0]),float(point[1]),float(point[2])});sk.radii.push_back(rad[start[i]+j]);
      if(j) sk.edges.push_back({uint32_t(base+j-1),uint32_t(base+j)});
    }
  }
  for(auto &value:result.components) if(value) {
    consolidate(*value);for(auto &v:value->vertices) for(int k=0;k<3;++k) v[k]*=aniso[k];
  }
  ctx.stats["assembly_us"]=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-assembly_begin).count();
  return result;
}
}
