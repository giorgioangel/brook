// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "draft_batch.hpp"
#include <algorithm>

namespace brook {
namespace {
unsigned blocks(size_t n) { return unsigned((n+255)/256); }
template<class T> Buffer<T> zero(Context &ctx,size_t count) { Buffer<T> out(ctx,std::max<size_t>(count,1));out.clear(ctx);return out; }
template<class T> Buffer<T> send(Context &ctx,const std::vector<T> &values) {
  Buffer<T> out(ctx,values.size());out.set(ctx,values.data(),values.size());ctx.synchronize();return out;
}
}
DraftLane::DraftLane(Context &ctx,int cap,int lanes,size_t store_capacity,size_t record_capacity) {
  active=zero<int>(ctx,cap);header=zero<int>(ctx,32);cursor_scratch=zero<long long>(ctx,cap);
  wcur_a=zero<int>(ctx,cap);wcur_b=zero<int>(ctx,cap);best=zero<unsigned long long>(ctx,cap);target_header=zero<int>(ctx,2);
  work_header=zero<int>(ctx,8);counts=zero<int>(ctx,8);ejected=zero<int>(ctx,cap);
  sdu=zero<float>(ctx,lanes);stb=zero<float>(ctx,lanes);sxyz=zero<long long>(ctx,2*size_t(lanes));sar=zero<int>(ctx,lanes);
  pba=Buffer<unsigned long long>(ctx,lanes);pbb=Buffer<unsigned long long>(ctx,lanes);
  store=Buffer<int>(ctx,store_capacity);rec_label=Buffer<int>(ctx,record_capacity);
  rec_length=Buffer<int>(ctx,record_capacity);rec_start=Buffer<int>(ctx,record_capacity);
}
void DraftLane::reserve_worklists(Context &ctx,size_t count) {
  if(count<=work_capacity) return;
  for(auto p:{&a,&b,&far_a,&far_b,&touched,&snap,&snapm}) { *p={};*p=Buffer<int>(ctx,count,true); }
  work_capacity=count;
}
DraftBatch::DraftBatch(Context &ctx,int k,int c,int nt,int nd,int foreground,int r,const DraftWindowPlan &p,
    const ArenaLayout &layout,const TargetOrder &order,const std::vector<double> &radii,const std::vector<long long> &roots)
    :drafts(k),cap(c),table(nt),dense(nd),reserve(r),cap_windows(p.drafting_labels),windows(p.windows),
     expanded_cap(0),groups(0),encoded_sides(p.log2[0]|(p.log2[1]<<8)|(p.log2[2]<<16)),
     arenas(int(layout.dimensions.size())|((p.log2[0]+p.log2[1]+p.log2[2])<<16)),plan(p) {
  if(k<2 || k>8 || c<=0 || c>INT32_MAX/k || nt>INT32_MAX-p.windows || layout.dimensions.size()>=65536 ||
      int64_t(p.windows)*p.voxels>r || nd>INT32_MAX-r) throw std::invalid_argument("invalid draft dimensions");
  if(c>INT32_MAX-p.windows) throw std::invalid_argument("draft active capacity exceeds int32");
  expanded_cap=c+p.windows;groups=k*c;
  std::vector<long long> offsets(layout.offsets.begin(),layout.offsets.end()),dimensions;
  for(auto shape:layout.dimensions) dimensions.insert(dimensions.end(),shape.begin(),shape.end());
  for(int i=0;i<windows;++i) {
    offsets.push_back(int64_t(dense)+int64_t(i+1)*p.voxels);
    for(int a=0;a<3;++a) dimensions.push_back(int64_t{1}<<p.log2[a]);
  }
  arena_offsets=send(ctx,offsets);arena_dimensions=send(ctx,dimensions);
  auto seg0=order.host_start,seg1=order.host_end;
  for(int i=0;i<windows;++i) { seg0.push_back(int64_t(foreground)+int64_t(i)*p.voxels);seg1.push_back(int64_t(foreground)+int64_t(i+1)*p.voxels); }
  window_start=send(ctx,seg0);window_end=send(ctx,seg1);
  auto sr=radii;auto root=roots;sr.resize(size_t(table)+windows,-1);root.resize(3*(size_t(table)+windows),0);
  soma_radius=send(ctx,sr);soma_root=send(ctx,root);
  threshold=zero<float>(ctx,size_t(table)+windows);delta=zero<float>(ctx,size_t(table)+windows);done=zero<int>(ctx,size_t(table)+windows);
  rail_best=Buffer<unsigned long long>(ctx,size_t(table)+windows);BROOK_CUDA(cudaMemsetAsync(rail_best.data(),0xff,rail_best.size()*8,ctx.stream));
  minfar=zero<unsigned long long>(ctx,size_t(table)+windows);
  target=zero<int>(ctx,groups);draft_position=zero<long long>(ctx,groups);length=zero<int>(ctx,groups);
  source_offset=zero<int>(ctx,groups);output_offset=zero<int>(ctx,groups);accepted=zero<int>(ctx,groups);accepted_count=zero<int>(ctx,cap);
  scan_offset=zero<long long>(ctx,size_t(cap)+1);first_dead=zero<unsigned long long>(ctx,size_t(cap)*(drafts+2));
  header=zero<int>(ctx,4);stats=zero<int>(ctx,8);expanded_header=zero<int>(ctx,32);invalidation_header=zero<int>(ctx,4);
  expanded_active=zero<int>(ctx,expanded_cap);expanded_target=zero<int>(ctx,expanded_cap);expanded_length=zero<int>(ctx,expanded_cap);
  expanded_offset=zero<int>(ctx,expanded_cap);zero_fill=zero<int>(ctx,expanded_cap);position_of=zero<int>(ctx,table);
  window_origin=zero<int>(ctx,3*size_t(windows));window_home=zero<int>(ctx,windows);window_label=zero<int>(ctx,windows);window_active=zero<int>(ctx,windows);
  route_bad=zero<int>(ctx,windows);window_bad=zero<int>(ctx,windows);contact_mask=zero<int>(ctx,windows);
  join_u=zero<int>(ctx,windows);join_rank=zero<int>(ctx,windows);join_home=zero<long long>(ctx,windows);
  route_rank=zero<int>(ctx,size_t(windows)*drafts);repick=zero<int>(ctx,groups);
  info=zero<int>(ctx,4*size_t(cap));number_drafted=zero<int>(ctx,cap);bits=zero<int>(ctx,128*size_t(cap_windows));
  depth=send(ctx,std::vector<int>(table,drafts));walk=Buffer<int>(ctx,size_t(foreground)+reserve,true);sources=Buffer<int>(ctx,size_t(foreground)+reserve,true);
}
void DraftBatch::tail(Context &ctx,const KernelModule &m,const DraftShared &s,DraftLane &c,int adaptive,unsigned long long condition) {
  const long long base=dense;
  m.get("k2_apply").launch(ctx,1024,256,c.header.data(),c.touched.data(),c.work_header.data(),drafts,cap,encoded_sides,base,
    window_origin.data(),window_home.data(),arena_offsets.data(),arena_dimensions.data(),accepted.data(),s.alive,s.flags);
  m.get("k2_write").launch(ctx,blocks(groups),256,c.active.data(),c.header.data(),drafts,cap,encoded_sides,base,table,length.data(),output_offset.data(),
    accepted.data(),walk.data(),window_start.data(),window_origin.data(),window_home.data(),arena_offsets.data(),arena_dimensions.data(),c.store.data(),s.penalty,
    join_home.data(),repick.data());
  m.get("kd_advance").launch(ctx,1,1,c.active.data(),drafts,cap,target.data(),length.data(),output_offset.data(),draft_position.data(),accepted_count.data(),
    c.rec_label.data(),c.rec_length.data(),c.rec_start.data(),c.ejected.data(),s.path_count,s.last_length,s.cursor,c.counts.data(),1,c.header.data(),stats.data(),
    condition,s.lane_stop,depth.data(),adaptive,accepted.data(),info.data());
}
void DraftBatch::body(Context &ctx,const KernelModule &m,const DraftShared &s,DraftLane &c,int rounds,int pool,int adaptive,int share,unsigned long long condition) {
  auto grid=[&](int value){return std::max(1,value/share);};const long long base=dense;
  // The draft lane runs beside the wide lane, on a share of its grids (BROOK_CRIT_SHARE).
  note_grid(ctx,"grid_crit_target",grid(s.target_grid));note_grid(ctx,"grid_crit_draft",grid(s.draft_grid));
  note_grid(ctx,"grid_crit_railroad",grid(s.railroad_grid));note_grid(ctx,"grid_crit_invalidation",grid(s.invalidation_grid));
  ctx.stats["grid_crit_share"]=share;
  m.get("target_lab").launch(ctx,grid(s.target_grid),128,s.alive,s.order,s.order_end,s.cursor,c.active.data(),c.header.data(),s.path_count,
    s.manual_start,s.manual_count,s.manual,s.max_paths,target.data(),c.cursor_scratch.data(),c.wcur_a.data(),c.wcur_b.data(),c.best.data(),c.target_header.data(),64,0,
    s.last_offset,s.last_length,s.wide_store,s.dbf,arena_offsets.data(),arena_dimensions.data(),arenas,s.wx,s.wy,s.wz,s.scale,s.constant);
  m.get("kd_first").launch(ctx,blocks(cap),256,c.active.data(),c.header.data(),target.data(),s.cursor,s.path_count,s.manual_count,draft_position.data());
  m.get("draft_lab").launch(ctx,grid(s.draft_grid),128,s.alive,s.order,s.order_end,c.active.data(),c.header.data(),header.data(),s.path_count,s.max_paths,
    drafts,cap,target.data(),draft_position.data(),depth.data(),c.cursor_scratch.data(),number_drafted.data(),c.wcur_a.data(),c.wcur_b.data(),bits.data(),c.target_header.data(),
    rounds,s.dbf,arena_offsets.data(),arena_dimensions.data(),arenas,s.wx,s.wy,s.wz,s.scale,s.constant);
  m.get("k2_setup").launch(ctx,blocks(expanded_cap),256,c.active.data(),c.header.data(),expanded_header.data(),drafts,cap,cap_windows,encoded_sides,base,table,
    target.data(),expanded_active.data(),expanded_target.data(),position_of.data(),window_origin.data(),window_home.data(),window_label.data(),window_active.data(),
    route_bad.data(),window_bad.data(),contact_mask.data(),soma_radius.data(),soma_root.data(),arena_offsets.data(),arena_dimensions.data(),arenas,header.data(),
    route_rank.data());
  m.get("k2_gather").launch(ctx,1024,256,header.data(),encoded_sides,base,table,window_origin.data(),window_home.data(),window_label.data(),window_active.data(),
    arena_offsets.data(),arena_dimensions.data(),s.labels,s.alive,s.dbf,s.penalty,s.graph);
  m.get("dstep_lab").launch(ctx,grid(s.railroad_grid),128,s.distance,s.penalty,s.labels,s.graph,s.has_graph,arena_offsets.data(),arena_dimensions.data(),arenas,
    expanded_active.data(),expanded_target.data(),expanded_header.data(),threshold.data(),delta.data(),rail_best.data(),minfar.data(),done.data(),
    c.a.data(),c.b.data(),c.far_a.data(),c.far_b.data(),s.flags,s.flags,c.counts.data(),c.touched.data(),INT32_MAX,s.total,s.lanes,
    c.sdu.data(),c.stb.data(),c.sxyz.data(),c.sar.data(),2.0f,5e-4f);
  m.get("backtrace_dist_lab").launch(ctx,expanded_cap,32,s.distance,s.penalty,s.labels,s.graph,s.has_graph,rail_best.data(),done.data(),expanded_active.data(),expanded_target.data(),expanded_header.data(),
    arena_offsets.data(),arena_dimensions.data(),arenas,soma_radius.data(),soma_root.data(),s.wx,s.wy,s.wz,0,expanded_offset.data(),expanded_length.data(),c.store.data(),s.total,
    walk.data(),window_start.data(),window_end.data(),zero_fill.data(),table,join_u.data(),join_rank.data());
  m.get("k2_lens").launch(ctx,blocks(groups),256,c.header.data(),drafts,cap,cap_windows,encoded_sides,base,table,target.data(),expanded_length.data(),length.data(),rail_best.data(),
    window_origin.data(),window_home.data(),arena_offsets.data(),arena_dimensions.data(),s.penalty,join_u.data(),join_rank.data(),join_home.data());
  m.get("k2_src_offsets").launch(ctx,1,1,c.header.data(),drafts,cap,target.data(),length.data(),source_offset.data(),invalidation_header.data());
  m.get("k2_src_copy").launch(ctx,blocks(groups),256,c.active.data(),c.header.data(),drafts,cap,length.data(),source_offset.data(),walk.data(),window_start.data(),table,sources.data());
  m.get("k2_route").launch(ctx,1024,256,invalidation_header.data(),sources.data(),position_of.data(),drafts,cap,cap_windows,encoded_sides,base,table,target.data(),length.data(),
    window_origin.data(),window_home.data(),arena_offsets.data(),arena_dimensions.data(),arenas,s.labels,s.graph,s.has_graph,s.penalty,s.distance,rail_best.data(),route_bad.data(),
    join_u.data(),join_rank.data(),join_home.data(),route_rank.data());
  m.get("k2_reset").launch(ctx,1024,256,s.distance,s.flags,s.flags,c.touched.data(),c.counts.data(),encoded_sides,base,table,
    window_origin.data(),window_home.data(),arena_offsets.data(),arena_dimensions.data(),s.penalty,rail_best.data(),window_bad.data());
  m.get("inval_lab").launch(ctx,grid(s.invalidation_grid),128,s.alive,s.dbf,s.labels,s.graph,s.has_graph,s.key,c.snap.data(),c.snapm.data(),arena_offsets.data(),arena_dimensions.data(),arenas,
    s.wx,s.wy,s.wz,s.scale,s.constant,std::min({s.wx,s.wy,s.wz}),sources.data(),invalidation_header.data(),c.a.data(),c.b.data(),c.far_a.data(),c.far_b.data(),
    s.flags,c.touched.data(),c.work_header.data(),s.total,s.lanes,c.sdu.data(),c.sxyz.data(),c.sar.data(),c.pba.data(),c.pbb.data());
  m.get("k2_mark").launch(ctx,1024,256,c.touched.data(),c.work_header.data(),drafts,encoded_sides,base,window_origin.data(),window_home.data(),arena_offsets.data(),arena_dimensions.data(),s.flags,window_bad.data());
  m.get("k2_contact").launch(ctx,1024,256,c.touched.data(),c.work_header.data(),drafts,encoded_sides,base,window_origin.data(),window_home.data(),arena_offsets.data(),arena_dimensions.data(),s.flags,contact_mask.data());
  m.get("kd_scan_offsets").launch(ctx,1,1,c.header.data(),drafts,cap,target.data(),length.data(),draft_position.data(),scan_offset.data(),header.data());
  m.get("kd_scan_init").launch(ctx,blocks(size_t(cap)*(drafts+2)),256,c.header.data(),drafts,first_dead.data());
  m.get("k2_scan").launch(ctx,1024,256,c.header.data(),header.data(),drafts,scan_offset.data(),draft_position.data(),s.order,s.alive,s.flags,first_dead.data());
  m.get("k2_accept").launch(ctx,blocks(cap),256,c.header.data(),drafts,cap,target.data(),length.data(),draft_position.data(),first_dead.data(),route_bad.data(),window_bad.data(),contact_mask.data(),s.flags,pool,
    accepted_count.data(),accepted.data(),info.data(),stats.data(),soma_radius.data(),table,join_rank.data(),route_rank.data(),repick.data());
  m.get("kd_offsets").launch(ctx,1,1,c.header.data(),drafts,cap,length.data(),accepted_count.data(),accepted.data(),output_offset.data());
  tail(ctx,m,s,c,adaptive,condition);
}
}
