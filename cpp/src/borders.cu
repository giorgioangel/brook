// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "skeleton.hpp"
#include "point_set.hpp"
#include "borders_batch.hpp"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

namespace brook {
namespace {
template<class Label> __global__ void extract_face(const Label *volume,Label *plane,
    long long sx,long long sy,long long fx,long long fy,int axis_a,int axis_b,int fixed_axis,long long fixed) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=fx*fy) return;
  long long p[3];p[axis_a]=i/fy;p[axis_b]=i%fy;p[fixed_axis]=fixed;
  plane[i]=volume[p[0]+sx*(p[1]+sy*p[2])];
}
template<class Label> void extract(Context &ctx,const Array &input,Array &plane,int a,int b,int fixed_axis,int64_t fixed) {
  extract_face<<<static_cast<unsigned>((plane.size()+255)/256),256,0,ctx.stream>>>(
    static_cast<const Label*>(input.data()),static_cast<Label*>(plane.data()),input.shape[0],input.shape[1],
    input.shape[a],input.shape[b],a,b,fixed_axis,fixed);
  BROOK_CUDA(cudaGetLastError());
}
template<class Label> std::vector<int64_t> download_labels(const Array &input) {
  std::vector<Label> data(input.size());input.copy_to_host(data.data(),input.bytes());return {data.begin(),data.end()};
}
std::vector<int64_t> download_labels(const Array &input) {
  switch(input.dtype) {
    case BROOK_U8:return download_labels<uint8_t>(input);
    case BROOK_U16:return download_labels<uint16_t>(input);
    case BROOK_U32:return download_labels<uint32_t>(input);
    case BROOK_U64:return download_labels<uint64_t>(input);
    default:throw std::invalid_argument("invalid border component dtype");
  }
}
float square_distance(float x,float y,float a,float b,float wx,float wy) {
  float dx=wx*(x-a),dy=wy*(y-b);return dx*dx+dy*dy;
}
std::array<float,4> keys(int64_t ix,int64_t iy,float cx,float cy,int64_t nx,int64_t ny,float wx,float wy) {
  float x=float(ix),y=float(iy),sx=float(nx),sy=float(ny);
  float center_x=float(double(wx*sx)/2.0),center_y=float(double(wy*sy)/2.0);
  float far_x=float(double(sx)-0.5),far_y=float(double(sy)-0.5);
  float corner=std::min({square_distance(x,y,-0.5f,-0.5f,wx,wy),square_distance(x,y,far_x,-0.5f,wx,wy),
    square_distance(x,y,far_x,far_y,wx,wy),square_distance(x,y,-0.5f,far_x,wx,wy)});
  float edge=float(std::min({double(wx)*(double(x)-0.5),double(wx)*(double(sx)-0.5-double(x)),
    double(wy)*(double(y)-0.5),double(wy)*(double(sy)-0.5-double(y))}));
  return {square_distance(x,y,cx,cy,wx,wy),square_distance(x,y,center_x,center_y,wx,wy),corner,edge};
}
}
std::vector<std::vector<Point>> serial_border_targets(Context &ctx,const Array &input,size_t count,std::array<float,3> aniso,
                                                    const brook_volume *host=nullptr) {
  ctx.activate();
  std::vector<PointSet> sets(count+1);
  const int axes[3][3]={{0,1,2},{0,2,1},{1,2,0}};
  for(int face=0;face<6;++face) {
    int a=axes[face/2][0],b=axes[face/2][1],c=axes[face/2][2];
    int64_t fixed=face%2?input.shape[c]-1:0,fx=input.shape[a],fy=input.shape[b];
    Array plane;
    if(host) {
      auto view=*host;view.data=static_cast<const char*>(host->data)+fixed*host->strides[c];
      view.shape[0]=fy;view.shape[1]=fx;view.shape[2]=1;
      view.strides[0]=host->strides[b];view.strides[1]=host->strides[a];view.strides[2]=0;
      plane=upload(ctx,view);
    } else {
      plane=allocate(ctx,{fy,fx,1},input.dtype);
      switch(input.dtype) {
      case BROOK_U8:extract<uint8_t>(ctx,input,plane,a,b,c,fixed);break;
      case BROOK_U16:extract<uint16_t>(ctx,input,plane,a,b,c,fixed);break;
      case BROOK_U32:extract<uint32_t>(ctx,input,plane,a,b,c,fixed);break;
      case BROOK_U64:extract<uint64_t>(ctx,input,plane,a,b,c,fixed);break;
      default:throw std::invalid_argument("invalid border component dtype");
      }
    }
    auto cc=connected_components(ctx,plane);
    auto distance=edt(ctx,cc.labels,{aniso[b],aniso[a],1},true,2);
    auto labels=download_labels(cc.labels);
    std::vector<float> d(distance.size());distance.copy_to_host(d.data(),distance.bytes());
    size_t k=cc.mapping.size();
    std::vector<float> maxdist(k,0),sumx(k,0),sumy(k,0);
    std::vector<uint32_t> sizes(k,0);
    std::vector<int64_t> first(k,INT64_MAX),pick(k,-1);
    // Match Kimimaro's sequential float32 centroid accumulation (x outer).
    for(int64_t x=0;x<fx;++x) for(int64_t y=0;y<fy;++y) {
      size_t i=y+fy*x;auto label=labels[i];if(!label) continue;
      sumx[label]+=float(x);sumy[label]+=float(y);++sizes[label];
      if(d[i]>0) { maxdist[label]=std::max(maxdist[label],d[i]);first[label]=std::min(first[label],x+fx*y); }
    }
    float wx=aniso[a],wy=aniso[b];
    std::vector<float> cx(k,0),cy(k,0);
    for(size_t label=1;label<k;++label) if(sizes[label]) {
      float px=(wx*sumx[label])/float(sizes[label]),py=(wy*sumy[label])/float(sizes[label]);
      if(px-(wx*float(fx))/2.0f<0) px=px+wx;
      if(py-(wy*float(fy))/2.0f<0) py=py+wy;
      cx[label]=float(int(px/wx));cy[label]=float(int(py/wy));
    }
    std::vector<std::array<float,4>> best(k);
    for(int64_t y=0;y<fy;++y) for(int64_t x=0;x<fx;++x) {
      size_t i=y+fy*x;auto label=labels[i];if(!label || d[i]==0 || d[i]!=maxdist[label]) continue;
      auto key=keys(x,y,cx[label],cy[label],fx,fy,wx,wy);
      if(pick[label]<0 || key<best[label]) { best[label]=key;pick[label]=x+fx*y; }
    }
    std::vector<size_t> order;
    for(size_t label=1;label<k;++label) if(pick[label]>=0) order.push_back(label);
    std::sort(order.begin(),order.end(),[&](auto x,auto y){return first[x]<first[y];});
    for(auto label:order) {
      Point p;p[a]=pick[label]%fx;p[b]=pick[label]/fx;p[c]=fixed;
      sets.at(cc.mapping[label]).add(p);
    }
  }
  std::vector<std::vector<Point>> result;result.reserve(sets.size());
  for(const auto &set:sets) result.push_back(set.values());
  return result;
}
std::vector<std::vector<Point>> border_targets_host(Context &ctx,const brook_volume &input,size_t count,std::array<float,3> aniso) {
  for(float a:aniso) if(!std::isfinite(a) || a<=0) throw std::invalid_argument("anisotropy must be finite and positive");
  if(input.memory!=BROOK_HOST) throw std::invalid_argument("host borders require host labels");
  Array descriptor;descriptor.shape={input.shape[0],input.shape[1],input.shape[2]};descriptor.dtype=input.dtype;
  if(!descriptor.size()) return std::vector<std::vector<Point>>(count+1);
  return serial_border_targets(ctx,descriptor,count,aniso,&input);
}
std::vector<std::vector<Point>> border_targets(Context &ctx,const Array &input,size_t count,std::array<float,3> aniso) {
  return border_targets_stacked(ctx,input,1,count,aniso);
}
std::vector<std::vector<Point>> border_targets_stacked(Context &ctx,const Array &input,int samples,size_t count,std::array<float,3> aniso) {
  ctx.activate();
  for(float a:aniso) if(!std::isfinite(a) || a<=0) throw std::invalid_argument("anisotropy must be finite and positive");
  if(!input.size()) return std::vector<std::vector<Point>>(count+1);
  const char *mode=std::getenv("BROOK_BORDER_BATCH");
  if(samples==1 && mode && std::string(mode)=="0") return serial_border_targets(ctx,input,count,aniso);
  auto data=batched_border_candidates(ctx,input,aniso,samples);
  const auto &shape=data.shape;
  const int axes[3][3]={{0,1,2},{0,2,1},{1,2,0}};
  std::vector<size_t> picks(data.groups.size(),0);
  // tie-break among a face component's farthest pixels: nearest the component's centroid, then the
  // face centre, corners and edges (the serial plan's rule)
  for(size_t i=0;i<data.groups.size();++i) {
    const auto &group=data.groups[i];
    if(group.scans.size()<2) continue;
    const int pair=group.face/2;int a=axes[pair][0],b=axes[pair][1];
    const int64_t fx=shape[a],fy=shape[b],A=data.A[pair],B=data.B[pair];
    float wx=aniso[a],wy=aniso[b];
    float px=(wx*group.sumx)/float(group.size),py=(wy*group.sumy)/float(group.size);
    if(px-(wx*float(fx))/2.0f<0) px+=wx;
    if(py-(wy*float(fy))/2.0f<0) py+=wy;
    float cx=float(int(px/wx)),cy=float(int(py/wy));
    std::array<float,4> best{};
    for(size_t k=0;k<group.scans.size();++k) {
      auto scan=group.scans[k];
      auto key=keys(scan%B,(scan/B)%A,cx,cy,fx,fy,wx,wy);
      if(k==0 || key<best) {best=key;picks[i]=k;}
    }
  }
  std::vector<size_t> order(data.groups.size());std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](auto a,auto b){return data.groups[a].order<data.groups[b].order;});
  // A component lies in one sample. Its set holds the points in the sample's own frame, whose hashes
  // give the order a separate call gives; the values then move into the stack's frame.
  std::vector<PointSet> sets(count+1);std::vector<int64_t> offsets(count+1,0);
  for(auto g:order) {
    const auto &group=data.groups[g];const int face=group.face,pair=face/2;
    const int64_t A=data.A[pair],B=data.B[pair];
    auto scan=group.scans[picks[g]];
    int a=axes[pair][0],b=axes[pair][1],c=axes[pair][2];
    Point p;p[a]=scan%B;p[b]=(scan/B)%A;p[c]=face%2?shape[c]-1:0;
    sets.at(group.owner).add(p);offsets.at(group.owner)=int64_t(group.sample)*shape[2];
  }
  std::vector<std::vector<Point>> result;result.reserve(sets.size());
  for(size_t id=0;id<sets.size();++id) {
    auto values=sets[id].values();
    for(auto &p:values) p[2]+=offsets[id];
    result.push_back(std::move(values));
  }
  return result;
}
}
