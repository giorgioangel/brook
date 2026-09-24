// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
// CUDA implementation of the face-stack border-target plan (all six faces at once).
// Root labels, packed candidate keys and first-live-pixel ordering match that plan.
#include "borders_batch.hpp"
#include "ccl_kernels.cuh"
#include "edt_kernels.cuh"
#include "graph.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace brook {
namespace {
// Distinct template arguments avoid duplicate CUDA kernel registrations across TUs.
struct BorderTag {};
struct BorderLabels {
  const uint32_t *data;
  __device__ uint32_t operator[](long long i) const { return data[i]; }
};
template<class Label> __global__ void stack_faces(const Label *input,Label *stack,
    long long sx,long long sy,long long sz,long long A,long long B,int pair,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n) return;
  long long y=i%A,x=(i/A)%B,slab=i/(A*B);
  const Label *sample=input+(slab/BORDER_PAIR_SLABS)*(sx*sy*sz);const int local=int(slab%BORDER_PAIR_SLABS);
  Label value=0;
  if(local!=1) {
    const int face=2*pair+(local==2);
    if(pair==0 && x<sx && y<sy) value=sample[x+sx*(y+sy*(face&1?sz-1:0))];
    if(pair==1 && x<sx && y<sz) value=sample[x+sx*((face&1?sy-1:0)+sy*y)];
    if(pair==2 && x<sy && y<sz) value=sample[(face&1?sx-1:0)+sx*(x+sy*y)];
  }
  stack[i]=value;
}
template<class Label> __global__ void face_labels(const Label *stack,const int *parent,uint32_t *labels,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) labels[i]=stack[i]?uint32_t(parent[i]+1):0;
}
__global__ void finish_reset(const uint32_t *labels,float *dist,int *maximum,unsigned long long *first,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) dist[i]=labels[i]?sqrtf(dist[i]):0.0f;
  if(i<=n) {maximum[i]=0;first[i]=INT64_MAX;}
}
__device__ unsigned int scan_key(long long v,long long A,long long B) {
  return static_cast<unsigned int>((v/(A*B)*A+v%A)*B+(v/A)%B);
}
__global__ void reduce_faces(const uint32_t *labels,const float *dist,int *maximum,unsigned long long *first,
    long long n,long long A,long long B) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n || !labels[i] || dist[i]==0) return;
  auto label=labels[i];int bits=__float_as_int(dist[i]);
  if(bits>maximum[label]) atomicMax(maximum+label,bits);
  unsigned long long key=scan_key(i,A,B);
  if(key<first[label]) atomicMin(first+label,key);
}
__global__ void collect_faces(const uint32_t *labels,const float *dist,const int *maximum,
    unsigned long long *out,int *count,long long n,long long A,long long B) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n && labels[i] && dist[i]!=0 && __float_as_int(dist[i])==maximum[labels[i]])
    out[atomicAdd(count,1)]=(static_cast<unsigned long long>(labels[i])<<32)|scan_key(i,A,B);
}
// Tie-break centroids. The serial plan sums a face component's x and y in float32, x outer and y
// inner. Coordinates are integers, so that sum is exact (equal to the integer sum) as long as it
// stays below 2^24; the exact integer sums and the component's bounding box are accumulated in
// parallel per face, and only a component whose sum passes 2^24 is re-summed sequentially on the
// host within its bounding box, in the serial order.
__global__ void face_sums(const uint32_t *face,long long A,long long B,long long slab_base,
    unsigned long long *sumx,unsigned long long *sumy,unsigned int *size,int *box) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=A*B) return;
  const uint32_t label=face[i];if(!label) return;
  const long long r=static_cast<long long>(label)-1-slab_base;      // root index, local to the slab
  const long long y=i%A,x=i/A;
  atomicAdd(sumx+r,static_cast<unsigned long long>(x));atomicAdd(sumy+r,static_cast<unsigned long long>(y));atomicAdd(size+r,1u);
  atomicMin(box+4*r,int(x));atomicMax(box+4*r+1,int(x));atomicMin(box+4*r+2,int(y));atomicMax(box+4*r+3,int(y));
}
__global__ void gather_sums(const long long *locals,int count,const unsigned long long *sumx,const unsigned long long *sumy,
    const unsigned int *size,const int *box,unsigned long long *ox,unsigned long long *oy,unsigned int *on,int *ob) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count) return;
  const long long r=locals[k];ox[k]=sumx[r];oy[k]=sumy[r];on[k]=size[r];
  for(int j=0;j<4;++j) ob[4*k+j]=box[4*r+j];
}
__global__ void reset_sums(long long n,unsigned long long *sumx,unsigned long long *sumy,unsigned int *size,int *box) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;
  sumx[i]=0;sumy[i]=0;size[i]=0;box[4*i]=INT32_MAX;box[4*i+1]=-1;box[4*i+2]=INT32_MAX;box[4*i+3]=-1;
}
template<class Label> __global__ void gather_groups(const Label *stack,const unsigned long long *first,
    const uint32_t *labels,unsigned long long *owners,unsigned long long *order,long long n) {
  long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) {owners[i]=stack[labels[i]-1];order[i]=first[labels[i]];}
}
bool enabled(const char *name) { const char *v=std::getenv(name);return !v || std::string(v)!="0"; }
}
// One axis pair of the face stack: [face, separator, face] slabs per sample, A x B per slab.
struct PairPlan {
  int pair;long long A,B,slabs,n;
  Array stack;
  Buffer<int> parent,maximum,V,count;
  Buffer<uint32_t> labels;
  Buffer<float> distance,H,table,wslab;
  Buffer<double> Z;
  Buffer<unsigned long long> first,candidates;
  std::unique_ptr<FixedGraph> graph;   // destroyed before the buffers it references
  PairPlan(Context &ctx,int p,long long a,long long b,int samples,brook_dtype dtype):pair(p),A(a),B(b),
      slabs(int64_t(BORDER_PAIR_SLABS)*samples),n(A*B*slabs),
      stack(allocate(ctx,{A,B,slabs},dtype)),parent(ctx,n),maximum(ctx,n+1),V(ctx,A*slabs*(B+3)),count(ctx,1),
      labels(ctx,n),distance(ctx,n),H(ctx,V.size()),table(ctx,(A+2)*slabs),wslab(ctx,slabs),Z(ctx,V.size()),
      first(ctx,n+1),candidates(ctx,n) {}
  size_t bytes() const {
    return stack.bytes()+n*(sizeof(int)*2+sizeof(uint32_t)+sizeof(float)+sizeof(unsigned long long)*2)
      +V.size()*16+table.size()*4+wslab.size()*4+16;
  }
  void update_spacing(Context &ctx,std::array<float,3> aniso) {
    const int axes[3][2]={{0,1},{0,2},{1,2}};
    std::vector<float> acc(table.size(),0),w(slabs,1);
    for(long long slab=0;slab<slabs;++slab) {
      float w0=1;
      if(slab%BORDER_PAIR_SLABS!=1) {w0=aniso[axes[pair][1]];w[slab]=aniso[axes[pair][0]];}
      for(long long i=1;i<A+2;++i) acc[slab*(A+2)+i]=acc[slab*(A+2)+i-1]+w0;
    }
    table.set(ctx,acc.data(),acc.size());wslab.set(ctx,w.data(),w.size());
  }
  template<class Label> void launch(Context &ctx) {
    unsigned grid=unsigned((n+255)/256);
    auto *s=static_cast<const Label*>(stack.data());
    cuda::ccl_init<Label,int,BorderTag><<<grid,256,0,ctx.stream>>>(parent.data(),n);
    cuda::ccl_union<Label,int,BorderTag><<<grid,256,0,ctx.stream>>>(s,parent.data(),A,B,slabs);
    cuda::ccl_flatten<Label,int,BorderTag><<<grid,256,0,ctx.stream>>>(parent.data(),n);
    face_labels<<<grid,256,0,ctx.stream>>>(s,parent.data(),labels.data(),n);
    BorderLabels lab{labels.data()};
    cuda::edt_pass1_x<<<unsigned((B*slabs+255)/256),256,0,ctx.stream>>>(distance.data(),lab,A,B,slabs,table.data(),A+2,1);
    cuda::edt_pass_fh<<<unsigned((A*slabs+255)/256),256,0,ctx.stream>>>(distance.data(),lab,V.data(),H.data(),Z.data(),
      A,B,slabs,1,1.0f,wslab.data(),1,B+3,0LL,A*slabs);
    finish_reset<<<unsigned((n+1+255)/256),256,0,ctx.stream>>>(labels.data(),distance.data(),maximum.data(),first.data(),n);
    count.clear(ctx);
    reduce_faces<<<grid,256,0,ctx.stream>>>(labels.data(),distance.data(),maximum.data(),first.data(),n,A,B);
    collect_faces<<<grid,256,0,ctx.stream>>>(labels.data(),distance.data(),maximum.data(),candidates.data(),count.data(),n,A,B);
    BROOK_CUDA(cudaGetLastError());
  }
  template<class Label> void run(Context &ctx,const Array &input,const std::array<int64_t,3> &shape,int samples,BorderCandidates &out) {
    stack_faces<<<unsigned((n+255)/256),256,0,ctx.stream>>>(static_cast<const Label*>(input.data()),
      static_cast<Label*>(stack.data()),shape[0],shape[1],shape[2],A,B,pair,n);
    BROOK_CUDA(cudaGetLastError());
    if(graphs_enabled() && enabled("BROOK_BORDER_GRAPH")) {
      if(!graph) {
        auto captured=std::make_unique<FixedGraph>(ctx);
        captured->capture([&]{launch<Label>(ctx);});graph=std::move(captured);
        ++ctx.stats["border_graph_captures"];
      }
      graph->run();
    } else launch<Label>(ctx);
    int nc=count.get(ctx,1)[0];auto packed=candidates.get(ctx,nc);
    std::sort(packed.begin(),packed.end());
    std::vector<uint32_t> group_labels;std::vector<size_t> starts;
    for(size_t i=0;i<packed.size();) {
      size_t j=i+1;auto label=uint32_t(packed[i]>>32);
      while(j<packed.size() && uint32_t(packed[j]>>32)==label) ++j;
      group_labels.push_back(label);starts.push_back(i);i=j;
    }
    if(group_labels.empty()) return;
    Buffer<uint32_t> gl(ctx,group_labels.size());gl.set(ctx,group_labels.data(),group_labels.size());
    Buffer<unsigned long long> owners(ctx,group_labels.size()),order(ctx,group_labels.size());
    gather_groups<<<unsigned((group_labels.size()+255)/256),256,0,ctx.stream>>>(static_cast<const Label*>(stack.data()),
      first.data(),gl.data(),owners.data(),order.data(),static_cast<long long>(group_labels.size()));
    BROOK_CUDA(cudaGetLastError());
    auto oh=owners.get(ctx,group_labels.size()),fh=order.get(ctx,group_labels.size());
    // the order key reproduces the single-volume plan's first-live-pixel ordering over faces
    const long long Ao=std::max(shape[1],shape[2]),Bo=std::max(shape[0],shape[1]);
    std::vector<uint32_t> tied_label;std::vector<long long> tied_slab;std::vector<size_t> tied_index;
    const size_t first_group=out.groups.size();
    for(size_t g=0;g<group_labels.size();++g) {
      size_t begin=starts[g],end=g+1<starts.size()?starts[g+1]:packed.size();
      const long long slab=uint32_t(packed[begin])/(A*B);
      BorderGroup group;group.pair=pair;group.sample=int(slab/BORDER_PAIR_SLABS);group.face=2*pair+(slab%BORDER_PAIR_SLABS==2);
      group.owner=oh[g];
      const unsigned long long f=fh[g];const long long fy=(f/B)%A,fx=f%B;
      group.order=(static_cast<unsigned long long>(group.sample*6+group.face)*Ao+fy)*Bo+fx;
      for(size_t i=begin;i<end;++i) group.scans.push_back(uint32_t(packed[i])%uint32_t(A*B));
      if(end-begin>1) { tied_label.push_back(group_labels[g]);tied_slab.push_back(slab);tied_index.push_back(first_group+g); }
      out.groups.push_back(std::move(group));
    }
    if(!tied_label.empty()) {
      const long long face_pixels=A*B;
      Buffer<unsigned long long> sumx(ctx,face_pixels),sumy(ctx,face_pixels);Buffer<unsigned int> size(ctx,face_pixels);Buffer<int> box(ctx,4*face_pixels);
      std::vector<long long> slabs_tied(tied_slab.begin(),tied_slab.end());
      std::sort(slabs_tied.begin(),slabs_tied.end());slabs_tied.erase(std::unique(slabs_tied.begin(),slabs_tied.end()),slabs_tied.end());
      std::vector<uint32_t> face_host;
      for(long long slab:slabs_tied) {
        reset_sums<<<unsigned((face_pixels+255)/256),256,0,ctx.stream>>>(face_pixels,sumx.data(),sumy.data(),size.data(),box.data());
        face_sums<<<unsigned((face_pixels+255)/256),256,0,ctx.stream>>>(labels.data()+slab*face_pixels,A,B,slab*face_pixels,sumx.data(),sumy.data(),size.data(),box.data());
        BROOK_CUDA(cudaGetLastError());
        std::vector<size_t> members;
        for(size_t i=0;i<tied_index.size();++i) if(tied_slab[i]==slab) members.push_back(i);
        std::vector<long long> locals;for(auto i:members) locals.push_back(static_cast<long long>(tied_label[i])-1-slab*face_pixels);
        Buffer<long long> li(ctx,locals.size());li.set(ctx,locals.data(),locals.size());
        Buffer<unsigned long long> gx(ctx,locals.size()),gy(ctx,locals.size());Buffer<unsigned int> gn(ctx,locals.size());Buffer<int> gb(ctx,4*locals.size());
        gather_sums<<<unsigned((locals.size()+255)/256),256,0,ctx.stream>>>(li.data(),int(locals.size()),sumx.data(),sumy.data(),size.data(),box.data(),
          gx.data(),gy.data(),gn.data(),gb.data());
        BROOK_CUDA(cudaGetLastError());
        auto hx=gx.get(ctx,locals.size()),hy=gy.get(ctx,locals.size());auto hn=gn.get(ctx,locals.size());auto hb=gb.get(ctx,4*locals.size());
        for(size_t k=0;k<members.size();++k) {
          auto &g=out.groups[tied_index[members[k]]];g.size=hn[k];
          if(hx[k]<(1ull<<24) && hy[k]<(1ull<<24)) { g.sumx=float(hx[k]);g.sumy=float(hy[k]);continue; }
          if(face_host.empty()) {                 // rare: re-sum in the serial order within the bounding box
            face_host.resize(face_pixels);
            BROOK_CUDA(cudaMemcpyAsync(face_host.data(),labels.data()+slab*face_pixels,face_pixels*sizeof(uint32_t),cudaMemcpyDeviceToHost,ctx.stream));ctx.synchronize();
          }
          float sx=0,sy=0;const uint32_t label=tied_label[members[k]];
          for(long long x=hb[4*k];x<=hb[4*k+1];++x) for(long long y=hb[4*k+2];y<=hb[4*k+3];++y) if(face_host[y+A*x]==label) { sx+=float(x);sy+=float(y); }
          g.sumx=sx;g.sumy=sy;
        }
        face_host.clear();
      }
    }
    ctx.synchronize();
  }
};
struct BorderPlan {
  std::array<int64_t,3> shape;          // one sample's shape
  std::array<float,3> spacing;
  brook_dtype dtype;
  int samples;
  std::array<std::unique_ptr<PairPlan>,3> pairs;
  static std::array<int64_t,3> sample_shape(const Array &input,int samples) {
    return {input.shape[0],input.shape[1],input.shape[2]/samples};
  }
  BorderPlan(Context &ctx,const Array &input,std::array<float,3> aniso,int count):shape(sample_shape(input,count)),spacing(aniso),dtype(input.dtype),samples(count) {
    const int axes[3][2]={{0,1},{0,2},{1,2}};
    for(int p=0;p<3;++p) pairs[p]=std::make_unique<PairPlan>(ctx,p,shape[axes[p][1]],shape[axes[p][0]],count,dtype);
    update_spacing(ctx,aniso);
  }
  void update_spacing(Context &ctx,std::array<float,3> aniso) {
    // Mark invalid before uploading: a failed update must not leave the cache
    // claiming its previous spacing still matches partially updated tables.
    spacing={NAN,NAN,NAN};
    for(auto &p:pairs) p->update_spacing(ctx,aniso);
    ctx.synchronize();
    spacing=aniso;
  }
  size_t bytes() const { size_t total=0;for(const auto &p:pairs) total+=p->bytes();return total; }
  template<class Label> BorderCandidates run(Context &ctx,const Array &input) {
    BorderCandidates out;out.shape=shape;out.samples=samples;
    for(int p=0;p<3;++p) { out.A[p]=pairs[p]->A;out.B[p]=pairs[p]->B;pairs[p]->run<Label>(ctx,input,shape,samples,out); }
    return out;
  }
};
BorderCandidates batched_border_candidates(Context &ctx,const Array &input,std::array<float,3> aniso,int samples) {
  if(samples<1 || input.shape[2]%samples) throw std::invalid_argument("stacked border samples must divide the z extent");
  const auto shape=BorderPlan::sample_shape(input,samples);
  const int axes[3][2]={{0,1},{0,2},{1,2}};
  for(int p=0;p<3;++p) {
    auto A=shape[axes[p][1]],B=shape[axes[p][0]];
    if(A<=0 || B<=0 || A>(INT32_MAX-1)/(int64_t(BORDER_PAIR_SLABS)*samples)/B) throw std::invalid_argument("border faces too large for packed keys");
  }
  if(!enabled("BROOK_BORDER_CACHE")) ctx.border_plan.reset();
  auto plan=ctx.border_plan;
  if(!plan || plan->shape!=shape || plan->samples!=samples || plan->dtype!=input.dtype ||
      (plan->spacing!=aniso && !enabled("BROOK_BORDER_REPARAM"))) {
    plan.reset();ctx.border_plan.reset();
    plan=std::make_shared<BorderPlan>(ctx,input,aniso,samples);
    ++ctx.stats["border_plan_builds"];
    size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
    if(plan->bytes()<=total/8 && enabled("BROOK_BORDER_CACHE")) ctx.border_plan=plan;   // batches of one shape reuse it
  } else if(plan->spacing!=aniso) {
    // The captured graph reads these tables through stable device pointers;
    // changing their contents preserves topology and all kernel arguments.
    plan->update_spacing(ctx,aniso);++ctx.stats["border_plan_reparams"];
  }
  switch(input.dtype) {
    case BROOK_U8:return plan->run<uint8_t>(ctx,input);
    case BROOK_U16:return plan->run<uint16_t>(ctx,input);
    case BROOK_U32:return plan->run<uint32_t>(ctx,input);
    case BROOK_U64:return plan->run<uint64_t>(ctx,input);
    default:throw std::invalid_argument("invalid border component dtype");
  }
}
}
