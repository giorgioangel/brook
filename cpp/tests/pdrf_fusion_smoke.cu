#include "trace_primitives.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

namespace {
// Original two-pass normalization is the oracle, including conditional stores.
__global__ void normalize(float *dbf,float *daf,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { if(dbf[i]==0) dbf[i]=__int_as_float(0x7f800000);if(isinf(daf[i])) daf[i]=0; }
}
// Original PDRF arithmetic, independent of the production template.
__global__ void original_pdrf(const float *dbf,const float *daf,float *out,int n,float m,float scale,
                             float inverse,float exponent,int squarings) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  float p=1.0f-dbf[i]*m;
  if(squarings>=0) for(int k=0;k<squarings;++k) p=p*p;
  else p=powf(p,exponent);
  p=p*scale;
  if(inverse!=0) p=p+daf[i]*inverse;
  out[i]=p;
}
brook::Array original(brook::Context &ctx,const brook::Array &dbf,const brook::Array &daf,
                      double maximum,double max_daf,double scale,double exponent) {
  float m=static_cast<float>(1.0/std::pow(maximum,1.01));
  int squarings=-1;
  if(exponent>0 && exponent<65536 && std::floor(exponent)==exponent && !(int(exponent)&(int(exponent)-1))) {
    squarings=0;for(int e=int(exponent);e>1;e>>=1) ++squarings;
  }
  auto out=brook::allocate(ctx,dbf.shape,BROOK_F32);
  original_pdrf<<<unsigned((dbf.size()+255)/256),256,0,ctx.stream>>>(
    static_cast<const float*>(dbf.data()),static_cast<const float*>(daf.data()),static_cast<float*>(out.data()),
    int(dbf.size()),m,float(scale),max_daf?float(1.0/max_daf):0.0f,float(exponent),squarings);
  BROOK_CUDA(cudaGetLastError());ctx.synchronize();return out;
}
brook::Array separate(brook::Context &ctx,brook::Array &dbf,brook::Array &daf,
                      double maximum,double max_daf,double scale,double exponent) {
  normalize<<<unsigned((dbf.size()+255)/256),256,0,ctx.stream>>>(
    static_cast<float*>(dbf.data()),static_cast<float*>(daf.data()),int(dbf.size()));
  BROOK_CUDA(cudaGetLastError());
  return brook::pdrf(ctx,dbf,daf,maximum,max_daf,scale,exponent);
}
void set(brook::Context &ctx,brook::Array &a,const std::vector<float> &values) {
  BROOK_CUDA(cudaMemcpyAsync(a.data(),values.data(),a.bytes(),cudaMemcpyHostToDevice,ctx.stream));
}
std::vector<float> host(const brook::Array &a) {
  std::vector<float> values(a.size());a.copy_to_host(values.data(),a.bytes());return values;
}
void equal(const std::vector<float> &a,const std::vector<float> &b,const char *what) {
  if(a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))
    throw std::runtime_error(std::string("bit mismatch: ")+what);
}
void check(brook::Context &ctx) {
  const std::array<int64_t,3> shape={19,17,13};
  auto dbf=brook::allocate(ctx,shape,BROOK_F32),daf=brook::allocate(ctx,shape,BROOK_F32);
  auto fused_dbf=brook::allocate(ctx,shape,BROOK_F32),fused_daf=brook::allocate(ctx,shape,BROOK_F32);
  std::mt19937 rng(470);std::uniform_real_distribution<float> value(-2,10);
  std::vector<float> boundary(dbf.size()),distance(dbf.size());
  for(size_t i=0;i<boundary.size();++i) { boundary[i]=value(rng);distance[i]=value(rng); }
  const uint32_t special[]={0,0x80000000,0x7f800000,0xff800000,0x7fc01234,0xffc04321,
                            1,0x80000001,0x7f7fffff,0xff7fffff,0x3f800000,0x7f801234};
  for(size_t i=0;i<sizeof(special)/sizeof(*special);++i)
    for(size_t j=0;j<sizeof(special)/sizeof(*special);++j) {
      std::memcpy(&boundary[12*i+j],special+i,4);std::memcpy(&distance[12*i+j],special+j,4);
    }
  int checks=0;
  for(double maximum:{10.0,0.0,1e-30,1e30,double(INFINITY),double(NAN)})
    for(double max_daf:{10.0,0.0,-0.0,1e-300,double(INFINITY),double(NAN)})
      for(double scale:{5000.0,0.0,-1.0})
        for(double exponent:{1.0,2.0,4.0,16.0,0.0,-1.0,2.5,65536.0}) {
          set(ctx,dbf,boundary);set(ctx,daf,distance);
          auto readonly=brook::pdrf(ctx,dbf,daf,maximum,max_daf,scale,exponent);
          auto old_readonly=original(ctx,dbf,daf,maximum,max_daf,scale,exponent);
          equal(host(readonly),host(old_readonly),"read-only PDRF arithmetic");
          equal(host(dbf),boundary,"read-only DBF ownership");equal(host(daf),distance,"read-only DAF ownership");
          set(ctx,fused_dbf,boundary);set(ctx,fused_daf,distance);
          auto expected=separate(ctx,dbf,daf,maximum,max_daf,scale,exponent);
          auto old_normalized=original(ctx,dbf,daf,maximum,max_daf,scale,exponent);
          equal(host(expected),host(old_normalized),"separate PDRF arithmetic");
          auto actual=brook::normalize_pdrf(ctx,fused_dbf,fused_daf,maximum,max_daf,scale,exponent);
          equal(host(dbf),host(fused_dbf),"normalized DBF");equal(host(daf),host(fused_daf),"normalized DAF");
          equal(host(expected),host(actual),"PDRF");++checks;
        }
  std::cout<<"PASS: "<<checks<<" bitwise normalized-field/PDRF comparisons and read-only ownership checks\n";
}
void benchmark(brook::Context &ctx) {
  for(int side:{19,64,256,512}) {
    std::array<int64_t,3> shape={side,side,side};
    auto dbf=brook::allocate(ctx,shape,BROOK_F32),daf=brook::allocate(ctx,shape,BROOK_F32);
    std::vector<float> boundary(dbf.size()),distance(dbf.size());
    for(size_t i=0;i<dbf.size();++i) { boundary[i]=i%7?float(i%100)*0.1f:0;distance[i]=i%7?float(i%1000):INFINITY; }
    std::vector<double> times[2];std::vector<float> reference;
    for(int rep=-2;rep<11;++rep) for(int turn=0;turn<2;++turn) {
      int mode=(rep+turn+2)%2;
      set(ctx,dbf,boundary);set(ctx,daf,distance);ctx.synchronize();
      auto start=std::chrono::steady_clock::now();
      auto out=mode?brook::normalize_pdrf(ctx,dbf,daf,10,1000,5000,16):separate(ctx,dbf,daf,10,1000,5000,16);
      double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(rep>=0) times[mode].push_back(ms);
      auto values=host(out);
      if(reference.empty()) reference=std::move(values);else equal(reference,values,"benchmark PDRF");
    }
    std::cout<<"{\"side\":"<<side<<",\"separate_ms\":[";
    for(int mode=0;mode<2;++mode) {
      if(mode) std::cout<<"],\"fused_ms\":[";
      for(size_t i=0;i<times[mode].size();++i) { if(i) std::cout<<',';std::cout<<times[mode][i]; }
    }
    std::cout<<"],\"exact\":true}"<<std::endl;
  }
}
}
int main(int argc,char **argv) {
  try {
    brook::Context ctx(0);
    if(argc==2 && std::string(argv[1])=="--benchmark") benchmark(ctx);else check(ctx);
    return 0;
  } catch(const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
