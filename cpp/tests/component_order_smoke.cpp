#include "lockstep_order.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

int main() {
  brook::Context ctx(0);std::mt19937 rng(482);int checks=0;
  for(int n:{1,17,1025,100003}) for(int density:{0,3,50,100}) for(int ties:{1,7,10000}) {
    std::vector<uint8_t> mask(n);std::vector<float> daf(n);std::vector<int> expected;
    for(int i=0;i<n;++i) {
      mask[i]=(rng()%100<unsigned(density))?uint8_t(1+rng()%7):0;
      daf[i]=float(rng()%unsigned(ties))*.37f;
      if(i%19==0) daf[i]=INFINITY;
      if(i%23==0) daf[i]=-0.0f;
      if(mask[i]) expected.push_back(i);
    }
    std::sort(expected.begin(),expected.end(),[&](int a,int b) {
      float x=std::isinf(daf[a])?0:daf[a],y=std::isinf(daf[b])?0:daf[b];return x==y?a>b:x>y;
    });
    auto dm=brook::allocate(ctx,{n,1,1},BROOK_U8),dd=brook::allocate(ctx,{n,1,1},BROOK_F32);
    BROOK_CUDA(cudaMemcpyAsync(dm.data(),mask.data(),mask.size(),cudaMemcpyHostToDevice,ctx.stream));
    BROOK_CUDA(cudaMemcpyAsync(dd.data(),daf.data(),daf.size()*4,cudaMemcpyHostToDevice,ctx.stream));
    auto result=brook::build_component_target_order(ctx,dm,dd,expected.size()).get(ctx,expected.size());
    if(result!=expected) return 1;
    ++checks;
  }
  std::cout<<"PASS: "<<checks<<" exact GPU target orders, including ties/infinity/signed zero/nonbinary masks\n";
}
