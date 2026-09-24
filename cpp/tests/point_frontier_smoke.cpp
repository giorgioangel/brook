#include "trace_primitives.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

int main() {
  brook::Context ctx(0);std::mt19937 rng(438);int checks=0;
  for(auto shape:std::vector<std::array<int64_t,3>>{{1,1,1},{129,7,5},{17,19,23},{64,64,64}}) {
    for(int density:{0,5,50,100}) {
      std::vector<uint8_t> mask(brook::volume_size(shape));
      for(auto &v:mask) v=(rng()%100)<unsigned(density);mask[0]=1;
      auto count=std::count(mask.begin(),mask.end(),uint8_t(1));
      brook_volume view{};view.struct_size=sizeof(view);view.abi_version=BROOK_ABI_VERSION;
      view.dtype=BROOK_U8;view.memory=BROOK_HOST;view.data=mask.data();int64_t stride=1;
      for(int a=0;a<3;++a) {view.shape[a]=shape[a];view.strides[a]=stride;stride*=shape[a];}
      auto input=brook::upload(ctx,view);
      for(auto spacing:std::vector<std::array<float,3>>{{1,1,1},{.37f,1.13f,2.9f}}) {
        auto expected=brook::euclidean_distance_field(ctx,input,0,spacing,0,nullptr,true);
        std::vector<float> reference(mask.size());expected.distance.copy_to_host(reference.data(),reference.size()*4);
        for(int repeat=0;repeat<3;++repeat) {
          auto actual=brook::euclidean_distance_field(ctx,input,0,spacing,0,nullptr,true,size_t(count));
          std::vector<float> values(mask.size());actual.distance.copy_to_host(values.data(),values.size()*4);
          if(actual.maximum!=expected.maximum || std::memcmp(values.data(),reference.data(),values.size()*4)) return 1;
          ++checks;
        }
      }
    }
  }
  std::cout<<"PASS: "<<checks<<" exact bounded/unbounded foreground fields\n";
}
