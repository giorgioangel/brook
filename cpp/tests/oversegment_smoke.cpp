#include <brook/brook.hpp>
#include <algorithm>
#include <iostream>
int main() {
 try {
  std::vector<uint16_t> labels(15,7);float vertices[]={0,1,0,4,1,0};uint32_t edges[]={0,1};
  int64_t ids[]={7},voff[]={0,2},eoff[]={0,1};
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=labels.data();volume.dtype=BROOK_U16;volume.memory=BROOK_HOST;
  volume.shape[0]=5;volume.shape[1]=3;volume.shape[2]=1;volume.strides[0]=2;volume.strides[1]=10;volume.strides[2]=30;
  brook_skeletons_view input{};input.skeleton_count=1;input.vertex_count=2;input.edge_count=1;
  input.labels=ids;input.vertex_offsets=voff;input.edge_offsets=eoff;input.vertices=vertices;input.edges=edges;
  auto output=brook::api::oversegment(volume,input);
  const std::vector<uint64_t> expected={1,1,2,2,2,1,1,2,2,2,1,1,2,2,2};
  if(output.features!=expected || output.segments!=std::vector<uint64_t>{1,2} || output.processed!=std::vector<uint8_t>{1}) return 1;
  if(std::any_of(labels.begin(),labels.end(),[](auto v){return v!=7;})) return 2;
  uint64_t tiny=0;
  if(brook_oversegment(nullptr,&volume,&input,std::array<double,3>{1,1,1}.data(),0,0,&tiny,1,nullptr,0,nullptr,0)!=BROOK_INVALID_ARGUMENT) return 3;
  std::cout<<"PASS: C/C++ oversegment ownership and output capacity\n";return 0;
 } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 4;}
}
