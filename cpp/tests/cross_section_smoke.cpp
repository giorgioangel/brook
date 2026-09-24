#include <brook/brook.hpp>
#include <cassert>
#include <vector>
int main() {
  std::vector<uint8_t> labels(13*15*17,7);
  std::vector<float> vertices;std::vector<uint32_t> edges;
  for(uint32_t i=0;i<13;++i) {vertices.insert(vertices.end(),{float(i),7,8});if(i) edges.insert(edges.end(),{i-1,i});}
  std::vector<float> areas(13);std::vector<uint8_t> contacts(13);
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=labels.data();volume.shape[0]=13;volume.shape[1]=15;volume.shape[2]=17;
  volume.strides[0]=1;volume.strides[1]=13;volume.strides[2]=13*15;volume.dtype=BROOK_U8;volume.memory=BROOK_HOST;
  brook_cross_section_input input{};input.label=7;input.vertex_count=13;input.edge_count=12;
  input.vertices=vertices.data();input.edges=edges.data();input.physical=1;input.areas=areas.data();input.contacts=contacts.data();
  brook_cross_section_options options;brook_cross_section_options_init(&options);
  brook::api::cross_sectional_area(volume,{input},options);
  for(size_t i=0;i<13;++i) {assert(areas[i]==255);assert(contacts[i]==uint8_t(60|(i==0?1:i==12?2:0)));}
  options.step=0;assert(brook_cross_sectional_area(&volume,&input,1,&options)==BROOK_INVALID_ARGUMENT);
  for(float a:areas) assert(a==255); // Failed calls do not partially overwrite outputs.
}
