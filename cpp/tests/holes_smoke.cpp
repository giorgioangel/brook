#include <brook/brook.hpp>
#include <cstring>
#include <vector>

int main() {
  constexpr int side=15;
  std::vector<uint8_t> hollow(side*side*side),solid(hollow.size());
  for(int z=2;z<13;++z) for(int y=2;y<13;++y) for(int x=2;x<13;++x) {
    int i=x+side*(y+side*z);solid[i]=42;
    if(x==2 || x==12 || y==2 || y==12 || z==2 || z==12) hollow[i]=42;
    else if(x>=6 && x<9 && y>=6 && y<9 && z>=6 && z<9) hollow[i]=77;
  }
  auto original=hollow;
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=hollow.data();volume.dtype=BROOK_U8;volume.memory=BROOK_HOST;
  for(int a=0;a<3;++a) volume.shape[a]=side;
  volume.strides[0]=1;volume.strides[1]=side;volume.strides[2]=side*side;
  auto filled=volume;filled.data=solid.data();
  brook::api::Context context;
  auto options=brook::api::skeletonize_defaults();options.dust_threshold=0;
  options.teasar.scale=1.5;options.teasar.constant=2;
  // The filled hollow box traces as the solid box, bit for bit; failures return base+1..base+3.
  auto compare=[&](int base) {
    for(unsigned flags:{0u,unsigned(BROOK_FIX_BRANCHING)}) {
      options.flags=flags;
      auto expected=context.skeletonize(filled,options);auto ref=expected.view();
      if(ref.skeleton_count!=1 || ref.labels[0]!=42 || !ref.vertex_count) return base+1;
      for(unsigned output:{0u,unsigned(BROOK_DEVICE_OUTPUT)}) {
        options.flags=flags|output|BROOK_FILL_HOLES;
        auto actual=context.skeletonize(volume,options);auto got=actual.view();
        if(got.skeleton_count!=ref.skeleton_count || got.vertex_count!=ref.vertex_count || got.edge_count!=ref.edge_count) return base+2;
        if(got.labels[0]!=42 || std::memcmp(got.vertices,ref.vertices,ref.vertex_count*3*sizeof(float)) ||
            std::memcmp(got.edges,ref.edges,ref.edge_count*2*sizeof(uint32_t)) ||
            std::memcmp(got.radii,ref.radii,ref.vertex_count*sizeof(float))) return base+3;
      }
    }
    return 0;
  };
  if(int code=compare(0)) return code;
  if(hollow!=original) return 4;
  // With a voxel graph the holes of each graph component are filled and the graph is used as given: a
  // graph that permits every step joins shell, gap and core into one component (42 by its last run),
  // which the fill makes the solid box.
  std::vector<uint32_t> words(hollow.size(),(1u<<26)-1);const auto given=words;
  brook_volume graph=volume;graph.data=words.data();graph.dtype=BROOK_U32;
  graph.strides[0]=4;graph.strides[1]=4*side;graph.strides[2]=4*side*side;
  options.voxel_graph=&graph;
  if(int code=compare(4)) return code;
  return hollow==original && words==given?0:8;
}
