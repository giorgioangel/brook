#include <brook/brook.hpp>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  constexpr int sx=17,sy=15,sz=11;
  std::vector<uint16_t> labels(sx*sy*sz,0);
  for(int z=2;z<8;++z) for(int y=3;y<10;++y) for(int x=0;x<sx;++x) labels[x+sx*(y+sy*z)]=7;
  for(int z=8;z<11;++z) for(int y=11;y<14;++y) for(int x=2;x<12;++x) labels[x+sx*(y+sy*z)]=19;
  brook_volume input{};input.struct_size=sizeof(input);input.abi_version=BROOK_ABI_VERSION;
  input.data=labels.data();input.dtype=BROOK_U16;input.memory=BROOK_HOST;
  input.shape[0]=sx;input.shape[1]=sy;input.shape[2]=sz;
  input.strides[0]=2;input.strides[1]=2*sx;input.strides[2]=2*sx*sy;
  brook::api::Context ctx;auto options=brook::api::skeletonize_defaults();
  options.dust_threshold=0;options.teasar.constant=2;options.teasar.soma_detection_threshold=1e6;
  options.anisotropy[0]=.37f;options.anisotropy[1]=1.13f;options.anisotropy[2]=2.9f;
  for(bool fix:{false,true}) {
    options.flags=BROOK_FIX_BORDERS|(fix?BROOK_FIX_BRANCHING:0);
    auto ref=ctx.skeletonize(input,options);auto expected=ref.view();
    for(size_t budget:{size_t{1},size_t{64}<<20}) for(bool device:{false,true}) {
      options.flags=BROOK_FIX_BORDERS|(fix?BROOK_FIX_BRANCHING:0)|(device?BROOK_DEVICE_OUTPUT:0);
      auto actual=ctx.skeletonize_streamed(input,options,budget);auto got=actual.view();
      if(got.skeleton_count!=expected.skeleton_count || got.vertex_count!=expected.vertex_count || got.edge_count!=expected.edge_count) return 1;
      if(std::memcmp(got.labels,expected.labels,got.skeleton_count*sizeof(int64_t)) ||
         std::memcmp(got.vertices,expected.vertices,got.vertex_count*3*sizeof(float)) ||
         std::memcmp(got.edges,expected.edges,got.edge_count*2*sizeof(uint32_t)) ||
         std::memcmp(got.radii,expected.radii,got.vertex_count*sizeof(float))) return 2;
      if(device && actual.device_view().vertex_count!=got.vertex_count) return 3;
    }
  }
  std::puts("PASS: streamed tracing C/C++ SDK, branching modes, budgets and host/device outputs");
  return 0;
}
