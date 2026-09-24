#include <brook/brook.hpp>
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

int main() {
  constexpr int sx=17,sy=15,sz=11;
  std::vector<uint16_t> labels(sx*sy*sz,0);
  for(int z=0;z<sz;++z) {labels[2+sx*(2+sy*z)]=7;labels[12+sx*(12+sy*z)]=7;labels[8+sx*(8+sy*z)]=19;}
  for(int y=2;y<=12;++y) for(int x=2;x<=12;++x) labels[x+sx*(y+sy*8)]=7;
  brook_volume input{};input.struct_size=sizeof(input);input.abi_version=BROOK_ABI_VERSION;
  input.data=labels.data();input.dtype=BROOK_U16;input.memory=BROOK_HOST;
  input.shape[0]=sx;input.shape[1]=sy;input.shape[2]=sz;
  input.strides[0]=2;input.strides[1]=2*sx;input.strides[2]=2*sx*sy;
  std::optional<brook::api::HostComponents> held;
  std::vector<uint8_t> expected;
  {
    brook::api::Context ctx;
    auto in_core=ctx.connected_components(input);auto reference=in_core.view();
    if(reference.labels.dtype!=BROOK_U8) return 1;
    expected.resize(labels.size());in_core.labels().copy_to_host(expected.data(),expected.size());
    for(size_t budget:{size_t{1},size_t{26010},size_t{64}<<20}) {
      held=ctx.connected_components_streamed(input,budget);auto view=held->view();
      if(view.labels.memory!=BROOK_HOST || view.labels.dtype!=reference.labels.dtype || view.count!=reference.count) return 2;
      if(std::memcmp(view.labels.data,expected.data(),expected.size()) ||
         std::memcmp(view.mapping,reference.mapping,(view.count+1)*sizeof(int64_t)) ||
         std::memcmp(view.roots,reference.roots,view.count*sizeof(int64_t))) return 3;
    }
  }
  if(std::memcmp(held->view().labels.data,expected.data(),expected.size())) return 4;
  std::puts("PASS: streamed CCL C/C++ SDK seam stitching, exact IDs/roots and host lifetime");
  return 0;
}
