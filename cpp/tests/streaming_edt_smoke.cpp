#include <brook/brook.hpp>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  constexpr int sx=31,sy=25,sz=21;
  std::vector<int16_t> labels(sx*sy*sz,0);
  for(int z=1;z<sz-2;++z) for(int y=2;y<sy-1;++y) for(int x=1;x<sx-1;++x)
    labels[x+sx*(y+sy*z)]=(x<11?-7:3);
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=labels.data();volume.dtype=BROOK_I16;volume.memory=BROOK_HOST;
  volume.shape[0]=sx;volume.shape[1]=sy;volume.shape[2]=sz;
  volume.strides[0]=2;volume.strides[1]=2*sx;volume.strides[2]=2*sx*sy;
  const float spacing[3]={.37f,1.13f,2.9f};
  brook::api::Context ctx;
  std::vector<float> expected(labels.size()),out(labels.size());
  for(bool border:{false,true}) {
    auto reference=ctx.edt(volume,spacing,border);
    reference.copy_to_host(expected.data(),expected.size()*sizeof(float));
    for(size_t budget:{size_t{1},size_t{100000},size_t{10000000}}) {
      ctx.edt_streamed(volume,spacing,out.data(),out.size()*sizeof(float),border,budget);
      if(std::memcmp(out.data(),expected.data(),out.size()*sizeof(float))) return 1;
    }
  }
  bool rejected=false;
  try {ctx.edt_streamed(volume,spacing,out.data(),sizeof(float));}
  catch(const std::exception&) {rejected=true;}
  if(!rejected) return 2;
  // Enough tiles to exercise the C-order GPU transpose, with partial last slabs.
  constexpr int lx=133,ly=129,lz=67;
  labels.assign(lx*ly*lz,0);expected.resize(labels.size());out.resize(labels.size());
  for(int x=1;x<lx-1;++x) for(int y=2;y<ly-2;++y) for(int z=3;z<lz-3;++z)
    labels[z+lz*(y+ly*x)]=(x<60?-7:3);
  volume.data=labels.data();volume.shape[0]=lx;volume.shape[1]=ly;volume.shape[2]=lz;
  volume.strides[0]=2*ly*lz;volume.strides[1]=2*lz;volume.strides[2]=2;
  auto reference=ctx.edt(volume,spacing,true);
  reference.copy_to_host(expected.data(),expected.size()*sizeof(float));
  ctx.edt_streamed(volume,spacing,out.data(),out.size()*sizeof(float),true,size_t{8}<<20);
  if(std::memcmp(out.data(),expected.data(),out.size()*sizeof(float))) return 3;
  std::puts("PASS: streamed EDT C/C++ SDK budgets, exact bytes and undersized output rejection");
  return 0;
}
