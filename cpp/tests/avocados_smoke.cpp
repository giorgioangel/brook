#include <brook/brook.hpp>
#include <cstdio>
#include <vector>

int main() {
  constexpr int side=19;
  std::vector<uint8_t> nested(side*side*side);
  for(int z=2;z<17;++z) for(int y=2;y<17;++y) for(int x=2;x<17;++x) {
    int i=x+side*(y+side*z);
    nested[i]=(x>=6 && x<13 && y>=6 && y<13 && z>=6 && z<13)?77:42;
  }
  auto original=nested;
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=nested.data();volume.dtype=BROOK_U8;volume.memory=BROOK_HOST;
  for(int a=0;a<3;++a) volume.shape[a]=side;
  volume.strides[0]=1;volume.strides[1]=side;volume.strides[2]=side*side;
  // The fruit's own pass paints the pit without recording a merge, so, as in Kimimaro, the distance
  // field is not recomputed and the trace differs from a solid cube's. Without branching correction
  // this is Kimimaro 5.8.1's skeleton; with it Brook returns the same route mirrored (x and z
  // swapped), which has equal cost.
  const float vertices[24][3]={{2,2,2},{3,3,3},{4,4,4},{5,5,4},{6,6,4},{7,7,4},
    {8,8,4},{9,9,4},{10,10,4},{11,11,4},{12,12,4},{13,13,4},{14,14,5},{14,14,6},
    {14,14,7},{14,14,8},{14,14,9},{14,14,10},{14,14,11},{14,14,12},{14,14,13},
    {14,14,14},{15,15,15},{16,16,16}};
  const float radii[24]={1,2,3,2.4494898319244385f,2,2,2,2,2,2,2,2.4494898319244385f,
    3,2.8284270763397217f,2.8284270763397217f,2.8284270763397217f,2.8284270763397217f,
    2.8284270763397217f,2.8284270763397217f,2.8284270763397217f,3,3,2,1};
  brook::api::Context context;
  auto options=brook::api::skeletonize_defaults();options.dust_threshold=0;
  // A partial Python parameter dictionary uses trace()'s own defaults.
  options.teasar.scale=10;options.teasar.pdrf_scale=5000;options.teasar.pdrf_exponent=16;
  options.teasar.soma_invalidation_scale=0.5;options.teasar.soma_invalidation_const=0;
  options.teasar.constant=2;options.teasar.soma_detection_threshold=1;
  options.teasar.soma_acceptance_threshold=100000;
  for(unsigned flags:{0u,unsigned(BROOK_FIX_BRANCHING)}) {
    for(unsigned output:{0u,unsigned(BROOK_DEVICE_OUTPUT)}) {
      options.flags=flags|output|BROOK_FIX_AVOCADOS;
      auto actual=context.skeletonize(volume,options);auto got=actual.view();
      if(got.skeleton_count!=1 || got.labels[0]!=42 || got.vertex_count!=24 || got.edge_count!=23) {
        std::fprintf(stderr,"flags=%u: skeletons=%zu vertices=%zu edges=%zu\n",options.flags,got.skeleton_count,got.vertex_count,got.edge_count);return 1;
      }
      for(int i=0;i<24;++i) {
        for(int a=0;a<3;++a) if(got.vertices[3*i+a]!=vertices[i][flags?2-a:a]) return 2;
        if(got.radii[i]!=radii[i]) return 3;
        if(i<23 && (got.edges[2*i]!=unsigned(i) || got.edges[2*i+1]!=unsigned(i+1))) return 4;
      }
    }
  }
  // Kimimaro runs the stage on Fortran-ordered arrays: the argmax takes the first maximum in x-fastest
  // order and labels are numbered by first voxel in that order. A pit (2) whose lobes tie at distance 2,
  // one inside a one-voxel fruit shell (1), one in the open: the enclosed lobe comes first and the pit
  // merges (label 2 remains); with x and z swapped the open lobe comes first and both remain, pit first.
  for(int swapped=0;swapped<2;++swapped) {
    constexpr int n=16,ny=9;std::vector<uint8_t> tie(n*ny*n,0);
    auto fill=[&](int x0,int x1,int y0,int y1,int z0,int z1,uint8_t value) {
      for(int z=z0;z<z1;++z) for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) tie[swapped?z+n*(y+ny*x):x+n*(y+ny*z)]=value;
    };
    fill(8,13,1,6,1,6,1);fill(9,12,2,5,2,5,2);fill(3,9,2,3,2,3,2);fill(3,4,2,3,2,9,2);fill(2,5,2,5,9,12,2);
    brook_volume pit=volume;pit.data=tie.data();pit.shape[0]=pit.shape[2]=n;pit.shape[1]=ny;pit.strides[1]=n;pit.strides[2]=n*ny;
    auto settings=brook::api::skeletonize_defaults();settings.dust_threshold=0;settings.flags=BROOK_FIX_AVOCADOS;
    settings.teasar.scale=1.5;settings.teasar.constant=1;settings.teasar.pdrf_scale=100000;settings.teasar.pdrf_exponent=4;
    settings.teasar.soma_detection_threshold=4.5;settings.teasar.soma_acceptance_threshold=1e6;
    settings.teasar.soma_invalidation_scale=1;settings.teasar.soma_invalidation_const=0;
    auto result=context.skeletonize(pit,settings);auto got=result.view();
    const int64_t labels[2]={2,1};const size_t count=swapped?2:1;
    if(got.skeleton_count!=count) return 6;
    for(size_t i=0;i<count;++i) if(got.labels[i]!=labels[i]) return 7;
  }
  return nested==original?0:5;
}
