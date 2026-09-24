#include <brook/brook.hpp>
#include <algorithm>
#include <vector>

int main() {
  std::vector<int16_t> labels(9*7*5,-7);
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=labels.data();volume.dtype=BROOK_I16;volume.memory=BROOK_HOST;
  volume.shape[0]=9;volume.shape[1]=7;volume.shape[2]=5;
  const float vertices[7][3]={{2.5900001525878906f,4.519999980926514f,5.800000190734863f},
    {2.2200000286102295f,3.3899998664855957f,5.800000190734863f},
    {1.850000023841858f,2.259999990463257f,5.800000190734863f},
    {1.4800000190734863f,2.259999990463257f,5.800000190734863f},
    {1.1100000143051147f,2.259999990463257f,5.800000190734863f},
    {0.7400000095367432f,2.259999990463257f,5.800000190734863f},
    {0.3700000047683716f,2.259999990463257f,5.800000190734863f}};
  const float radii[7]={.7400000095367432f,1.1100000143051147f,1.4800000190734863f,
    1.850000023841858f,1.4800000190734863f,1.1100000143051147f,.7400000095367432f};
  brook::api::Context ctx;
  for(bool c_order:{false,true}) {
    volume.strides[0]=c_order?70:2;volume.strides[1]=c_order?10:18;volume.strides[2]=c_order?2:126;
    auto path=ctx.connect_points(volume,{1,2,2},{7,4,2},{.37f,1.13f,2.9f});auto view=path.view();
    if(view.skeleton_count!=1 || view.labels[0]!=0 || view.vertex_count!=7 || view.edge_count!=6) return 1;
    for(int i=0;i<7;++i) {
      for(int a=0;a<3;++a) if(view.vertices[3*i+a]!=vertices[i][a]) return 2;
      if(view.radii[i]!=radii[i]) return 3;
      if(i<6 && (view.edges[2*i]!=unsigned(i) || view.edges[2*i+1]!=unsigned(i+1))) return 4;
    }
    auto singleton=ctx.connect_points(volume,{1,2,2},{1,2,2},{.37f,1.13f,2.9f});auto one=singleton.view();
    if(one.vertex_count!=1 || one.edge_count || one.radii[0]!=radii[6]) return 5;
  }
  // A ball whose float32 EDT depends on the pass order. Kimimaro copies the mask to Fortran
  // order, so both layouts give its values (the ball is symmetric under axis permutation).
  std::vector<int16_t> ball(7*7*7);
  for(int z=0;z<7;++z) for(int y=0;y<7;++y) for(int x=0;x<7;++x)
    ball[x+7*(y+7*z)]=(x-3)*(x-3)+(y-3)*(y-3)+(z-3)*(z-3)<=6?-7:0;
  const float ball_vertices[5][3]={{5.5f,5.199999809265137f,6.800000190734863f},
    {4.400000095367432f,3.8999998569488525f,5.100000381469727f},
    {3.3000001907348633f,3.8999998569488525f,5.100000381469727f},
    {2.200000047683716f,3.8999998569488525f,5.100000381469727f},
    {1.100000023841858f,2.5999999046325684f,3.4000000953674316f}};
  const float ball_radii[5]={1.100000023841858f,2.200000047683716f,3.3000001907348633f,
    2.200000047683716f,1.100000023841858f};
  volume.data=ball.data();volume.shape[0]=volume.shape[1]=volume.shape[2]=7;
  for(bool c_order:{false,true}) {
    volume.strides[0]=c_order?98:2;volume.strides[1]=14;volume.strides[2]=c_order?2:98;
    auto path=ctx.connect_points(volume,{1,2,2},{5,4,4},{1.1f,1.3f,1.7f});auto view=path.view();
    if(view.vertex_count!=5 || view.edge_count!=4) return 7;
    for(int i=0;i<5;++i) {
      for(int a=0;a<3;++a) if(view.vertices[3*i+a]!=ball_vertices[i][a]) return 8;
      if(view.radii[i]!=ball_radii[i]) return 9;
    }
  }
  return std::all_of(labels.begin(),labels.end(),[](auto v){return v==-7;})?0:6;
}
