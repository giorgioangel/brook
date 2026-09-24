#include <brook/brook.hpp>
#include <vector>

int main() {
  std::vector<uint32_t> image(32*8*8,0);
  for(int z=3;z<5;++z) for(int y=3;y<5;++y) for(int x=2;x<30;++x) image[x+32*(y+8*z)]=42;
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=image.data();volume.dtype=BROOK_U32;volume.memory=BROOK_HOST;
  volume.shape[0]=32;volume.shape[1]=volume.shape[2]=8;
  volume.strides[0]=4;volume.strides[1]=128;volume.strides[2]=1024;
  auto options=brook::api::skeletonize_defaults();options.dust_threshold=0;
  options.teasar.scale=1.5;options.teasar.constant=2;options.flags=BROOK_FIX_BRANCHING;
  auto held=[&] {
    brook::api::Context context;
    auto components=context.connected_components(volume);
    auto view=components.view();
    if(view.count!=1 || view.mapping[1]!=42) throw std::runtime_error("wrong C++ component mapping");
    return components.labels();
  }();
  auto tensor=held.dlpack();
  if(tensor->dl_tensor.shape[0]!=32 || tensor->dl_tensor.device.device_type!=kDLCUDA) return 3;
  auto result=[&] {
    brook::api::Context context;
    return context.skeletonize(volume,options);
  }();
  const auto data=result.view();
  if(data.skeleton_count!=1 || data.labels[0]!=42 || !data.edge_count || !data.vertex_count) return 1;
  if(data.vertex_offsets[1]!=int64_t(data.vertex_count) || data.edge_offsets[1]!=int64_t(data.edge_count)) return 2;
  options.flags|=BROOK_DEVICE_OUTPUT;
  auto device_result=[&] { brook::api::Context context;return context.skeletonize(volume,options); }();
  const auto device_view=device_result.device_view();
  if(device_view.vertex_count!=data.vertex_count || device_view.edge_count!=data.edge_count ||
      device_view.vertices.memory!=BROOK_DEVICE || device_view.vertices.shape[1]!=3 || device_view.vertices.strides[0]!=12) return 4;
  auto vertex_tensor=device_result.array(BROOK_VERTICES).dlpack();
  if(vertex_tensor->dl_tensor.ndim!=2 || vertex_tensor->dl_tensor.shape[1]!=3) return 5;
  const auto snapshot=device_result.view();
  for(size_t i=0;i<3*data.vertex_count;++i) if(snapshot.vertices[i]!=data.vertices[i]) return 6;
  for(size_t i=0;i<2*data.edge_count;++i) if(snapshot.edges[i]!=data.edges[i]) return 7;
  brook::api::Context context;
  brook_label_query query{};query.kind=BROOK_QUERY_UNSIGNED;query.integer_label=42;
  query.centroid[0]=15;query.centroid[1]=query.centroid[2]=3.5;
  auto nearest=context.nearest_label_voxels(volume,{query});
  if(nearest.size()!=1 || nearest[0]!=std::array<int64_t,3>{15,3,3}) return 11;
  auto merged=context.merge({&result,&device_result},{{0,0,0},{0,0,0}});
  auto processed=context.postprocess_canonical(merged,0,0);
  const auto after=processed.view();
  if(after.skeleton_count!=data.skeleton_count || after.vertex_count!=data.vertex_count || after.edge_count!=data.edge_count) return 8;
  for(size_t i=0;i<3*data.vertex_count;++i) if(after.vertices[i]!=data.vertices[i]) return 9;
  for(size_t i=0;i<2*data.edge_count;++i) if(after.edges[i]!=data.edges[i]) return 10;
  return 0;
}
