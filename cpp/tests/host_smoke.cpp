#include <brook/brook.hpp>
#include <array>

int main() {
  const uint64_t labels[]={0,7,0,7};
  brook_volume volume{};volume.struct_size=sizeof(volume);volume.abi_version=BROOK_ABI_VERSION;
  volume.data=labels;volume.dtype=BROOK_U64;volume.memory=BROOK_HOST;
  volume.shape[0]=volume.shape[1]=2;volume.shape[2]=1;
  volume.strides[0]=8;volume.strides[1]=16;volume.strides[2]=32;
  brook_label_query query{};query.kind=BROOK_QUERY_UNSIGNED;query.integer_label=7;
  query.centroid[0]=1;query.centroid[1]=0.5;
  auto absent=query;absent.integer_label=UINT64_MAX;
  auto far=query;far.centroid[0]=far.centroid[1]=100;
  auto result=brook::api::nearest_label_voxels_host(volume,{query,absent,far});
  if(result!=std::vector<std::array<int64_t,3>>{{1,0,0},{-1,-1,-1},{1,1,0}}) return 1;
  int64_t coordinates[3];volume.memory=BROOK_DEVICE;
  if(brook_nearest_label_voxels(nullptr,&volume,&query,1,coordinates,3)!=BROOK_INVALID_ARGUMENT) return 2;
  return 0;
}
