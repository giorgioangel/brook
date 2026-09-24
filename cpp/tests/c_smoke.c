#include <brook/brook.h>
#include <brook/dlpack.h>
#include <stdio.h>
#include <string.h>

#define CHECK(call) do { if ((call) != BROOK_SUCCESS) { fprintf(stderr,"%s\n",brook_last_error()); return 1; } } while (0)
int main(void) {
  const uint8_t labels[8] = {1,1,0,0,0,0,2,2};
  brook_volume input = {0}, view = {0};
  input.struct_size=sizeof(input); input.abi_version=BROOK_ABI_VERSION;
  input.data=labels; input.dtype=BROOK_U8; input.memory=BROOK_HOST;
  input.shape[0]=8; input.shape[1]=input.shape[2]=1;
  input.strides[0]=1; input.strides[1]=input.strides[2]=8;
  brook_context *ctx=NULL;
  CHECK(brook_context_create(0,&ctx));
  brook_label_query queries[5]={{0}};
  for(int i=0;i<5;++i) queries[i].kind=BROOK_QUERY_UNSIGNED;
  queries[0].integer_label=1;queries[0].centroid[0]=0.5;
  queries[1].integer_label=3;
  queries[2].integer_label=0;queries[2].centroid[0]=4.9;
  queries[3].kind=BROOK_QUERY_SIGNED;queries[3].integer_label=(uint64_t)-1;
  queries[4].integer_label=257;
  int64_t nearest[15];CHECK(brook_nearest_label_voxels(ctx,&input,queries,5,nearest,15));
  const int64_t expected_nearest[15]={0,0,0,-1,-1,-1,5,0,0,-1,-1,-1,-1,-1,-1};
  if(memcmp(nearest,expected_nearest,sizeof(nearest))) return 10;
  if(brook_nearest_label_voxels(ctx,&input,queries,5,nearest,14)!=BROOK_INVALID_ARGUMENT) return 11;
  brook_components *cc=NULL;
  CHECK(brook_connected_components(ctx,&input,&cc));
  const int64_t *mapping=NULL,*roots=NULL; size_t count=0;
  CHECK(brook_components_view(cc,&view,&mapping,&roots,&count));
  if(count!=2 || mapping[1]!=1 || mapping[2]!=2 || roots[0]!=0 || roots[1]!=6) return 2;
  uint8_t result[8]; CHECK(brook_components_copy_to_host(cc,result,sizeof(result)));
  if(memcmp(result,labels,sizeof(labels))) return 3;
  brook_components_destroy(cc);
  const float aniso[3]={1,1,1}; brook_array *distance=NULL;
  CHECK(brook_edt(ctx,&input,aniso,0,&distance));
  const uint32_t bits[8]={0};
  brook_volume graph=input;graph.data=bits;graph.dtype=BROOK_U32;
  graph.strides[0]=4;graph.strides[1]=graph.strides[2]=32;
  CHECK(brook_graph_components(ctx,&input,&graph,&cc));
  CHECK(brook_components_view(cc,&view,&mapping,&roots,&count));
  CHECK(brook_components_copy_to_host(cc,result,sizeof(result)));
  /* an all-zero graph isolates every voxel: the four foreground voxels are components 1..4, in
   * root order, numbered without gaps for the background (as connected components are) */
  const uint8_t graph_expected[8]={1,2,0,0,0,0,3,4};
  if(count!=4 || memcmp(result,graph_expected,8) || mapping[3]!=2 || mapping[4]!=2 || roots[2]!=6 || roots[3]!=7) return 6;
  brook_components_destroy(cc);
  brook_array *graph_distance=NULL;
  CHECK(brook_graph_edt(ctx,&input,&graph,aniso,0,&graph_distance));
  float gd[8];CHECK(brook_array_copy_to_host(graph_distance,gd,sizeof(gd)));
  for(int i=0;i<8;++i) if(gd[i]!=(labels[i]?0.5f:0.0f)) return 7;
  brook_array_destroy(graph_distance);
  /* Results own their device allocation independently of the context. */
  brook_context_destroy(ctx);
  float d[8]; CHECK(brook_array_copy_to_host(distance,d,sizeof(d)));
  if(d[0]!=2 || d[1]!=1 || d[2]!=0 || d[6]!=1 || d[7]!=2) return 4;
  DLManagedTensor *exported=NULL;DLManagedTensorVersioned *versioned=NULL;
  CHECK(brook_array_to_dlpack(distance,&exported));
  CHECK(brook_array_to_dlpack_versioned(distance,0,&versioned));
  if(exported->dl_tensor.device.device_type!=kDLCUDA || exported->dl_tensor.ndim!=3 ||
      exported->dl_tensor.shape[0]!=8 || exported->dl_tensor.strides[0]!=1 ||
      versioned->version.major!=1 || versioned->version.minor!=0 || versioned->flags) return 8;
  brook_array_destroy(distance);
  CHECK(brook_context_create(0,&ctx));
  brook_volume borrowed=input;borrowed.data=(const char*)exported->dl_tensor.data+exported->dl_tensor.byte_offset;
  borrowed.dtype=BROOK_F32;borrowed.memory=BROOK_DEVICE;
  for(int i=0;i<3;++i) borrowed.strides[i]=exported->dl_tensor.strides[i]*4;
  CHECK(brook_array_upload(ctx,&borrowed,&distance));
  exported->deleter(exported);versioned->deleter(versioned);brook_context_destroy(ctx);
  float copied[8];CHECK(brook_array_copy_to_host(distance,copied,sizeof(copied)));
  if(memcmp(d,copied,sizeof(d))) return 9;
  brook_array_destroy(distance);
  if(brook_context_create(0,NULL)!=BROOK_INVALID_ARGUMENT) return 5;
  return 0;
}
