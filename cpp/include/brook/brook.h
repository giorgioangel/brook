// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
/* Brook C API (GPL-3.0-only). BROOK_ABI_VERSION 1 is provisional within 0.x. Calls block;
 * errors return a brook_status, and brook_last_error() holds the text (thread-local). */
#ifndef BROOK_BROOK_H
#define BROOK_BROOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BROOK_ABI_VERSION 1u

typedef struct brook_context brook_context;
typedef struct brook_array brook_array;
typedef struct brook_components brook_components;
typedef struct brook_host_components brook_host_components;
typedef struct brook_skeletons brook_skeletons;
/* Include brook/dlpack.h when accessing the exported tensors' fields. */
typedef struct DLManagedTensor DLManagedTensor;
typedef struct DLManagedTensorVersioned DLManagedTensorVersioned;

typedef enum brook_status {
  BROOK_SUCCESS = 0, BROOK_INVALID_ARGUMENT = 1, BROOK_OUT_OF_MEMORY = 2,
  BROOK_CUDA_ERROR = 3, BROOK_INTERNAL_ERROR = 4
} brook_status;

typedef enum brook_dtype {
  BROOK_U8 = 0, BROOK_U16 = 1, BROOK_U32 = 2, BROOK_U64 = 3,
  BROOK_I8 = 4, BROOK_I16 = 5, BROOK_I32 = 6, BROOK_I64 = 7,
  BROOK_F32 = 8, BROOK_F64 = 9
} brook_dtype;

typedef enum brook_memory { BROOK_HOST = 0, BROOK_DEVICE = 1 } brook_memory;

/* data points to logical (0,0,0); byte strides may be negative for host input.
 * Zero-initialize and set struct_size/abi_version. Input is borrowed for the call.
 * Device input must be on the context's device and ready before the call. */
typedef struct brook_volume {
  size_t struct_size;
  uint32_t abi_version;
  const void *data;
  int64_t shape[3];
  int64_t strides[3];
  brook_dtype dtype;
  brook_memory memory;
} brook_volume;

typedef struct brook_teasar_parameters {
  double scale, constant, pdrf_scale, pdrf_exponent;
  double soma_detection_threshold, soma_acceptance_threshold;
  double soma_invalidation_scale, soma_invalidation_const;
  int64_t max_paths;
  uint32_t has_max_paths;
} brook_teasar_parameters;

enum brook_skeletonize_flags {
  BROOK_FIX_BRANCHING = 1u, BROOK_FIX_BORDERS = 2u, BROOK_FILTER_OBJECTS = 4u,
  BROOK_DEVICE_OUTPUT = 8u,
  /* With voxel_graph, the holes of each graph component are filled; the graph is used as given. */
  BROOK_FILL_HOLES = 16u,
  /* Uses teasar.soma_detection_threshold for candidate detection. */
  BROOK_FIX_AVOCADOS = 32u
};
typedef struct brook_skeletonize_options {
  size_t struct_size;
  uint32_t abi_version;
  brook_teasar_parameters teasar;
  float anisotropy[3];
  double dust_threshold;
  uint32_t flags;
  const int64_t *object_ids;
  size_t object_ids_count;
  /* Each target is three consecutive (x,y,z) int64 values, in caller order. */
  const int64_t *targets_before;
  size_t targets_before_count;
  const int64_t *targets_after;
  size_t targets_after_count;
  /* Optional edge bitfields, same shape as labels: uint32, or uint8 for XY. */
  const brook_volume *voxel_graph;
} brook_skeletonize_options;

/* Host packed result. Offsets have skeleton_count+1 entries; edge indices are
 * local to each skeleton. All pointers remain valid until the handle is freed. */
typedef struct brook_skeletons_view {
  size_t skeleton_count, vertex_count, edge_count;
  const int64_t *labels, *vertex_offsets, *edge_offsets;
  const float *vertices; /* vertex_count x 3, row major, physical coordinates */
  const uint32_t *edges; /* edge_count x 2, row major */
  const float *radii;
  float anisotropy[3];
} brook_skeletons_view;
typedef struct brook_skeletons_device_view {
  size_t skeleton_count, vertex_count, edge_count;
  const int64_t *labels, *vertex_offsets, *edge_offsets; /* host metadata */
  brook_volume vertices,edges,radii; /* device arrays; row-major vertices/edges */
  float anisotropy[3];
} brook_skeletons_device_view;
typedef enum brook_skeleton_buffer { BROOK_VERTICES=0, BROOK_EDGES=1, BROOK_RADII=2 } brook_skeleton_buffer;

typedef struct brook_cross_section_options {
  size_t struct_size;
  uint32_t abi_version;
  float anisotropy[3];
  int32_t smoothing_window, step;
  uint32_t fill_holes, multipass, repair_contacts, in_place;
} brook_cross_section_options;
typedef struct brook_cross_section_input {
  int64_t label;
  size_t vertex_count, edge_count;
  const float *vertices; /* row-major vertex_count x 3 */
  const uint32_t *edges; /* row-major edge_count x 2 */
  uint32_t physical; /* divide vertices by anisotropy and round to even */
  const float *initial_areas; /* optional vertex_count entries */
  const uint8_t *initial_contacts; /* optional vertex_count entries */
  float *areas; /* caller-owned vertex_count entries */
  uint8_t *contacts; /* caller-owned vertex_count entries */
} brook_cross_section_input;
void brook_cross_section_options_init(brook_cross_section_options *options);
/* Compiled host implementation; no CUDA context/device initialization required.
 * Inputs are borrowed; outputs are written only after the entire call succeeds.
 * Integer host labels support arbitrary byte strides. */
brook_status brook_cross_sectional_area(const brook_volume *labels,
    const brook_cross_section_input *inputs,size_t count,const brook_cross_section_options *options);

typedef enum brook_label_query_kind {
  BROOK_QUERY_UNSIGNED=0, BROOK_QUERY_SIGNED=1, BROOK_QUERY_FLOAT=2, BROOK_QUERY_NO_MATCH=3,
  BROOK_QUERY_WEAK_FLOAT=4 /* cast to the floating volume dtype before comparing */
} brook_label_query_kind;
typedef struct brook_label_query {
  uint64_t integer_label; /* signed values use their two's-complement representation */
  double floating_label;
  double centroid[3]; /* voxel coordinates, not physical coordinates */
  uint32_t kind;
} brook_label_query;
/* A NULL context uses compiled CPU search and requires host input; no CUDA
 * device initialization is needed. With a context, dispatch may use CUDA.
 * Output is count x 3 host int64 coordinates, (-1,-1,-1) for an absent label.
 * Equal Euclidean distances choose the lexicographically first (x,y,z).
 * coordinate_capacity is the number of int64 entries, at least 3*count. */
brook_status brook_nearest_label_voxels(brook_context *context,const brook_volume *labels,
    const brook_label_query *queries,size_t count,int64_t *coordinates,size_t coordinate_capacity);

uint32_t brook_abi_version(void);
const char *brook_version(void);
/* Thread-local error text: valid until the next Brook call on that thread. */
const char *brook_last_error(void);
/* Runs Brook's startup check (GPU, driver, build). A failed check returns BROOK_CUDA_ERROR, with
 * the problem and the fix in brook_last_error(); a negative or out-of-range device returns
 * BROOK_INVALID_ARGUMENT. */
brook_status brook_context_create(int device, brook_context **out);
void brook_context_destroy(brook_context *context);
brook_status brook_context_synchronize(brook_context *context);

/* Result storage is owned by the returned handle; views are borrowed. */
brook_status brook_array_upload(brook_context *context, const brook_volume *input, brook_array **out);
void brook_array_destroy(brook_array *array);
brook_status brook_array_view(const brook_array *array, brook_volume *out);
brook_status brook_array_copy_to_host(const brook_array *array, void *out, size_t bytes);
/* Zero-copy exports retain storage independently of the array/context handles.
 * The consumer must call the returned tensor's deleter exactly once. */
brook_status brook_array_to_dlpack(const brook_array *array, DLManagedTensor **out);
brook_status brook_array_to_dlpack_versioned(const brook_array *array, uint32_t max_minor,
                                           DLManagedTensorVersioned **out);

brook_status brook_connected_components(brook_context *context, const brook_volume *labels,
                                        brook_components **out);
/* Stream host labels through bounded GPU slabs. The returned handle owns host
 * labels and mapping/roots; their views remain valid until handle destruction. */
brook_status brook_connected_components_streamed(brook_context *context,const brook_volume *labels,
                                                 size_t budget_bytes,brook_host_components **out);
void brook_host_components_destroy(brook_host_components *components);
brook_status brook_host_components_view(const brook_host_components *components,brook_volume *labels,
                                        const int64_t **mapping,const int64_t **roots,size_t *count);
void brook_components_destroy(brook_components *components);
brook_status brook_components_view(const brook_components *components, brook_volume *labels,
                                  const int64_t **mapping, const int64_t **roots, size_t *count);
brook_status brook_components_copy_to_host(const brook_components *components, void *out, size_t bytes);
brook_status brook_components_labels(const brook_components *components, brook_array **out);
brook_status brook_edt(brook_context *context, const brook_volume *labels,
                      const float anisotropy[3], int black_border, brook_array **out);
/* Stream host labels through the GPU into caller-owned F-contiguous host output.
 * Input/output must not overlap. output_bytes must cover float32[shape].
 * budget_bytes==0 selects BROOK_STREAM_BUDGET_MB, which must be a positive integer number of
 * MiB, or, when it is unset, a quarter of free GPU memory (at least 64 MiB).
 * The scheduling budget has a minimum of one XY plane and one XZ tile. */
brook_status brook_edt_streamed(brook_context *context,const brook_volume *labels,
    const float anisotropy[3],int black_border,size_t budget_bytes,float *output,size_t output_bytes);
brook_status brook_graph_components(brook_context *context, const brook_volume *labels,
                                   const brook_volume *voxel_graph, brook_components **out);
brook_status brook_graph_edt(brook_context *context, const brook_volume *labels,
                            const brook_volume *voxel_graph, const float anisotropy[3],
                            int black_border, brook_array **out);

/* Initialize to the Python skeletonize() defaults before overriding fields. */
void brook_skeletonize_options_init(brook_skeletonize_options *options);
brook_status brook_skeletonize(brook_context *context, const brook_volume *labels,
                              const brook_skeletonize_options *options, brook_skeletons **out);
/* Host-resident streamed preamble and per-component GPU tracing. The scheduling
 * budget controls slabs/tiles; individual component workspaces must fit on GPU.
 * voxel_graph, fill_holes and fix_avocados are unsupported in this mode. */
brook_status brook_skeletonize_streamed(brook_context *context,const brook_volume *labels,
    const brook_skeletonize_options *options,size_t budget_bytes,brook_skeletons **out);
/* One unlabeled path (label 0), ordered from end to start as in Kimimaro's point_to_point,
 * with vertices in physical coordinates. Nonzero input values are foreground. As in
 * kimimaro.connect_points, which copies its mask to Fortran order, the result does not
 * depend on the host strides; device input follows the normal Fortran-contiguous volume contract. */
brook_status brook_connect_points(brook_context *context,const brook_volume *labels,
                                 const int64_t start[3],const int64_t end[3],const float anisotropy[3],
                                 double pdrf_scale,double pdrf_exponent,brook_skeletons **out);
enum brook_oversegment_flags {
  BROOK_OVERSEGMENT_BINARY = 1u, BROOK_OVERSEGMENT_FILL_HOLES = 2u,
  BROOK_OVERSEGMENT_IN_PLACE = 4u, BROOK_OVERSEGMENT_FLOAT32_COORDINATES = 8u
};
/* Kimimaro-compatible oversegmentation with exact feature ownership; with FILL_HOLES, the
 * fill of a label touching a volume face can differ (docs/USAGE.md). Host labels and
 * packed host skeleton input.
 * Features are uint64, Fortran order; segments follow packed vertex order.
 * The caller supplies capacities in elements. Optional processed has one byte
 * per skeleton (whether its segments property needs to be registered).
 * IN_PLACE renumbers writable label storage; its descriptor is still borrowed.
 * A null context is supported unless FILL_HOLES is requested. */
brook_status brook_oversegment(brook_context *context,const brook_volume *labels,
    const brook_skeletons_view *skeletons,const double anisotropy[3],int downsample,uint32_t flags,
    uint64_t *features,size_t feature_capacity,uint64_t *segments,size_t segment_capacity,
    uint8_t *processed,size_t processed_capacity);
brook_status brook_skeletons_get_view(const brook_skeletons *result, brook_skeletons_view *out);
/* Device-output handles create a stable host snapshot on the first get_view(). */
brook_status brook_skeletons_get_device_view(const brook_skeletons *result, brook_skeletons_device_view *out);
brook_status brook_skeletons_get_array(const brook_skeletons *result, brook_skeleton_buffer buffer, brook_array **out);
/* Both operations return device-output handles. Origins are count x 3 float32,
 * or NULL for zero offsets. Canonical postprocessing uses Brook's device tie rules. */
brook_status brook_merge_skeletons(brook_context *context, const brook_skeletons *const *fragments,
                                  const float *origins, size_t count, brook_skeletons **out);
brook_status brook_postprocess_canonical(brook_context *context, const brook_skeletons *input,
                                        double dust_threshold, double tick_threshold, brook_skeletons **out);
/* Kimimaro-compatible postprocessing and joining, computed on the CPU from borrowed host
 * views; no CUDA context required. Input coordinates must be finite. Edges are local to each
 * skeleton. Postprocess preserves label order. Join treats all input skeletons as fragments.
 * Thresholds compare in double precision. Join radius follows Kimimaro's float32
 * distance-matrix comparisons. */
brook_status brook_postprocess_cpu(const brook_skeletons_view *input,double dust_threshold,
                                   double tick_threshold,brook_skeletons **out);
brook_status brook_join_cpu(const brook_skeletons_view *input,double radius,
                            int restrict_by_radius,brook_skeletons **out);
/* Outputs of brook_postprocess_cpu and brook_join_cpu retain source vertex indices into the
 * packed input. Borrowed until result destruction; unavailable for other result types. */
brook_status brook_skeletons_get_sources(const brook_skeletons *result,const uint64_t **sources,size_t *count);
void brook_skeletons_destroy(brook_skeletons *result);

#ifdef __cplusplus
}
#endif
#endif
