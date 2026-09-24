// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include "trace_primitives.hpp"
#include "skeleton.hpp"
#include "point_set.hpp"
#include "batched.hpp"
#include "voxel_graph.hpp"
#include "dlpack.hpp"
#include "packed.hpp"
#include "batch.hpp"
#include "post.hpp"
#include "nearest.hpp"
#include "draft_plan.hpp"
#include "integer_set.hpp"
#include "point_path.hpp"
#include "streaming.hpp"
#include "oversegment_bindings.hpp"
#include "cpu_post_bindings.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <set>

namespace py=pybind11;
namespace {
// CUDA destruction may synchronize with another thread's work. Never retain the
// Python GIL while that destruction waits for the other thread to make progress.
template<class T> struct GpuDeleter {
  void operator()(T *value) const noexcept {
    if(Py_IsInitialized() && PyGILState_Check()) {
      py::gil_scoped_release unlock;delete value;
    } else delete value;
  }
};
void same_device(const brook::Context &ctx,const brook::Array &a) {
  if(ctx.device!=a.device) throw py::value_error("array is on a different CUDA device");
  if(a.logical_shape) throw py::value_error("volume kernels require an array created with Context.upload()");
}
void same_device(const brook::Context &ctx,const brook::Array *a) { if(a) same_device(ctx,*a); }
void same_device(const brook::Context &ctx,const brook::TraceWorkspace &ws) {
  if(ctx.device!=ws.key.storage()->device) throw py::value_error("workspace is on a different CUDA device");
}
template<class T> void same_device(const brook::Context&,const T&) {}
template<class R,class... Args> auto guarded(R (*fn)(brook::Context&,Args...)) {
  return [fn](brook::Context &ctx,Args... args)->R {
    brook::ContextCall call(ctx);(same_device(ctx,args),...);
    return fn(ctx,std::forward<Args>(args)...);
  };
}
template<class T> struct TensorDeleter {
  void operator()(T *tensor) const noexcept { if(tensor && tensor->deleter) tensor->deleter(tensor); }
};
void destroy_capsule(PyObject *capsule) noexcept {
  if(PyCapsule_IsValid(capsule,"dltensor")) {
    auto tensor=static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule,"dltensor"));TensorDeleter<DLManagedTensor>{}(tensor);
  } else if(PyCapsule_IsValid(capsule,"dltensor_versioned")) {
    auto tensor=static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(capsule,"dltensor_versioned"));TensorDeleter<DLManagedTensorVersioned>{}(tensor);
  }
}
py::capsule dlpack(const brook::Array &array,py::object stream,py::object max_version,py::object dl_device,py::object copy) {
  DLDevice target{kDLCUDA,array.device};
  if(!dl_device.is_none()) {
    auto device=py::cast<std::pair<int,int>>(dl_device);
    if(device.first!=kDLCPU && device.first!=kDLCUDA) throw py::buffer_error("unsupported DLPack device type");
    target={static_cast<DLDeviceType>(device.first),device.second};
  }
  bool transfer=target.device_type!=kDLCUDA || target.device_id!=array.device;
  if(transfer && !copy.is_none() && !py::cast<bool>(copy)) throw py::buffer_error("requested device requires a copy");
  bool copied=transfer || (!copy.is_none() && py::cast<bool>(copy));
  if(!stream.is_none()) {
    int64_t value=py::cast<int64_t>(stream);
    if(target.device_type==kDLCPU || value==0 || value < -1) throw py::value_error("invalid DLPack stream");
  }
  bool versioned=false;uint32_t minor=0;
  if(!max_version.is_none()) {
    auto version=py::cast<std::pair<uint32_t,uint32_t>>(max_version);
    versioned=version.first>=1;minor=version.first>1?DLPACK_MINOR_VERSION:version.second;
  }
  // Every public operation completes its producer stream before returning.
  // Copies below also complete before the capsule can reach a consumer stream.
  if(versioned) {
    std::unique_ptr<DLManagedTensorVersioned,TensorDeleter<DLManagedTensorVersioned>> tensor;
    { py::gil_scoped_release unlock;tensor.reset(brook::export_dlpack_versioned(array,target,copied,minor)); }
    py::capsule capsule(tensor.get(),"dltensor_versioned",destroy_capsule);tensor.release();return capsule;
  }
  std::unique_ptr<DLManagedTensor,TensorDeleter<DLManagedTensor>> tensor;
  { py::gil_scoped_release unlock;tensor.reset(brook::export_dlpack(array,target,copied)); }
  py::capsule capsule(tensor.get(),"dltensor",destroy_capsule);tensor.release();return capsule;
}
py::dtype numpy_dtype(brook_dtype d) {
  static const char *names[]={"uint8","uint16","uint32","uint64","int8","int16","int32","int64","float32","float64"};
  return py::dtype(names[static_cast<unsigned>(d)]);
}
brook_dtype native_dtype(const py::dtype &d) {
  if(!py::cast<bool>(d.attr("isnative"))) throw py::value_error("non-native byte order is not supported");
  if(d.equal(py::dtype::of<bool>())) return BROOK_U8;
  for(unsigned i=0;i<10;++i) if(d.equal(numpy_dtype(static_cast<brook_dtype>(i)))) return static_cast<brook_dtype>(i);
  throw py::type_error("unsupported array dtype");
}
// A device-resident array described by the CUDA array interface (CuPy, PyTorch, Brook's own
// DeviceArray, ...) copied device-to-device into Brook's layout; a C-contiguous source is
// transposed on the GPU. The producer's work is fenced with a device synchronization because
// Brook's stream does not synchronize with the legacy default stream.
// The producer's pending work on its stream must complete before Brook reads the memory. Brook's
// stream does not synchronize with the legacy default stream, so an event on the producer's stream
// (CUDA array interface v3 `stream`, or an explicit handle: 1 legacy, 2 per-thread, else a
// cudaStream_t) is what Brook's stream waits for. No handle: the legacy default stream.
// (Resolved to a plain handle while the GIL is held; -1 stands for "none".)
int64_t stream_handle(py::object stream) { return stream.is_none()?-1:py::cast<int64_t>(stream); }
void fence_producer(brook::Context &ctx,int64_t handle) {
  cudaStream_t producer=nullptr;
  if(handle==2) producer=cudaStreamPerThread;
  else if(handle>2) producer=reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(handle));
  cudaEvent_t event=nullptr;
  BROOK_CUDA(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
  BROOK_CUDA(cudaEventRecord(event,producer));
  BROOK_CUDA(cudaStreamWaitEvent(ctx.stream,event,0));
  BROOK_CUDA(cudaEventDestroy(event));
}
brook::Array upload_cuda_array(brook::Context &ctx,py::object obj,py::object stream,bool borrow) {
  if(!py::hasattr(obj,"__cuda_array_interface__")) throw py::type_error("object does not expose __cuda_array_interface__");
  py::dict cai=obj.attr("__cuda_array_interface__");
  if(stream.is_none() && cai.contains("stream")) stream=cai["stream"];
  const int64_t handle=stream_handle(stream);
  auto shape=py::cast<std::vector<int64_t>>(cai["shape"]);
  py::dtype dtype(py::cast<std::string>(cai["typestr"]));
  brook_volume v{};v.struct_size=sizeof(v);v.abi_version=BROOK_ABI_VERSION;
  v.dtype=native_dtype(dtype);v.memory=BROOK_DEVICE;
  v.data=reinterpret_cast<const void*>(py::cast<uintptr_t>(py::tuple(cai["data"])[0]));
  const int64_t item=static_cast<int64_t>(dtype.itemsize());
  std::vector<int64_t> strides(shape.size());
  if(cai.contains("strides") && !cai["strides"].is_none()) strides=py::cast<std::vector<int64_t>>(cai["strides"]);
  else { int64_t acc=item;for(int i=int(shape.size())-1;i>=0;--i) { strides[i]=acc;acc*=std::max<int64_t>(shape[i],1); } }
  int ndim=int(shape.size());
  while(ndim>3 && shape[ndim-1]==1) --ndim;
  if(ndim>3) throw py::value_error("at most three non-singleton volume dimensions are supported");
  for(int i=0;i<3;++i) { v.shape[i]=i<ndim?shape[i]:1;v.strides[i]=i<ndim?strides[i]:item; }
  bool fortran=true;int64_t stride=item;
  for(int i=0;i<3;++i) { if(v.shape[i]>1 && v.strides[i]!=stride) fortran=false;stride*=v.shape[i]; }
  if(borrow && fortran) {
    // Zero copy: Brook reads the producer's memory in place and keeps the producer alive for as
    // long as any array of this call refers to it. The release runs under the GIL.
    cudaPointerAttributes attributes{};
    BROOK_CUDA(cudaPointerGetAttributes(&attributes,v.data));
    if(attributes.device==ctx.device && attributes.type==cudaMemoryTypeDevice) {
      // `keep` owns one reference to the producer; dropping it under the GIL is the release
      auto keep=std::make_shared<py::object>(obj);
      auto allocation=std::make_shared<brook::Allocation>(const_cast<void*>(v.data),size_t(brook::volume_size({v.shape[0],v.shape[1],v.shape[2]}))*item,ctx.device,
        [keep]() mutable { py::gil_scoped_acquire gil;*keep=py::object(); });
      brook::Array out;out.storage=allocation;out.shape={v.shape[0],v.shape[1],v.shape[2]};out.dtype=v.dtype;out.device=ctx.device;
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      fence_producer(ctx,handle);
      return out;
    }
  }
  py::gil_scoped_release unlock;brook::ContextCall call(ctx);
  fence_producer(ctx,handle);
  auto out=brook::upload(ctx,v);ctx.synchronize();return out;
}
brook::TraceParameters trace_parameters(const py::dict &values);
brook::SkeletonizeOptions skeletonize_options(py::dict params,std::array<float,3> aniso,double dust,bool branching,bool borders,
    std::optional<std::vector<int64_t>> ids,std::vector<brook::Point> before,std::vector<brook::Point> after,bool fill_holes,bool fix_avocados,
    std::optional<bool> black_border) {
  brook::SkeletonizeOptions options;
  options.teasar=trace_parameters(params);options.anisotropy=aniso;options.dust=dust;
  options.fix_branching=branching;options.fix_borders=borders;options.fill_holes=fill_holes;options.fix_avocados=fix_avocados;
  options.black_border_override=black_border;
  options.avocado_detection=params.contains("soma_detection_threshold")?py::cast<double>(params["soma_detection_threshold"]):0;
  options.object_ids=std::move(ids);
  options.targets_before=std::move(before);options.targets_after=std::move(after);
  return options;
}
brook_volume volume(const py::array &a) {
  brook_volume v{}; v.struct_size=sizeof(v); v.abi_version=BROOK_ABI_VERSION;
  v.dtype=native_dtype(a.dtype()); v.data=a.data(); v.memory=BROOK_HOST;
  int ndim=static_cast<int>(a.ndim());
  while(ndim>3 && a.shape(ndim-1)==1) --ndim;
  if(ndim>3) throw py::value_error("at most three non-singleton volume dimensions are supported");
  for(int i=0;i<3;++i) { v.shape[i]=i<ndim?a.shape(i):1; v.strides[i]=i<ndim?a.strides(i):a.itemsize(); }
  return v;
}
py::array download(const brook::Array &a) {
  auto dimensions=a.tensor_shape(),pitches=a.tensor_strides();
  std::vector<py::ssize_t> shape(dimensions.begin(),dimensions.end()),strides(pitches.begin(),pitches.end());
  py::array out(numpy_dtype(a.dtype),shape,strides);
  { py::gil_scoped_release unlock; a.copy_to_host(out.mutable_data(),a.bytes()); }
  return out;
}
py::array_t<int64_t> integers(const std::vector<int64_t> &values) {
  py::array_t<int64_t> out(values.size());
  if(!values.empty()) std::memcpy(out.mutable_data(),values.data(),values.size()*sizeof(int64_t));
  return out;
}
brook_label_query query_label(py::handle label,brook_dtype dtype) {
  brook_label_query q{};q.kind=BROOK_QUERY_NO_MATCH;
  if(!PyNumber_Check(label.ptr())) return q;
  bool strong=py::hasattr(label,"dtype");
  if(strong && py::cast<std::string>(py::reinterpret_borrow<py::object>(label).attr("dtype").attr("kind"))=="b") {
    q.kind=BROOK_QUERY_UNSIGNED;q.integer_label=py::cast<bool>(label);return q;
  }
  if(dtype==BROOK_F32 || dtype==BROOK_F64 || !PyIndex_Check(label.ptr())) {
    q.kind=strong?BROOK_QUERY_FLOAT:BROOK_QUERY_WEAK_FLOAT;q.floating_label=py::cast<double>(label);return q;
  }
  auto integer=py::reinterpret_steal<py::object>(PyNumber_Index(label.ptr()));
  if(!integer) throw py::error_already_set();
  int overflow=0;long long signed_value=PyLong_AsLongLongAndOverflow(integer.ptr(),&overflow);
  if(PyErr_Occurred()) throw py::error_already_set();
  if(!overflow) { q.kind=BROOK_QUERY_SIGNED;q.integer_label=uint64_t(signed_value); }
  else if(overflow>0) {
    q.integer_label=PyLong_AsUnsignedLongLong(integer.ptr());
    if(PyErr_Occurred()) PyErr_Clear();else q.kind=BROOK_QUERY_UNSIGNED;
  }
  return q;
}
py::dict synapse_targets(brook::Context *ctx,py::array labels,py::dict synapses,bool) {
  int dimensions=std::min<int>(labels.ndim(),3);
  if(!dimensions) throw py::value_error("synapse labels require at least one dimension");
  for(int a=3;a<labels.ndim();++a) if(!labels.shape(a)) throw py::index_error("cannot select index zero from an empty trailing axis");
  brook_volume view{};view.struct_size=sizeof(view);view.abi_version=BROOK_ABI_VERSION;
  view.dtype=native_dtype(labels.dtype());view.memory=BROOK_HOST;view.data=labels.data();
  for(int a=0;a<3;++a) { view.shape[a]=a<dimensions?labels.shape(a):1;view.strides[a]=a<dimensions?labels.strides(a):labels.itemsize(); }
  std::vector<brook_label_query> queries;
  std::vector<std::pair<py::object,std::vector<size_t>>> groups;
  for(auto label:synapses) {
    if(labels.dtype().equal(py::dtype::of<bool>()) && PyLong_Check(label.first.ptr())) {
      PyLong_AsLongLong(label.first.ptr());if(PyErr_Occurred()) throw py::error_already_set();
    }
    auto pattern=query_label(label.first,view.dtype);py::dict by_tag;
    for(auto entry:py::reinterpret_borrow<py::iterable>(label.second)) {
      auto pair=py::cast<py::sequence>(entry);
      if(pair.size()!=2) throw py::value_error("synapse entries must contain a centroid and an SWC label");
      auto centroid=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(pair[0]);
      if(!centroid || centroid.ndim()!=1 || centroid.size()!=dimensions) throw py::value_error("centroid dimensions do not match labels");
      auto q=pattern;for(int a=0;a<dimensions;++a) q.centroid[a]=centroid.data()[a];
      py::object tag=pair[1];
      if(!by_tag.contains(tag)) by_tag[tag]=py::list();
      py::cast<py::list>(by_tag[tag]).append(queries.size());queries.push_back(q);
    }
    for(auto item:by_tag) groups.emplace_back(py::reinterpret_borrow<py::object>(item.first),py::cast<std::vector<size_t>>(item.second));
  }
  std::vector<std::array<int64_t,3>> closest;
  if(!queries.empty()) {
    py::gil_scoped_release unlock;
    if(ctx) { brook::ContextCall call(*ctx);closest=brook::nearest_label_voxels(*ctx,view,queries); }
    else closest=brook::nearest_label_voxels_host(view,queries);
  }
  py::dict result;
  auto coordinate_type=py::module_::import("numpy").attr("intp");
  for(const auto &group:groups) {
    std::set<std::array<int64_t,3>> points;
    for(size_t i:group.second) if(closest[i][0]>=0) points.insert(closest[i]);
    for(const auto &p:points) {
      py::tuple coordinate(dimensions);for(int a=0;a<dimensions;++a) coordinate[a]=coordinate_type(p[a]);
      result[coordinate]=group.first;
    }
  }
  return result;
}
brook::TraceParameters trace_parameters(const py::dict &values) {
  brook::TraceParameters p;
  for(auto entry:values) {
    auto key=py::cast<std::string>(entry.first);
    if(key=="max_paths") { if(!entry.second.is_none()) p.max_paths=py::cast<int64_t>(entry.second);continue; }
    double v=py::cast<double>(entry.second);
    if(key=="scale") p.scale=v;
    else if(key=="const") p.constant=v;
    else if(key=="pdrf_scale") p.pdrf_scale=v;
    else if(key=="pdrf_exponent") p.pdrf_exponent=v;
    else if(key=="soma_detection_threshold") p.soma_detection=v;
    else if(key=="soma_acceptance_threshold") p.soma_acceptance=v;
    else if(key=="soma_invalidation_scale") p.soma_scale=v;
    else if(key=="soma_invalidation_const") p.soma_constant=v;
    else throw py::type_error("unexpected TEASAR parameter: "+key);
  }
  return p;
}
}
void bind_cross_sections(py::module_ &m);
void bind_runtime(py::module_ &m);
void bind_intake(py::module_ &m);
void bind_public_primitives(py::module_ &m);
void bind_scheduling(py::module_ &m);
PYBIND11_MODULE(_core,m) {
  m.def("_oversegment_host",[](py::array labels,py::object skeletons,py::object anisotropy,bool progress,bool fill_holes,bool in_place,int downsample) {
    return oversegment_python(nullptr,labels,skeletons,anisotropy,progress,fill_holes,in_place,downsample);
  },py::arg("labels"),py::arg("skeletons"),py::arg("anisotropy")=py::none(),py::arg("progress")=false,
    py::arg("fill_holes")=false,py::arg("in_place")=false,py::arg("downsample")=0);

  bind_cross_sections(m);
  bind_scheduling(m);
  m.doc()="Brook CUDA extension";
  brook::python::bind_cpu_post(m);
  py::register_exception<brook::CudaError>(m,"CudaError",PyExc_RuntimeError);
  py::class_<brook::Skeleton>(m,"Skeleton")
    .def_readonly("id",&brook::Skeleton::label)
    .def_property_readonly("vertices",[](brook::Skeleton &s) {
      return py::array(py::dtype::of<float>(),{py::ssize_t(s.vertices.size()),py::ssize_t(3)},
        {py::ssize_t(12),py::ssize_t(4)},s.vertices.data(),py::cast(&s,py::return_value_policy::reference));
    })
    .def_property_readonly("edges",[](brook::Skeleton &s) {
      return py::array(py::dtype::of<uint32_t>(),{py::ssize_t(s.edges.size()),py::ssize_t(2)},
        {py::ssize_t(8),py::ssize_t(4)},s.edges.data(),py::cast(&s,py::return_value_policy::reference));
    })
    .def_property_readonly("radii",[](brook::Skeleton &s) {
      return py::array(py::dtype::of<float>(),{py::ssize_t(s.radii.size())},{py::ssize_t(4)},s.radii.data(),
        py::cast(&s,py::return_value_policy::reference));
    });
  py::class_<brook::Array,std::unique_ptr<brook::Array,GpuDeleter<brook::Array>>>(m,"DeviceArray")
    .def_property_readonly("shape",[](const brook::Array&a){return py::tuple(py::cast(a.tensor_shape()));})
    .def_property_readonly("strides",[](const brook::Array&a){return py::tuple(py::cast(a.tensor_strides()));})
    .def_property_readonly("ndim",[](const brook::Array&a){return a.tensor_shape().size();})
    .def_property_readonly("dtype",[](const brook::Array&a){return numpy_dtype(a.dtype);})
    .def_property_readonly("nbytes",&brook::Array::bytes)
    .def_property_readonly("size",&brook::Array::size)
    .def_property_readonly("device",[](const brook::Array&a){return a.device;})
    .def("to_host",&download)
    .def("get",&download)
    .def("__dlpack_device__",[](const brook::Array &a){return py::make_tuple(int(kDLCUDA),a.device);})
    .def("__dlpack__",&dlpack,py::arg("stream")=py::none(),py::kw_only(),py::arg("max_version")=py::none(),
         py::arg("dl_device")=py::none(),py::arg("copy")=py::none())
    .def_property_readonly("__cuda_array_interface__",[](const brook::Array&a) {
      py::dict d;
      d["shape"]=py::tuple(py::cast(a.tensor_shape()));
      d["strides"]=py::tuple(py::cast(a.tensor_strides()));
      d["typestr"]=numpy_dtype(a.dtype).attr("str");
      d["data"]=py::make_tuple(reinterpret_cast<uintptr_t>(a.data()),false);
      d["version"]=3; d["stream"]=py::none(); return d;
    });
  py::class_<brook::PackedSkeletons,std::unique_ptr<brook::PackedSkeletons,GpuDeleter<brook::PackedSkeletons>>>(m,"PackedSkeletons")
    .def_property_readonly("labels",[](const brook::PackedSkeletons &p){return integers(p.labels);})
    .def_property_readonly("v_off",[](const brook::PackedSkeletons &p){return integers(p.vertex_offsets);})
    .def_property_readonly("e_off",[](const brook::PackedSkeletons &p){return integers(p.edge_offsets);})
    .def_property_readonly("vertices",[](const brook::PackedSkeletons &p){return p.vertices;})
    .def_property_readonly("edges",[](const brook::PackedSkeletons &p){return p.edges;})
    .def_property_readonly("radii",[](const brook::PackedSkeletons &p){return p.radii;})
    .def_readonly("anisotropy",&brook::PackedSkeletons::anisotropy)
    .def("to_skeletons",&brook::unpack_skeletons,py::call_guard<py::gil_scoped_release>());
  py::class_<brook::Context,std::unique_ptr<brook::Context,GpuDeleter<brook::Context>>>(m,"Context")
    .def(py::init<int>(),py::arg("device")=0,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("device",[](const brook::Context &ctx){return ctx.device;})
    .def_property_readonly("stats",[](const brook::Context &ctx) {
      std::map<std::string,int64_t> result;
      { py::gil_scoped_release unlock;std::lock_guard lock(ctx.call_mutex);result=ctx.stats; }
      return result;
    })
    .def("synchronize",[](brook::Context &ctx){brook::ContextCall call(ctx);ctx.synchronize();},py::call_guard<py::gil_scoped_release>())
    .def("synapses_to_targets",[](brook::Context &ctx,py::array labels,py::dict synapses,bool progress) {
      return synapse_targets(&ctx,labels,synapses,progress);
    },py::arg("labels"),py::arg("synapses"),py::arg("progress")=false)
    .def("oversegment",&oversegment_python,py::arg("labels"),py::arg("skeletons"),py::arg("anisotropy")=py::none(),
      py::arg("progress")=false,py::arg("fill_holes")=false,py::arg("in_place")=false,py::arg("downsample")=0)
    .def("connect_points",[](brook::Context &ctx,py::array labels,brook::Point start,brook::Point end,
                              std::array<float,3> spacing,double scale,double exponent) {
      // Kimimaro's astype(bool) foreground; it then copies the mask to Fortran order, so the
      // input layout does not change the result.
      py::array mask=labels.attr("astype")(py::dtype::of<bool>());auto input=volume(mask);
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      return brook::connect_points(ctx,brook::upload(ctx,input),start,end,spacing,scale,exponent);
    },py::arg("labels"),py::arg("start"),py::arg("end"),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("pdrf_scale")=100000,py::arg("pdrf_exponent")=4)
    .def("edt_streamed",[](brook::Context &ctx,py::array labels,std::array<float,3> spacing,bool bb,size_t budget) {
      auto input=volume(labels);
      std::vector<py::ssize_t> shape(input.shape,input.shape+3);
      std::vector<py::ssize_t> strides={4,4*input.shape[0],4*input.shape[0]*input.shape[1]};
      py::array result(py::dtype::of<float>(),shape,strides);
      auto *out=static_cast<float*>(result.mutable_data());size_t capacity=size_t(result.size());
      {py::gil_scoped_release unlock;brook::ContextCall call(ctx);
        brook::edt_streamed(ctx,input,spacing,bb,budget,out,capacity);}
      return result;
    },py::arg("labels"),py::arg("anisotropy")=std::array<float,3>{1,1,1},py::arg("black_border")=false,py::arg("budget_bytes")=0)
    .def("upload",[](brook::Context &ctx,py::array a) {
      auto v=volume(a); py::gil_scoped_release unlock;
      brook::ContextCall call(ctx);
      ctx.activate(); auto out=brook::upload(ctx,v); ctx.synchronize(); return out;
    })
    .def("upload_tensor",[](brook::Context &ctx,py::array input) {
      if(input.ndim()>4) throw py::value_error("tensor rank exceeds four");   // rank 4: a (B, Z, Y, X) batch
      auto dtype=native_dtype(input.dtype());
      auto a=py::array::ensure(input,py::array::c_style);
      if(!a) throw py::value_error("cannot make a contiguous tensor");
      std::vector<int64_t> shape(a.shape(),a.shape()+a.ndim()),strides(a.strides(),a.strides()+a.ndim());
      brook_volume v{};v.struct_size=sizeof(v);v.abi_version=BROOK_ABI_VERSION;v.dtype=dtype;v.memory=BROOK_HOST;v.data=a.data();
      v.shape[0]=a.size();v.shape[1]=v.shape[2]=1;
      v.strides[0]=a.itemsize();v.strides[1]=v.strides[2]=a.nbytes();
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      auto out=brook::upload(ctx,v);ctx.synchronize();return brook::tensor_view(std::move(out),std::move(shape),std::move(strides));
    })
    .def("pack",[](brook::Context &ctx,std::vector<int64_t> labels,std::vector<int64_t> voff,std::vector<int64_t> eoff,
                   py::array_t<float,py::array::c_style|py::array::forcecast> vertices,
                   py::array_t<uint32_t,py::array::c_style|py::array::forcecast> edges,
                   py::array_t<float,py::array::c_style|py::array::forcecast> radii,std::array<float,3> aniso) {
      if(vertices.ndim()!=2 || vertices.shape(1)!=3 || edges.ndim()!=2 || edges.shape(1)!=2 || radii.ndim()!=1 || radii.size()!=vertices.shape(0) ||
         voff.size()!=labels.size()+1 || eoff.size()!=labels.size()+1 || voff.front()!=0 || eoff.front()!=0 ||
         voff.back()!=vertices.shape(0) || eoff.back()!=edges.shape(0) || !std::is_sorted(voff.begin(),voff.end()) || !std::is_sorted(eoff.begin(),eoff.end()))
        throw py::value_error("invalid packed skeleton arrays or offsets");
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      brook::PackedSkeletons p;p.labels=std::move(labels);p.vertex_offsets=std::move(voff);p.edge_offsets=std::move(eoff);p.anisotropy=aniso;
      auto v=brook::allocate(ctx,{3,vertices.shape(0),1},BROOK_F32),e=brook::allocate(ctx,{2,edges.shape(0),1},BROOK_U32);
      auto r=brook::allocate(ctx,{radii.size(),1,1},BROOK_F32);
      if(v.bytes()) BROOK_CUDA(cudaMemcpyAsync(v.data(),vertices.data(),v.bytes(),cudaMemcpyHostToDevice,ctx.stream));
      if(e.bytes()) BROOK_CUDA(cudaMemcpyAsync(e.data(),edges.data(),e.bytes(),cudaMemcpyHostToDevice,ctx.stream));
      if(r.bytes()) BROOK_CUDA(cudaMemcpyAsync(r.data(),radii.data(),r.bytes(),cudaMemcpyHostToDevice,ctx.stream));
      ctx.synchronize();p.vertices=brook::row_view(v,0,vertices.shape(0),3);p.edges=brook::row_view(e,0,edges.shape(0),2);
      p.radii=brook::row_view(r,0,radii.size(),1);return p;
    },py::arg("labels"),py::arg("v_off"),py::arg("e_off"),py::arg("vertices"),py::arg("edges"),py::arg("radii"),
      py::arg("anisotropy")=std::array<float,3>{1,1,1})
    .def("merge_fragments",guarded(&brook::merge_fragments),py::arg("fragments"),py::arg("origins"),py::call_guard<py::gil_scoped_release>())
    .def("skeleton_components",guarded(&brook::skeleton_components),py::arg("packed"),py::call_guard<py::gil_scoped_release>())
    .def("remove_dust",guarded(&brook::remove_dust),py::arg("packed"),py::arg("threshold"),py::call_guard<py::gil_scoped_release>())
    .def("remove_loops",guarded(&brook::remove_loops),py::arg("packed"),py::call_guard<py::gil_scoped_release>())
    .def("remove_ticks",guarded(&brook::remove_ticks),py::arg("packed"),py::arg("threshold"),py::call_guard<py::gil_scoped_release>())
    .def("join_components",guarded(&brook::join_components),py::arg("packed"),py::call_guard<py::gil_scoped_release>())
    .def("postprocess",guarded(&brook::postprocess),py::arg("packed"),py::arg("dust_threshold")=1500,
         py::arg("tick_threshold")=3500,py::call_guard<py::gil_scoped_release>())
    .def("prune_unused",guarded(&brook::prune_unused),py::arg("packed"),py::call_guard<py::gil_scoped_release>())
    .def("connected_components_streamed",[](brook::Context &ctx,py::array labels,size_t budget) {
      auto input=volume(labels);auto result=std::make_unique<brook::HostComponents>();
      {py::gil_scoped_release unlock;brook::ContextCall call(ctx);
        *result=brook::connected_components_streamed(ctx,input,budget);}
      auto *p=result.get();auto view=p->view();
      py::capsule owner(p,[](void *v){delete static_cast<brook::HostComponents*>(v);});result.release();
      std::vector<py::ssize_t> shape(view.shape,view.shape+3),strides(view.strides,view.strides+3);
      py::array array(numpy_dtype(view.dtype),shape,strides,view.data,owner);
      return py::make_tuple(std::move(array),integers(p->mapping),integers(p->roots));
    },py::arg("labels"),py::arg("budget_bytes")=0)
    .def("connected_components",[](brook::Context &ctx,const brook::Array &a) {
      brook::Components c;
      { py::gil_scoped_release unlock;brook::ContextCall call(ctx);same_device(ctx,a);c=brook::connected_components(ctx,a); }
      return py::make_tuple(std::move(c.labels),integers(c.mapping),integers(c.roots));
    })
    .def("graph_components",[](brook::Context &ctx,const brook::Array &a,const brook::Array &graph,bool capture,bool return_keys) -> py::tuple {
      brook::Components c;
      { py::gil_scoped_release unlock;brook::ContextCall call(ctx);same_device(ctx,a);same_device(ctx,graph);c=brook::graph_components(ctx,a,graph,capture); }
      if(return_keys) return py::make_tuple(std::move(c.labels),integers(c.mapping),integers(c.roots),integers(c.keys));
      return py::make_tuple(std::move(c.labels),integers(c.mapping),integers(c.roots));
    },py::arg("labels"),py::arg("graph"),py::arg("capture")=false,py::arg("return_keys")=false)
    .def("graph_edt",guarded(&brook::graph_edt),py::arg("labels"),py::arg("graph"),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("black_border")=false,py::arg("capture")=false,py::arg("fused")=false,py::arg("compact")=true,py::arg("segment")=0,
      py::call_guard<py::gil_scoped_release>())
    .def("edt",guarded(&brook::edt),py::arg("labels"),py::arg("anisotropy")=std::array<float,3>{1,1,1},
         py::arg("black_border")=false,py::arg("dimensions")=3,py::arg("segment")=0,py::call_guard<py::gil_scoped_release>())
    .def("euclidean_distance_field",[](brook::Context &ctx,const brook::Array &a,int64_t source,
                                      std::array<float,3> aniso,float radius,const brook::Array *graph,bool reference_weights) {
      brook::DistanceField result;
      { py::gil_scoped_release unlock;brook::ContextCall call(ctx);same_device(ctx,a);same_device(ctx,graph);result=brook::euclidean_distance_field(ctx,a,source,aniso,radius,graph,reference_weights); }
      return py::make_tuple(std::move(result.distance),result.maximum);
    },py::arg("mask"),py::arg("source"),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("free_radius")=0,py::arg("voxel_graph")=nullptr,py::arg("reference_weights")=false)
    .def("parental_field",guarded(&brook::parental_field),py::arg("field"),py::arg("source"),
         py::arg("rail_terminal")=false,py::arg("voxel_graph")=nullptr,py::call_guard<py::gil_scoped_release>())
    .def("parental_field_banded",guarded(&brook::parental_field_banded),py::arg("mask"),py::arg("field"),py::arg("source"),
         py::call_guard<py::gil_scoped_release>())
    .def("pdrf",guarded(&brook::pdrf),py::arg("dbf"),py::arg("daf"),py::arg("dbf_max"),py::arg("max_daf"),
         py::arg("scale"),py::arg("exponent"),py::call_guard<py::gil_scoped_release>())
    .def("fill_voids",guarded(&brook::fill_voids),py::call_guard<py::gil_scoped_release>())
    .def("backtrace",guarded(&brook::backtrace),py::arg("parents"),py::arg("target"),py::arg("source"),py::call_guard<py::gil_scoped_release>())
    .def("railroad",guarded(&brook::railroad),py::arg("field"),py::arg("target"),py::arg("workspace"),
         py::arg("voxel_graph")=nullptr,py::call_guard<py::gil_scoped_release>())
    .def("invalidate",guarded(&brook::invalidate),py::arg("mask"),py::arg("dbf"),py::arg("scale"),py::arg("const"),
         py::arg("anisotropy"),py::arg("path"),py::arg("workspace"),py::arg("voxel_graph")=nullptr,
         py::call_guard<py::gil_scoped_release>())
    .def("trace",[](brook::Context &ctx,brook::Array mask,brook::Array dbf,py::dict params,
                    std::array<float,3> aniso,bool branching,std::vector<brook::Point> before,
                    std::vector<brook::Point> after,std::optional<brook::Point> root,const brook::Array *graph) {
      brook::SkeletonizeOptions options;options.teasar=trace_parameters(params);options.anisotropy=aniso;
      options.fix_branching=branching;
      py::gil_scoped_release unlock;
      brook::ContextCall call(ctx);same_device(ctx,mask);same_device(ctx,dbf);same_device(ctx,graph);
      return brook::trace_component(ctx,std::move(mask),std::move(dbf),options,std::move(before),std::move(after),root,graph);
    },py::arg("mask"),py::arg("dbf"),py::arg("teasar_params")=py::dict(),
      py::arg("anisotropy")=std::array<float,3>{1,1,1},py::arg("fix_branching")=true,
      py::arg("manual_targets_before")=std::vector<brook::Point>{},py::arg("manual_targets_after")=std::vector<brook::Point>{},
      py::arg("root")=std::nullopt,py::arg("voxel_graph")=nullptr)
    .def("border_targets",guarded(&brook::border_targets),py::arg("components"),py::arg("count"),
         py::arg("anisotropy")=std::array<float,3>{1,1,1},py::call_guard<py::gil_scoped_release>())
    .def("_fix_avocados",[](brook::Context &ctx,brook::Array labels,brook::Array distance,std::vector<int64_t> mapping,double threshold,std::array<float,3> spacing,bool border,
                            std::vector<int64_t> roots,int64_t segment,const brook::Array *graph,std::vector<int64_t> keys) {
      brook::Components cc;cc.labels=std::move(labels);cc.mapping=std::move(mapping);cc.roots=std::move(roots);cc.keys=std::move(keys);
      {py::gil_scoped_release unlock;brook::ContextCall call(ctx);same_device(ctx,cc.labels);same_device(ctx,distance);same_device(ctx,graph);
       brook::fix_avocados(ctx,cc,distance,threshold,spacing,border,segment,graph);}
      return py::make_tuple(std::move(cc.labels),std::move(distance),cc.mapping,cc.roots);
    },py::arg("components"),py::arg("distance"),py::arg("mapping"),py::arg("threshold"),py::arg("anisotropy")=std::array<float,3>{1,1,1},py::arg("black_border")=false,
      py::arg("roots")=std::vector<int64_t>{},py::arg("segment")=0,py::arg("voxel_graph")=nullptr,py::arg("keys")=std::vector<int64_t>{})
    .def("_fill_all_holes",[](brook::Context &ctx,brook::Array labels) {
      brook::ContextCall call(ctx);same_device(ctx,labels);
      return brook::fill_all_holes(ctx,std::move(labels));
    },py::arg("components"),py::call_guard<py::gil_scoped_release>())
    .def("skeletonize_streamed",[](brook::Context &ctx,py::array labels,py::dict params,std::array<float,3> aniso,
                           double dust,bool branching,bool borders,std::optional<std::vector<int64_t>> ids,
                           std::vector<brook::Point> before,std::vector<brook::Point> after,py::object graph,bool fill_holes,
                           bool fix_avocados,size_t budget,std::optional<bool> black_border) {
      if(!graph.is_none()) throw py::value_error("streamed skeletonization does not support voxel_graph");
      brook::SkeletonizeOptions options;options.teasar=trace_parameters(params);options.anisotropy=aniso;options.dust=dust;
      options.fix_branching=branching;options.fix_borders=borders;options.fill_holes=fill_holes;options.fix_avocados=fix_avocados;
      options.object_ids=std::move(ids);options.targets_before=std::move(before);options.targets_after=std::move(after);
      options.black_border_override=black_border;
      auto input=volume(labels);py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      return brook::skeletonize_streamed(ctx,input,options,budget);
    },py::arg("labels"),py::arg("teasar_params")=py::dict(),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("dust_threshold")=1000,py::arg("fix_branching")=true,py::arg("fix_borders")=true,
      py::arg("object_ids")=std::nullopt,py::arg("extra_targets_before")=std::vector<brook::Point>{},
      py::arg("extra_targets_after")=std::vector<brook::Point>{},py::arg("voxel_graph")=py::none(),py::arg("fill_holes")=false,
      py::arg("fix_avocados")=false,py::arg("budget_bytes")=0,py::arg("black_border")=std::nullopt)
    .def("skeletonize",[](brook::Context &ctx,py::array labels,py::dict params,std::array<float,3> aniso,
                           double dust,bool branching,bool borders,std::optional<std::vector<int64_t>> ids,
                           std::vector<brook::Point> before,std::vector<brook::Point> after,py::object graph,bool fill_holes,bool fix_avocados,std::optional<bool> black_border) {
      auto options=skeletonize_options(params,aniso,dust,branching,borders,std::move(ids),std::move(before),std::move(after),fill_holes,fix_avocados,black_border);
      auto input=volume(labels);
      py::array graph_array;std::optional<brook_volume> graph_view;
      if(!graph.is_none()) { graph_array=py::cast<py::array>(graph);graph_view=volume(graph_array); }
      py::gil_scoped_release unlock;
      brook::ContextCall call(ctx);
      if(graph_view) options.voxel_graph=brook::upload(ctx,*graph_view);
      return brook::skeletonize(ctx,brook::upload(ctx,input),options);
    },py::arg("labels"),py::arg("teasar_params")=py::dict(),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("dust_threshold")=1000,py::arg("fix_branching")=true,py::arg("fix_borders")=true,
      py::arg("object_ids")=std::nullopt,py::arg("extra_targets_before")=std::vector<brook::Point>{},
      py::arg("extra_targets_after")=std::vector<brook::Point>{},py::arg("voxel_graph")=py::none(),py::arg("fill_holes")=false,py::arg("fix_avocados")=false,py::arg("black_border")=std::nullopt)
    .def("skeletonize_packed",[](brook::Context &ctx,py::array labels,py::dict params,std::array<float,3> aniso,
                           double dust,bool branching,bool borders,std::optional<std::vector<int64_t>> ids,
                           std::vector<brook::Point> before,std::vector<brook::Point> after,py::object graph,bool fill_holes,bool fix_avocados,std::optional<bool> black_border) {
      auto options=skeletonize_options(params,aniso,dust,branching,borders,std::move(ids),std::move(before),std::move(after),fill_holes,fix_avocados,black_border);
      auto input=volume(labels);
      py::array graph_array;std::optional<brook_volume> graph_view;
      if(!graph.is_none()) { graph_array=py::cast<py::array>(graph);graph_view=volume(graph_array); }
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      if(graph_view) options.voxel_graph=brook::upload(ctx,*graph_view);
      return brook::skeletonize_packed(ctx,brook::upload(ctx,input),options);
    },py::arg("labels"),py::arg("teasar_params")=py::dict(),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("dust_threshold")=1000,py::arg("fix_branching")=true,py::arg("fix_borders")=true,
      py::arg("object_ids")=std::nullopt,py::arg("extra_targets_before")=std::vector<brook::Point>{},
      py::arg("extra_targets_after")=std::vector<brook::Point>{},py::arg("voxel_graph")=py::none(),py::arg("fill_holes")=false,py::arg("fix_avocados")=false,py::arg("black_border")=std::nullopt)
    .def("upload_cuda_array",&upload_cuda_array,py::arg("array"),py::arg("stream")=py::none(),py::arg("borrow")=false)
    // Manual targets are per sample here: one list of points per sample, in that sample's own
    // coordinates (an empty outer list means no sample has any).
    .def("skeletonize_batch_device",[](brook::Context &ctx,std::vector<brook::Array> samples,py::dict params,std::array<float,3> aniso,
                           double dust,bool branching,bool borders,std::vector<std::vector<brook::Point>> before,
                           std::vector<std::vector<brook::Point>> after,
                           bool fill_holes,bool fix_avocados,std::optional<bool> black_border,std::vector<std::optional<brook::Array>> voxel_graphs) {
      auto options=skeletonize_options(params,aniso,dust,branching,borders,std::nullopt,{},{},fill_holes,fix_avocados,black_border);
      // one device graph per sample (an empty array: no graph for that sample); no list: no graphs
      std::vector<brook::Array> graphs;graphs.reserve(voxel_graphs.size());
      for(auto &g:voxel_graphs) graphs.push_back(g?std::move(*g):brook::Array{});
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      for(const auto &s:samples) same_device(ctx,s);
      for(const auto &g:graphs) if(g.storage) same_device(ctx,g);
      return brook::skeletonize_batch(ctx,std::move(samples),options,before,after,std::move(graphs));
    },py::arg("samples"),py::arg("teasar_params")=py::dict(),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("dust_threshold")=1000,py::arg("fix_branching")=true,py::arg("fix_borders")=true,
      py::arg("extra_targets_before")=std::vector<std::vector<brook::Point>>{},
      py::arg("extra_targets_after")=std::vector<std::vector<brook::Point>>{},
      py::arg("fill_holes")=false,py::arg("fix_avocados")=false,py::arg("black_border")=std::nullopt,
      py::arg("voxel_graphs")=std::vector<std::optional<brook::Array>>{})
    .def("skeletonize_device",[](brook::Context &ctx,const brook::Array &labels,py::dict params,std::array<float,3> aniso,
                           double dust,bool branching,bool borders,std::optional<std::vector<int64_t>> ids,
                           std::vector<brook::Point> before,std::vector<brook::Point> after,std::optional<brook::Array> graph,bool fill_holes,bool fix_avocados,std::optional<bool> black_border) {
      auto options=skeletonize_options(params,aniso,dust,branching,borders,std::move(ids),std::move(before),std::move(after),fill_holes,fix_avocados,black_border);
      if(graph) options.voxel_graph=std::move(*graph);   // a device-resident graph (borrowed or copied by upload_cuda_array)
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);same_device(ctx,labels);
      if(options.voxel_graph.storage) same_device(ctx,options.voxel_graph);
      return brook::skeletonize(ctx,labels,options);
    },py::arg("labels"),py::arg("teasar_params")=py::dict(),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("dust_threshold")=1000,py::arg("fix_branching")=true,py::arg("fix_borders")=true,
      py::arg("object_ids")=std::nullopt,py::arg("extra_targets_before")=std::vector<brook::Point>{},
      py::arg("extra_targets_after")=std::vector<brook::Point>{},py::arg("voxel_graph")=std::nullopt,py::arg("fill_holes")=false,py::arg("fix_avocados")=false,py::arg("black_border")=std::nullopt)
    .def("skeletonize_packed_device",[](brook::Context &ctx,const brook::Array &labels,py::dict params,std::array<float,3> aniso,
                           double dust,bool branching,bool borders,std::optional<std::vector<int64_t>> ids,
                           std::vector<brook::Point> before,std::vector<brook::Point> after,std::optional<brook::Array> graph,bool fill_holes,bool fix_avocados,std::optional<bool> black_border) {
      auto options=skeletonize_options(params,aniso,dust,branching,borders,std::move(ids),std::move(before),std::move(after),fill_holes,fix_avocados,black_border);
      if(graph) options.voxel_graph=std::move(*graph);
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);same_device(ctx,labels);
      if(options.voxel_graph.storage) same_device(ctx,options.voxel_graph);
      return brook::skeletonize_packed(ctx,labels,options);
    },py::arg("labels"),py::arg("teasar_params")=py::dict(),py::arg("anisotropy")=std::array<float,3>{1,1,1},
      py::arg("dust_threshold")=1000,py::arg("fix_branching")=true,py::arg("fix_borders")=true,
      py::arg("object_ids")=std::nullopt,py::arg("extra_targets_before")=std::vector<brook::Point>{},
      py::arg("extra_targets_after")=std::vector<brook::Point>{},py::arg("voxel_graph")=std::nullopt,py::arg("fill_holes")=false,py::arg("fix_avocados")=false,py::arg("black_border")=std::nullopt);
  py::class_<brook::BatchedSkeletons,std::unique_ptr<brook::BatchedSkeletons,GpuDeleter<brook::BatchedSkeletons>>>(m,"BatchedSkeletons")
    .def_property_readonly("packed",[](const brook::BatchedSkeletons &b){return b.packed;})
    .def_property_readonly("labels",[](const brook::BatchedSkeletons &b){return integers(b.labels);})
    .def_property_readonly("sample",[](const brook::BatchedSkeletons &b){return integers(b.sample);})
    .def_property_readonly("s_off",[](const brook::BatchedSkeletons &b){return integers(b.sample_offsets);})
    .def("slice",&brook::packed_slice,py::arg("first"),py::arg("last"));
  py::class_<brook::TraceWorkspace,std::unique_ptr<brook::TraceWorkspace,GpuDeleter<brook::TraceWorkspace>>>(m,"TraceWorkspace")
    .def(py::init([](brook::Context &ctx,int size) {
      brook::ContextCall call(ctx);
      return std::unique_ptr<brook::TraceWorkspace,GpuDeleter<brook::TraceWorkspace>>(new brook::TraceWorkspace(ctx,size));
    }),py::arg("context"),py::arg("size"),py::call_guard<py::gil_scoped_release>());
  m.attr("abi_version")=BROOK_ABI_VERSION;
  m.def("synapses_to_targets",[](py::array labels,py::dict synapses,bool progress) {
    return synapse_targets(nullptr,labels,synapses,progress);
  },py::arg("labels"),py::arg("synapses"),py::arg("progress")=false);
  m.def("_integer_set_order",[](std::vector<size_t> initial,std::vector<size_t> removed,std::vector<size_t> added) {
    brook::IntegerSet a,b,c;for(auto v:initial)a.add(v);for(auto v:removed)b.add(v);for(auto v:added)c.add(v);
    a.subtract(b);a.merge(c);return a.values();
  });
  m.def("_point_set_order",[](const std::vector<brook::Point> &points) {
    brook::PointSet set;for(auto p:points) set.add(p);return set.values();
  });
  m.def("_draft_window_plan",[](brook::Context &ctx,std::vector<int> starts,std::vector<int> lengths,
      std::vector<int> path,std::vector<float> dbf,std::array<int64_t,3> shape,std::array<float,3> aniso,
      double scale,double constant,int forced,int reserve,size_t table,size_t label_bytes,int labels,int drafts) {
    if(starts.size()!=lengths.size() || starts.size()>INT32_MAX) throw py::value_error("invalid draft records");
    for(size_t i=0;i<starts.size();++i) if(starts[i]<0 || lengths[i]<0 || size_t(starts[i])+size_t(lengths[i])>path.size()) throw py::value_error("draft path is out of bounds");
    for(int v:path) if(v<0 || size_t(v)>=dbf.size()) throw py::value_error("draft voxel is out of bounds");
    for(auto s:shape) if(s<=0) throw py::value_error("invalid draft volume dimensions");
    for(float a:aniso) if(!(a>0)) throw py::value_error("invalid draft anisotropy");
    std::array<int,3> requested;std::optional<brook::DraftWindowPlan> fitted;
    {
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      brook::Buffer<int> ds(ctx,starts.size()),dl(ctx,lengths.size()),dp(ctx,path.size());brook::Buffer<float> dd(ctx,dbf.size());
      ds.set(ctx,starts.data(),starts.size());dl.set(ctx,lengths.data(),lengths.size());dp.set(ctx,path.data(),path.size());dd.set(ctx,dbf.data(),dbf.size());
      requested=brook::plan_draft_window(ctx,ds.data(),dl.data(),dp.data(),dd.data(),int(starts.size()),shape,aniso,scale,constant,forced);
      fitted=brook::fit_draft_window(requested,reserve,table,label_bytes,labels,drafts);
    }
    py::dict out;out["requested"]=py::cast(requested);
    out["fitted"]=fitted?py::cast(fitted->log2):py::none();out["labels"]=fitted?fitted->drafting_labels:0;
    out["windows"]=fitted?fitted->windows:0;out["voxels"]=fitted?fitted->voxels:0;return out;
  });
  m.def("_draft_reserve_budget",&brook::draft_reserve_budget);
  m.def("_draft_estimate",[](brook::Context &ctx,std::vector<int> active,std::vector<long long> cursor,
      std::vector<long long> ends,std::vector<int> order,std::vector<uint8_t> alive,std::vector<int> paths,std::vector<long long> voxels) {
    if(cursor.size()!=ends.size() || cursor.size()!=paths.size() || cursor.size()!=voxels.size()) throw py::value_error("draft table sizes differ");
    for(int label:active) if(label<0 || size_t(label)>=cursor.size() || cursor[label]<0 || ends[label]<cursor[label] || size_t(ends[label])>order.size()) throw py::value_error("invalid draft label range");
    for(int v:order) if(v<0 || size_t(v)>=alive.size()) throw py::value_error("invalid draft order index");
    brook::DraftEstimate out;
    {
      py::gil_scoped_release unlock;brook::ContextCall call(ctx);
      brook::Buffer<int> da(ctx,active.size()),doo(ctx,order.size()),dp(ctx,paths.size());brook::Buffer<uint8_t> dal(ctx,alive.size());
      brook::Buffer<long long> dc(ctx,cursor.size()),de(ctx,ends.size()),dv(ctx,voxels.size());
      da.set(ctx,active.data(),active.size());doo.set(ctx,order.data(),order.size());dp.set(ctx,paths.data(),paths.size());dal.set(ctx,alive.data(),alive.size());
      dc.set(ctx,cursor.data(),cursor.size());de.set(ctx,ends.data(),ends.size());dv.set(ctx,voxels.data(),voxels.size());
      out=brook::estimate_draft_paths(ctx,da.data(),int(active.size()),dc.data(),de.data(),doo.data(),dal.data(),dp.data(),dv.data());
    }
    return py::make_tuple(out.labels,out.remaining);
  });
  m.def("_prepare_batched",[](brook::Context &ctx,py::array labels,py::dict params,std::array<float,3> aniso,double dust,bool branching,bool fix_borders) {
    brook::SkeletonizeOptions options;options.teasar=trace_parameters(params);options.anisotropy=aniso;
    options.dust=dust;options.fix_branching=branching;options.fix_borders=fix_borders;
    auto view=volume(labels);brook::Components cc;brook::BatchedPreparation prep;
    {
      py::gil_scoped_release unlock;
      brook::ContextCall call(ctx);
      auto input=brook::upload(ctx,view);cc=brook::connected_components(ctx,input);
      auto boxes=brook::analyze(ctx,cc.labels,cc.roots.size());
      bool bb=cc.roots.size()==1 && boxes[1].count==int64_t(input.size());
      auto dbf=brook::edt(ctx,cc.labels,aniso,bb);
      auto targets=fix_borders?brook::border_targets(ctx,cc.labels,cc.roots.size(),aniso):std::vector<std::vector<brook::Point>>(cc.mapping.size());
      prep=brook::prepare_batched(ctx,cc,dbf,boxes,targets,options);
    }
    py::dict result;result["active"]=py::cast(prep.active);result["soma"]=py::cast(prep.soma);
    result["dbf_max"]=py::cast(prep.dbf_max);result["max_daf"]=py::cast(prep.max_daf);
    result["root"]=py::cast(prep.root);result["target"]=py::cast(prep.target);result["daf"]=py::cast(prep.daf);
    result["parents"]=prep.parents.storage?py::cast(prep.parents):py::none();
    return result;
  },py::arg("context"),py::arg("labels"),py::arg("teasar_params"),py::arg("anisotropy")=std::array<float,3>{1,1,1},
    py::arg("dust_threshold")=1000,py::arg("fix_branching")=true,py::arg("fix_borders")=true);
  bind_runtime(m);
  bind_intake(m);
  bind_public_primitives(m);
}
