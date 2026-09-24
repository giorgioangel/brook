// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "device_check.hpp"
#include "brook_build_config.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <set>

namespace brook {
namespace {
// Compiled with the library's architecture list: if the driver cannot load this kernel for a
// device, it cannot load any other Brook kernel either.
__global__ void image_probe() {}

std::atomic<unsigned long long> passed_low{0};
std::mutex passed_mutex;
std::set<int> passed_high;
bool passed(int device) {
  if(device<64) return (passed_low.load(std::memory_order_acquire)>>device)&1ull;
  std::lock_guard<std::mutex> lock(passed_mutex);return passed_high.count(device)>0;
}
void mark_passed(int device) {
  if(device<64) passed_low.fetch_or(1ull<<device,std::memory_order_acq_rel);
  else { std::lock_guard<std::mutex> lock(passed_mutex);passed_high.insert(device); }
}
// Conditional graph nodes (the tracing loops) need a CUDA 12.3 driver; a CUDA 13 runtime needs a
// CUDA 13.0 driver whatever Brook uses.
int required_driver(int runtime) { return std::max(12030,runtime/1000*1000); }
std::string driver_branch(int version) {
  if(version==12030) return " (R545+)";
  if(version==13000) return " (R580+)";
  return "";
}
bool image_error(cudaError_t code) {
  switch(code) {
    case cudaErrorInvalidDeviceFunction: case cudaErrorInvalidKernelImage: case cudaErrorNoKernelImageForDevice:
    case cudaErrorInvalidPtx: case cudaErrorJitCompilerNotFound: case cudaErrorUnsupportedPtxVersion:
    case cudaErrorJitCompilationDisabled: case cudaErrorInvalidResourceHandle: return true;
    default: return false;
  }
}
std::string error_tag(cudaError_t code) {
  return std::string("[")+cudaGetErrorName(code)+": "+cudaGetErrorString(code)+"]";
}
std::string named_device(const DeviceFacts &f) {
  return "CUDA device "+std::to_string(f.device)+(f.name.empty()?"":" ("+f.name+")");
}
// The newest architecture number in a list such as "sm_90 sm_100a compute_120" (0 for none).
int newest_architecture(const std::string &names) {
  int newest=0;
  for(size_t at=names.find('_');at!=std::string::npos;at=names.find('_',at+1))
    newest=std::max(newest,std::atoi(names.c_str()+at+1));
  return newest;
}
std::string image_message(const DeviceFacts &f) {
  const int number=f.major*10+f.minor;
  const std::string cc=std::to_string(f.major)+"."+std::to_string(f.minor),arch=std::to_string(number);
  const std::string device="CUDA device "+std::to_string(f.device)+" ("+(f.name.empty()?"":f.name+", ")+"compute capability "+cc+")";
  std::string contents;   // what the build has first, then what it lacks
  if(!f.sass.empty() && !f.ptx.empty()) contents="SASS for "+f.sass+" and PTX for "+f.ptx;
  else if(!f.sass.empty()) contents="SASS for "+f.sass+" and no PTX";
  else if(!f.ptx.empty()) contents="PTX for "+f.ptx+" and no SASS";
  else contents="no SASS and no PTX";
  std::string build="build Brook for this GPU with CMAKE_CUDA_ARCHITECTURES="+arch+
    " (pip: --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="+arch+") or native";
  // Every toolkit Brook accepts (12.3+) compiles for 9.0; a GPU newer than both that and everything
  // in this build may need a newer toolkit than the one that made it.
  if(number>90 && number>std::max(newest_architecture(f.sass),newest_architecture(f.ptx)))
    build+=", using a CUDA toolkit that supports compute capability "+cc+" (this build was made with CUDA "+cuda_version_string(f.toolkit)+")";
  std::string lead="This Brook build has no SASS for "+device,fix;
  switch(f.image) {
    case cudaErrorUnsupportedPtxVersion:
      lead+=", and the installed driver (CUDA "+cuda_version_string(f.driver)+") cannot compile its PTX, which comes from CUDA "+cuda_version_string(f.toolkit);
      fix="Update the NVIDIA driver, or "+build;break;
    case cudaErrorJitCompilationDisabled:
      lead+=", and PTX JIT compilation is disabled (CUDA_DISABLE_PTX_JIT)";
      fix="Unset CUDA_DISABLE_PTX_JIT, or "+build;break;
    case cudaErrorJitCompilerNotFound:
      lead+=", and the driver's PTX JIT compiler (libnvidia-ptxjitcompiler.so.1) was not found";
      fix="Make the driver's PTX JIT compiler library available, or "+build;break;
    case cudaErrorInvalidPtx:
      lead+=", and the driver could not compile its PTX";
      fix="Update the NVIDIA driver, or "+build;break;
    default:
      lead="This Brook build has no GPU code for "+device;
      fix="B"+build.substr(1);
  }
  return lead+": it contains "+contents+". "+fix+". "+error_tag(f.image);
}
}

std::string cuda_version_string(int version) {
  return std::to_string(version/1000)+"."+std::to_string((version%1000)/10);
}

DeviceProblem device_support_problem(const DeviceFacts &f) {
  if(f.count<=0) {
    std::string where=!f.driver?": no NVIDIA driver is loaded":f.visible?std::string(" (CUDA_VISIBLE_DEVICES=\"")+f.visible+"\")":"";
    return {cudaErrorNoDevice,"Brook found no CUDA device"+where+". Brook needs an NVIDIA GPU of compute capability 8.0 or newer."};
  }
  const int needed=required_driver(f.runtime);
  if(f.driver<needed) {
    const bool runtime_reason=needed>12030;
    std::string text="Brook needs an NVIDIA driver that supports CUDA "+cuda_version_string(needed)+" or newer"+driver_branch(needed)+
      (runtime_reason?" because this build uses the CUDA "+cuda_version_string(f.runtime)+" runtime":std::string(" for conditional CUDA graph nodes"))+
      "; the installed driver supports CUDA "+cuda_version_string(f.driver)+". Update the NVIDIA driver";
    if(runtime_reason && f.driver>=12030) text+=", or install a Brook build made with CUDA "+std::to_string(f.driver/1000);
    return {cudaErrorInsufficientDriver,text+"."};
  }
  if(f.major<8)
    return {cudaErrorNotSupported,"Brook needs an NVIDIA GPU of compute capability 8.0 or newer (Ampere or later); "+named_device(f)+
      " has compute capability "+std::to_string(f.major)+"."+std::to_string(f.minor)+"."};
  if(!f.cooperative)
    return {cudaErrorNotSupported,named_device(f)+" does not support cooperative kernel launches (cudaDevAttrCooperativeLaunch is 0), which Brook needs."};
  if(f.image!=cudaSuccess) {
    if(image_error(f.image)) return {f.image,image_message(f)};
    return {f.image,"Brook could not check its GPU code on "+named_device(f)+": cudaFuncGetAttributes failed. "+error_tag(f.image)};
  }
  return {};
}

int visible_device_count() {
  int count=0;
  const cudaError_t status=cudaGetDeviceCount(&count);
  if(status==cudaSuccess) return count;
  cudaGetLastError();
  int driver=0,runtime=0;cudaDriverGetVersion(&driver);cudaRuntimeGetVersion(&runtime);
  if(status==cudaErrorNoDevice || !driver) return 0;   // no GPU, or no NVIDIA driver loaded
  if(status==cudaErrorInsufficientDriver) {
    DeviceFacts facts;facts.driver=driver;facts.runtime=runtime;
    auto problem=device_support_problem(facts);
    if(problem.code!=cudaSuccess) throw DeviceSupportError(status,problem.message);
  }
  if(status==cudaErrorSystemDriverMismatch)
    throw DeviceSupportError(status,"Brook cannot use CUDA: the CUDA user-mode driver (libcuda.so) does not match the loaded NVIDIA kernel driver. "
      "Check that LD_LIBRARY_PATH does not select a CUDA forward-compatibility package (cuda-compat) this system cannot use, or reinstall the NVIDIA driver. "+error_tag(status));
  if(status==cudaErrorCompatNotSupportedOnDevice)
    throw DeviceSupportError(status,"Brook cannot use CUDA: the CUDA forward-compatibility package (cuda-compat) on LD_LIBRARY_PATH does not support this GPU. "
      "Remove it and use a Brook build made with CUDA "+std::to_string(driver/1000)+", or update the NVIDIA driver. "+error_tag(status));
  throw CudaError(CudaError::Message{},status,"Brook could not list the CUDA devices: cudaGetDeviceCount failed. "+error_tag(status));
}

void require_supported_device(int device) {
  if(device<0) {
    int current=0;
    if(cudaGetDevice(&current)==cudaSuccess) device=current;
    else cudaGetLastError();
  }
  if(device>=0 && passed(device)) return;
  DeviceFacts f;
  f.count=visible_device_count();
  cudaDriverGetVersion(&f.driver);cudaRuntimeGetVersion(&f.runtime);f.toolkit=CUDART_VERSION;
  f.visible=std::getenv("CUDA_VISIBLE_DEVICES");f.sass=BROOK_CUDA_SASS;f.ptx=BROOK_CUDA_PTX;
  f.device=std::max(device,0);
  if(f.count>0) {
    if(f.device>=f.count) throw std::invalid_argument("CUDA device ordinal is out of range");
    BROOK_CUDA(cudaDeviceGetAttribute(&f.major,cudaDevAttrComputeCapabilityMajor,f.device));
    BROOK_CUDA(cudaDeviceGetAttribute(&f.minor,cudaDevAttrComputeCapabilityMinor,f.device));
    BROOK_CUDA(cudaDeviceGetAttribute(&f.cooperative,cudaDevAttrCooperativeLaunch,f.device));
  }
  auto problem=device_support_problem(f);
  if(problem.code==cudaSuccess) {
    DeviceGuard guard(f.device);
    cudaFuncAttributes attributes{};
    f.image=cudaFuncGetAttributes(&attributes,image_probe);
    if(f.image!=cudaSuccess) { cudaGetLastError();problem=device_support_problem(f); }
  }
  if(problem.code!=cudaSuccess) {
    if(f.count>0) {   // the name only matters for the message
      cudaDeviceProp properties{};
      if(cudaGetDeviceProperties(&properties,f.device)==cudaSuccess) { f.name=properties.name;problem=device_support_problem(f); }
      else cudaGetLastError();
    }
    throw DeviceSupportError(problem.code,problem.message);
  }
  mark_passed(f.device);
}

BuildInfo build_info() {
  BuildInfo info;
  info.architectures=BROOK_CUDA_ARCHITECTURES;info.sass=BROOK_CUDA_SASS;info.ptx=BROOK_CUDA_PTX;info.toolkit=CUDART_VERSION;
  cudaRuntimeGetVersion(&info.runtime);cudaDriverGetVersion(&info.driver);
  return info;
}

std::string compiled_architectures() {
  const std::string sass=BROOK_CUDA_SASS,ptx=BROOK_CUDA_PTX;
  return (sass.empty()?std::string("no SASS"):"SASS "+sass)+"; "+(ptx.empty()?std::string("no PTX"):"PTX "+ptx);
}
}
