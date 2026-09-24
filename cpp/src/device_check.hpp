// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "runtime.hpp"
#include <string>

namespace brook {
// Startup check: every Context runs it once per device before its first CUDA work, and Python's
// require_gpu() runs the same function, so both report the same message.
class DeviceSupportError : public CudaError {
 public:
  DeviceSupportError(cudaError_t value,const std::string &message):CudaError(CudaError::Message{},value,message) {}
};
// What the check looks at. device_support_problem() only formats, so tests can feed it any facts.
struct DeviceFacts {
  int count=1;                  // visible CUDA devices
  const char *visible=nullptr;  // CUDA_VISIBLE_DEVICES, when set
  int device=0,driver=0,runtime=0,toolkit=0,major=0,minor=0,cooperative=1;
  cudaError_t image=cudaSuccess;   // cudaFuncGetAttributes on a kernel compiled with the library's architectures
  std::string name,sass,ptx;
};
struct DeviceProblem { cudaError_t code=cudaSuccess;std::string message; };
DeviceProblem device_support_problem(const DeviceFacts &facts);   // code cudaSuccess: supported
int visible_device_count();                // 0 without a driver or device; throws when the driver is too old
void require_supported_device(int device); // -1: the current device; cached after the first success
struct BuildInfo { std::string architectures,sass,ptx;int toolkit=0,runtime=0,driver=0; };
BuildInfo build_info();
std::string compiled_architectures();      // "SASS sm_90; PTX compute_90"
std::string cuda_version_string(int version);
}
