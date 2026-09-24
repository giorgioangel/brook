// The startup check: its messages for made-up devices, and the real check on device 0.
#include "device_check.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

namespace {
int failures=0;
void expect(bool condition,const char *what) {
  if(!condition) { std::printf("FAILED: %s\n",what);++failures; }
}
bool has(const std::string &text,const char *part) { return text.find(part)!=std::string::npos; }
brook::DeviceFacts supported() {
  brook::DeviceFacts f;
  f.count=1;f.device=0;f.driver=12080;f.runtime=12080;f.toolkit=12080;f.major=9;f.minor=0;f.cooperative=1;
  f.name="NVIDIA H100 80GB HBM3";f.sass="sm_89";f.ptx="";
  return f;
}
std::string show(const char *label,const brook::DeviceFacts &f,cudaError_t code) {
  auto problem=brook::device_support_problem(f);
  std::printf("%s [%d]: %s\n",label,int(problem.code),problem.message.c_str());
  expect(problem.code==code,label);
  return problem.message;
}
}

int main() {
  auto f=supported();
  expect(brook::device_support_problem(f).code==cudaSuccess && brook::device_support_problem(f).message.empty(),"supported device");

  auto none=f;none.count=0;none.visible="";
  expect(has(show("A visible",none,cudaErrorNoDevice),"(CUDA_VISIBLE_DEVICES=\"\")"),"A names CUDA_VISIBLE_DEVICES");
  none.driver=0;
  expect(has(show("A driver",none,cudaErrorNoDevice),"no NVIDIA driver is loaded"),"A without a driver");

  auto turing=f;turing.major=7;turing.minor=5;turing.name="Tesla T4";
  expect(has(show("B",turing,cudaErrorNotSupported),"compute capability 7.5"),"B names the capability");

  auto solo=f;solo.cooperative=0;
  expect(has(show("C",solo,cudaErrorNotSupported),"cooperative kernel launches"),"C");

  auto old_driver=f;old_driver.driver=12020;
  auto d1=show("D conditional",old_driver,cudaErrorInsufficientDriver);
  expect(has(d1,"CUDA 12.3 or newer (R545+) for conditional CUDA graph nodes") && has(d1,"supports CUDA 12.2"),"D conditional nodes");
  auto cu13=f;cu13.runtime=13040;cu13.toolkit=13040;
  auto d2=show("D runtime",cu13,cudaErrorInsufficientDriver);
  expect(has(d2,"CUDA 13.0 or newer (R580+) because this build uses the CUDA 13.4 runtime") && has(d2,"made with CUDA 12"),"D runtime");

  auto image=f;image.image=cudaErrorNoKernelImageForDevice;
  auto e1=show("E 209",image,cudaErrorNoKernelImageForDevice);
  expect(has(e1,"no GPU code for CUDA device 0 (NVIDIA H100 80GB HBM3, compute capability 9.0)") && has(e1,"SASS for sm_89 and no PTX") &&
         has(e1,"CMAKE_CUDA_ARCHITECTURES=90") && has(e1,"cudaErrorNoKernelImageForDevice"),"E 209");
  expect(!has(e1,"CUDA toolkit that supports"),"E 209: every toolkit Brook accepts builds for 9.0");
  auto newer=image;newer.major=10;newer.minor=0;newer.name="NVIDIA B200";newer.sass="sm_80 sm_90";newer.ptx="";
  auto e_newer=show("E newer GPU",newer,cudaErrorNoKernelImageForDevice);
  expect(has(e_newer,"CMAKE_CUDA_ARCHITECTURES=100") &&
         has(e_newer,"using a CUDA toolkit that supports compute capability 10.0 (this build was made with CUDA 12.8)"),"E names the toolkit for a newer GPU");
  image.image=cudaErrorInvalidResourceHandle;
  expect(has(show("E 400",image,cudaErrorInvalidResourceHandle),"no GPU code"),"E 400");
  image.image=cudaErrorUnsupportedPtxVersion;image.sass="";image.ptx="compute_80";image.toolkit=12080;image.driver=12040;
  auto e222=show("E 222",image,cudaErrorUnsupportedPtxVersion);
  expect(has(e222,"cannot compile its PTX, which comes from CUDA 12.8") && has(e222,"PTX for compute_80 and no SASS"),"E 222");
  image.image=cudaErrorJitCompilationDisabled;image.driver=12080;
  expect(has(show("E 223",image,cudaErrorJitCompilationDisabled),"CUDA_DISABLE_PTX_JIT"),"E 223");
  image.image=cudaErrorIllegalAddress;
  expect(has(show("other",image,cudaErrorIllegalAddress),"could not check its GPU code"),"other errors are not called missing code");

  std::printf("compiled: %s\n",brook::compiled_architectures().c_str());
  expect(!brook::compiled_architectures().empty(),"compiled architectures");
  auto info=brook::build_info();
  expect(!info.architectures.empty() && info.toolkit>=12030 && info.runtime>0,"build info");

  const int count=brook::visible_device_count();
  std::printf("visible devices: %d\n",count);
  if(count>0) {
    auto begin=std::chrono::steady_clock::now();
    brook::require_supported_device(0);
    double first=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count(),fastest=1e30;
    for(int i=0;i<1000;++i) {
      begin=std::chrono::steady_clock::now();
      brook::require_supported_device(i%2?0:-1);
      fastest=std::min(fastest,std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count());
    }
    std::printf("check: first %.1f us, cached %.3f us\n",first,fastest);
    expect(fastest<10,"the cached check is free");
    brook::Context ctx(0);
    bool out_of_range=false;
    try { brook::Context beyond(count); } catch(const std::invalid_argument &) { out_of_range=true; }
    expect(out_of_range,"an ordinal past the last device is an invalid argument");
  } else {   // CUDA_VISIBLE_DEVICES= : the check and a Context both give text A
    for(int attempt=0;attempt<2;++attempt) {
      std::string text;
      try { if(attempt) brook::Context ctx(0);else brook::require_supported_device(-1); }
      catch(const brook::DeviceSupportError &error) { text=error.what();expect(error.code==cudaErrorNoDevice,"no-device code"); }
      std::printf("no device: %s\n",text.c_str());
      expect(has(text,"Brook found no CUDA device"),"text A without devices");
    }
  }
  return failures?1:0;
}
