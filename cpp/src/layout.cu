// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#include "runtime.hpp"
#include <algorithm>
#include <cstdlib>

namespace brook {
namespace {
// Transpose X/Z for each Y plane: both global reads and writes are contiguous.
// Swapping the logical X/Z extents implements the inverse C-to-F conversion.
template<class T> __global__ void transpose_xz(const T *input,T *output,long long sx,long long sy,long long sz) {
  __shared__ T tile[32][33];
  long long nx=(sx+31)/32,nz=(sz+31)/32,tiles=nx*nz*sy;
  for(long long b=blockIdx.x;b<tiles;b+=gridDim.x) {
    long long bx=b%nx,bz=(b/nx)%nz,y=b/(nx*nz);
    long long x=bx*32+threadIdx.x,z=bz*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(x<sx && z+j<sz)
      tile[threadIdx.y+j][threadIdx.x]=input[x+sx*(y+sy*(z+j))];
    __syncthreads();
    x=bx*32+threadIdx.y;z=bz*32+threadIdx.x;
    for(int j=0;j<32;j+=8) if(x+j<sx && z<sz)
      output[z+sz*(y+sy*(x+j))]=tile[threadIdx.x][threadIdx.y+j];
    __syncthreads();
  }
}
template<class T> void run(Context &ctx,const Array &input,Array &output) {
  auto s=output.shape;long long blocks=((s[2]+31)/32)*((s[0]+31)/32)*s[1];
  transpose_xz<<<unsigned(std::min(blocks,65535LL)),dim3(32,8),0,ctx.stream>>>(
    static_cast<const T*>(input.data()),static_cast<T*>(output.data()),s[2],s[1],s[0]);
  BROOK_CUDA(cudaGetLastError());
}
}
bool upload_c_contiguous(Context &ctx,const brook_volume &view,Array &output) {
  const bool device_source=view.memory==BROOK_DEVICE;   // no host fallback: always transpose on the GPU
  auto mode=std::getenv("BROOK_LAYOUT_GPU");
  if(!device_source && mode && std::string(mode)=="0") return false;
  if(!device_source && (!mode || std::string(mode)!="1")) {
    int processors=0;BROOK_CUDA(cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,ctx.device));
    // Avoid allocation/launch overhead when the transpose cannot populate one
    // block per SM. This follows the actual grid, not a fixed voxel threshold.
    auto s=output.shape;auto tiles=((s[0]+31)/32)*((s[2]+31)/32)*s[1];
    if(tiles<processors) return false;
  }
  size_t free=0,total=0;BROOK_CUDA(cudaMemGetInfo(&free,&total));
  // The second buffer follows the engine's 85% free-memory admission rule.
  if(!device_source && output.bytes()>free/100*85) return false;
  Array staging;
  try {staging=allocate(ctx,output.shape,output.dtype);}
  catch(const CudaError &e) {
    if(device_source || e.code!=cudaErrorMemoryAllocation) throw;
    cudaGetLastError();return false;
  }
  BROOK_CUDA(cudaMemcpyAsync(staging.data(),view.data,output.bytes(),device_source?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice,ctx.stream));
  switch(dtype_size(output.dtype)) {
    case 1:run<uint8_t>(ctx,staging,output);break;
    case 2:run<uint16_t>(ctx,staging,output);break;
    case 4:run<uint32_t>(ctx,staging,output);break;
    case 8:run<uint64_t>(ctx,staging,output);break;
  }
  ctx.synchronize();return true;
}
}
