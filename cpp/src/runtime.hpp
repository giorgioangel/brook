// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include <brook/brook.h>
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <memory>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brook {
class CudaError : public std::runtime_error {
 public:
  cudaError_t code;
  CudaError(cudaError_t value, const char *expression)
    : std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(value)), code(value) {}
  struct Message {};   // tag: the text is already the whole message
  CudaError(Message, cudaError_t value, const std::string &message) : std::runtime_error(message), code(value) {}
};
inline void check(cudaError_t value, const char *expression) {
  if (value != cudaSuccess) throw CudaError(value, expression);
}
#define BROOK_CUDA(expr) ::brook::check((expr), #expr)

class DeviceGuard {
  int previous_=-1;
  bool changed_=false;
 public:
  explicit DeviceGuard(int device) {
    BROOK_CUDA(cudaGetDevice(&previous_));
    if(previous_!=device) { BROOK_CUDA(cudaSetDevice(device));changed_=true; }
  }
  ~DeviceGuard() { if(changed_) cudaSetDevice(previous_); }
  DeviceGuard(const DeviceGuard&)=delete;
};
class DestructorDeviceGuard {
  int previous_=-1;
  bool changed_=false;
 public:
  explicit DestructorDeviceGuard(int device) noexcept {
    if(cudaGetDevice(&previous_)!=cudaSuccess) previous_=-1;
    if(previous_!=device) { cudaSetDevice(device);changed_=true; }
  }
  ~DestructorDeviceGuard() { if(changed_ && previous_>=0) cudaSetDevice(previous_); }
  DestructorDeviceGuard(const DestructorDeviceGuard&)=delete;
};

struct BorderPlan;
struct Context {
  int device;
  cudaStream_t stream = nullptr;
  std::map<std::string,int64_t> stats;
  mutable std::mutex call_mutex;
  std::shared_ptr<BorderPlan> border_plan;
  explicit Context(int ordinal);
  ~Context();
  Context(const Context &) = delete;
  void activate() const { BROOK_CUDA(cudaSetDevice(device)); }
  void synchronize() const { activate(); BROOK_CUDA(cudaStreamSynchronize(stream)); }
};

// Public entry points serialize one context and restore the caller's CUDA device.
// Python must release the GIL before constructing this guard.
struct ContextCall {
  std::unique_lock<std::mutex> lock;
  DeviceGuard device;
  explicit ContextCall(Context &ctx):lock(ctx.call_mutex),device(ctx.device) {}
};

struct Allocation {
  void *data = nullptr;
  size_t bytes = 0;
  int device = 0;
  bool managed = false;
  std::function<void()> release;   // borrowed memory: runs instead of cudaFree when the last user goes
  Allocation(size_t count, int ordinal, bool use_managed=false);
  // Borrow `external` (device memory owned elsewhere) for as long as this allocation lives.
  Allocation(void *external, size_t count, int ordinal, std::function<void()> on_release);
  ~Allocation();
  Allocation(const Allocation &) = delete;
};

template<class T> class Buffer {
  std::shared_ptr<Allocation> storage_;
  size_t size_ = 0;
  size_t byte_offset_ = 0;
 public:
  Buffer() = default;
  Buffer(Context &ctx, size_t n, bool managed=false) : size_(n) {
    if(n>SIZE_MAX/sizeof(T)) throw std::invalid_argument("buffer byte size overflow");
    storage_=std::make_shared<Allocation>(n*sizeof(T),ctx.device,managed);
  }
  // Internal typed scratch view. Shared ownership keeps its arena alive until
  // every view is gone; views never own or free an interior CUDA pointer.
  Buffer(std::shared_ptr<Allocation> storage,size_t offset,size_t n)
    : storage_(std::move(storage)),size_(n),byte_offset_(offset) {
    if(!storage_ || offset%alignof(T) || offset>storage_->bytes || n>(storage_->bytes-offset)/sizeof(T))
      throw std::invalid_argument("invalid scratch buffer view");
  }
  T *data() const { return storage_ && storage_->data ? reinterpret_cast<T *>(static_cast<char *>(storage_->data)+byte_offset_) : nullptr; }
  size_t size() const { return size_; }
  std::shared_ptr<Allocation> storage() const { return storage_; }
  void clear(Context &ctx) { if (size_) BROOK_CUDA(cudaMemsetAsync(data(), 0, size_ * sizeof(T), ctx.stream)); }
  void set(Context &ctx, const T *host, size_t n) {
    if (n > size_) throw std::invalid_argument("buffer upload exceeds capacity");
    if (n) BROOK_CUDA(cudaMemcpyAsync(data(), host, n * sizeof(T), cudaMemcpyHostToDevice, ctx.stream));
  }
  std::vector<T> get(Context &ctx, size_t n) const {
    if (n > size_) throw std::invalid_argument("buffer download exceeds capacity");
    std::vector<T> out(n);
    if (n) BROOK_CUDA(cudaMemcpyAsync(out.data(), data(), n * sizeof(T), cudaMemcpyDeviceToHost, ctx.stream));
    ctx.synchronize();
    return out;
  }
};

struct Array {
  std::shared_ptr<Allocation> storage;
  std::array<int64_t, 3> shape = {0, 0, 0};
  brook_dtype dtype = BROOK_U8;
  int device = 0;
  size_t byte_offset = 0;
  std::optional<std::vector<int64_t>> logical_shape;
  std::vector<int64_t> logical_strides;
  size_t size() const;
  size_t bytes() const;
  void *data() const { return storage && storage->data ? static_cast<char*>(storage->data)+byte_offset : nullptr; }
  std::vector<int64_t> tensor_shape() const;
  std::vector<int64_t> tensor_strides() const;
  brook_volume view() const;
  void copy_to_host(void *out, size_t capacity) const;
};

size_t dtype_size(brook_dtype dtype);
size_t volume_size(const std::array<int64_t, 3> &shape);
Array allocate(Context &ctx, std::array<int64_t, 3> shape, brook_dtype dtype);
Array upload(Context &ctx, const brook_volume &view);
Array tensor_view(Array array,std::vector<int64_t> shape,std::vector<int64_t> strides);
struct Components {
  Array labels; std::vector<int64_t> mapping, roots; bool input_uniform=false;
  // keys[id]: Kimimaro's id of compact id `id` where the two differ (voxel graphs: see graph_components), empty otherwise.
  std::vector<int64_t> keys;
};
Components connected_components(Context &ctx, const Array &labels);
// segment > 0: the z axis is a stack of independent samples of that depth; z lines never cross them.
Array edt(Context &ctx, const Array &labels, std::array<float, 3> anisotropy, bool black_border,int dimensions=3,int64_t segment=0);
}
