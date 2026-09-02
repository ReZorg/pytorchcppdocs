// Validation-only weak stubs for the CUDA device library and CUDA runtime
// that the CPU-only LibTorch wheel does not ship. In real CUDA builds
// libc10_cuda.so / libcudart.so provide these symbols and the weak
// definitions are overridden; they are never executed here because the
// runtime availability check is false.

#include <c10/core/Stream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <stdexcept>

#define CUDA_STUB __attribute__((weak))

extern "C" {
CUDA_STUB cudaError_t cudaGetLastError() { return cudaSuccess; }
CUDA_STUB const char* cudaGetErrorString(cudaError_t) { return "stub"; }
CUDA_STUB cudaError_t cudaGetDeviceCount(int* count) {
  *count = 0;
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaSetDevice(int) { return cudaSuccess; }
CUDA_STUB cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
CUDA_STUB cudaError_t cudaStreamSynchronize(cudaStream_t) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaStreamIsCapturing(cudaStream_t,
                                            cudaStreamCaptureStatus*) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaStreamGetPriority(cudaStream_t, int*) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaDeviceGetStreamPriorityRange(int*, int*) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
CUDA_STUB cudaError_t cudaStreamCreateWithPriority(cudaStream_t*,
                                                   unsigned int, int) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
CUDA_STUB cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t,
                                          unsigned int) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
CUDA_STUB cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
CUDA_STUB cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
CUDA_STUB cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) {
  return cudaSuccess;
}
CUDA_STUB cudaError_t cudaMemcpyAsync(void*, const void*, size_t,
                                      cudaMemcpyKind, cudaStream_t) {
  return cudaSuccess;
}
}  // extern "C"

namespace c10::cuda {

CUDA_STUB DeviceIndex device_count() noexcept { return 0; }
CUDA_STUB DeviceIndex device_count_ensure_non_zero() { return 0; }
CUDA_STUB DeviceIndex current_device() { return -1; }
CUDA_STUB void set_device(DeviceIndex, const bool) {}
CUDA_STUB cudaError_t GetDeviceCount(int* count) {
  *count = 0;
  return cudaSuccess;
}
CUDA_STUB cudaError_t GetDevice(DeviceIndex* device) {
  *device = 0;
  return cudaSuccess;
}
CUDA_STUB cudaError_t SetDevice(DeviceIndex, const bool) { return cudaSuccess; }
CUDA_STUB cudaError_t MaybeSetDevice(DeviceIndex) { return cudaSuccess; }
CUDA_STUB DeviceIndex ExchangeDevice(DeviceIndex) { return -1; }
CUDA_STUB DeviceIndex MaybeExchangeDevice(DeviceIndex) { return -1; }
CUDA_STUB void device_synchronize() {}

CUDA_STUB CUDAErrorLogCapture::CUDAErrorLogCapture() noexcept {}
CUDA_STUB void c10_cuda_check_implementation(
    int, const char*, const char*, unsigned int, bool,
    CUDAErrorLogCapture*) {}

CUDA_STUB CUDAStream getCurrentCUDAStream(DeviceIndex) {
  throw std::runtime_error("stub");
}
CUDA_STUB CUDAStream getDefaultCUDAStream(DeviceIndex) {
  throw std::runtime_error("stub");
}
CUDA_STUB CUDAStream getStreamFromPool(const bool, DeviceIndex) {
  throw std::runtime_error("stub");
}
CUDA_STUB CUDAStream getStreamFromPool(const int, DeviceIndex) {
  throw std::runtime_error("stub");
}
CUDA_STUB void setCurrentCUDAStream(CUDAStream) {
  throw std::runtime_error("stub");
}
CUDA_STUB cudaStream_t CUDAStream::stream() const {
  throw std::runtime_error("stub");
}
CUDA_STUB bool CUDAStream::query() const { throw std::runtime_error("stub"); }
CUDA_STUB void CUDAStream::synchronize() const {
  throw std::runtime_error("stub");
}

namespace CUDACachingAllocator {
CUDA_STUB std::atomic<CUDAAllocator*> allocator;
}  // namespace CUDACachingAllocator

}  // namespace c10::cuda
