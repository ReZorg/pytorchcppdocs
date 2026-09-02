// Validation-only weak stubs for device libs absent from the CPU-only wheel.
#include <c10/core/Stream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <stdexcept>
#define STUB __attribute__((weak))
extern "C" {
STUB cudaError_t cudaGetLastError() { return cudaSuccess; }
STUB cudaError_t cudaGetDeviceCount(int* c) { *c = 0; return cudaSuccess; }
STUB cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
STUB cudaError_t cudaSetDevice(int) { return cudaSuccess; }
STUB cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
STUB cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
STUB cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*) { return cudaSuccess; }
STUB cudaError_t cudaStreamGetPriority(cudaStream_t, int*) { return cudaSuccess; }
STUB cudaError_t cudaDeviceGetStreamPriorityRange(int*, int*) { return cudaSuccess; }
STUB cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
STUB cudaError_t cudaStreamCreateWithPriority(cudaStream_t*, unsigned int, int) { return cudaSuccess; }
STUB cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
STUB cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) { return cudaSuccess; }
STUB cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int) { return cudaSuccess; }
STUB cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
STUB cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
STUB cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
STUB cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
STUB cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) { return cudaSuccess; }
STUB cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
}
namespace c10::cuda {
STUB DeviceIndex device_count() noexcept { return 0; }
STUB DeviceIndex device_count_ensure_non_zero() { return 0; }
STUB DeviceIndex current_device() { return -1; }
STUB void set_device(DeviceIndex, const bool) {}
STUB cudaError_t GetDeviceCount(int* c) { *c = 0; return cudaSuccess; }
STUB cudaError_t GetDevice(DeviceIndex* d) { *d = 0; return cudaSuccess; }
STUB cudaError_t SetDevice(DeviceIndex, const bool) { return cudaSuccess; }
STUB cudaError_t MaybeSetDevice(DeviceIndex) { return cudaSuccess; }
STUB DeviceIndex ExchangeDevice(DeviceIndex) { return -1; }
STUB DeviceIndex MaybeExchangeDevice(DeviceIndex) { return -1; }
STUB void device_synchronize() {}
STUB CUDAErrorLogCapture::CUDAErrorLogCapture() noexcept {}
STUB void c10_cuda_check_implementation(int, const char*, const char*, unsigned int, bool, CUDAErrorLogCapture*) {}
STUB CUDAStream getCurrentCUDAStream(DeviceIndex) { throw std::runtime_error("stub"); }
STUB CUDAStream getDefaultCUDAStream(DeviceIndex) { throw std::runtime_error("stub"); }
STUB CUDAStream getStreamFromPool(const bool, DeviceIndex) { throw std::runtime_error("stub"); }
STUB CUDAStream getStreamFromPool(const int, DeviceIndex) { throw std::runtime_error("stub"); }
STUB void setCurrentCUDAStream(CUDAStream) { throw std::runtime_error("stub"); }
STUB cudaStream_t CUDAStream::stream() const { throw std::runtime_error("stub"); }
STUB bool CUDAStream::query() const { throw std::runtime_error("stub"); }
STUB void CUDAStream::synchronize() const { throw std::runtime_error("stub"); }
namespace CUDACachingAllocator { STUB std::atomic<CUDAAllocator*> allocator; }
}  // namespace c10::cuda
namespace c10::xpu {
STUB XPUStream getCurrentXPUStream(DeviceIndex) { throw std::runtime_error("stub"); }
STUB XPUStream getStreamFromPool(const bool, DeviceIndex) { throw std::runtime_error("stub"); }
STUB XPUStream getStreamFromPool(const int, DeviceIndex) { throw std::runtime_error("stub"); }
STUB void setCurrentXPUStream(XPUStream) { throw std::runtime_error("stub"); }
STUB void syncStreamsOnDevice(DeviceIndex) { throw std::runtime_error("stub"); }
STUB int XPUStream::priority() const { throw std::runtime_error("stub"); }
STUB sycl::queue& XPUStream::queue() const { throw std::runtime_error("stub"); }
STUB c10::DeviceIndex device_count() noexcept { return 0; }
STUB c10::DeviceIndex current_device() { return -1; }
STUB void set_device(c10::DeviceIndex) {}
STUB c10::DeviceIndex exchangeDevice(c10::DeviceIndex) { return -1; }
STUB c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex) { return -1; }
STUB void device_synchronize() {}
STUB sycl::device& get_raw_device(DeviceIndex) { throw std::runtime_error("stub"); }
STUB sycl::context& get_device_context() { throw std::runtime_error("stub"); }
STUB void getDeviceProperties(DeviceProp*, DeviceIndex) {}
STUB void init_device_cache(DeviceIndex) {}
}  // namespace c10::xpu
