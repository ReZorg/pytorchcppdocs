// CUDA utility functions: torch::cuda::is_available, cudnn_is_available,
// device_count, manual_seed, manual_seed_all, synchronize, plus the c10/at
// device queries, device properties, and stream management utilities.
//
// Adapted from docs: api/cuda/utilities.md

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

// Device-property queries and library handles are only declared when
// PyTorch was built with CUDA support (USE_CUDA is defined by its headers).
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include <iostream>

int main() {
  // --- Availability queries (safe on CPU-only builds and machines) ---
  std::cout << "torch::cuda::is_available: " << torch::cuda::is_available()
            << std::endl;
  std::cout << "torch::cuda::cudnn_is_available: "
            << torch::cuda::cudnn_is_available() << std::endl;

  // --- Device management ---
  size_t num_devices = torch::cuda::device_count();
  std::cout << "torch::cuda::device_count: " << num_devices << std::endl;

  // --- Random number generation ---
  // Seeding the CUDA RNG is a no-op when no CUDA device is present.
  torch::cuda::manual_seed(42);
  torch::cuda::manual_seed_all(42);
  std::cout << "Set CUDA manual seed (42) on current device and all devices"
            << std::endl;

  if (!torch::cuda::is_available()) {
    std::cout << "CUDA not available, skipping device-specific utilities"
              << std::endl;
    return 0;
  }

  // Get the index of the current CUDA device.
  c10::DeviceIndex current = c10::cuda::current_device();
  std::cout << "c10::cuda::current_device: " << current << std::endl;

#ifdef USE_CUDA
  // --- Device properties ---
  // Query properties of the current device.
  cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
  std::cout << "Device: " << props->name << std::endl;
  std::cout << "Compute capability: " << props->major << "." << props->minor
            << std::endl;

  // Query a specific device.
  cudaDeviceProp* dev0_props = at::cuda::getDeviceProperties(0);
  std::cout << "Device 0 multiprocessors: " << dev0_props->multiProcessorCount
            << std::endl;

  // Check whether device 0 can directly access the memory of a peer device.
  if (num_devices > 1) {
    bool can_access = at::cuda::canDeviceAccessPeer(0, 1);
    std::cout << "canDeviceAccessPeer(0, 1): " << can_access << std::endl;
  }

  // Warp size of the current device.
  int warp_size = at::cuda::warp_size();
  std::cout << "at::cuda::warp_size: " << warp_size << std::endl;

  // --- Library handles ---
  // Handles for CUDA math libraries on the current device and stream,
  // primarily useful when writing custom CUDA kernels that call cuBLAS or
  // cuSPARSE directly.
  cublasHandle_t blas_handle = at::cuda::getCurrentCUDABlasHandle();
  std::cout << "getCurrentCUDABlasHandle: "
            << (blas_handle != nullptr ? "valid" : "null") << std::endl;

  cublasLtHandle_t blaslt_handle = at::cuda::getCurrentCUDABlasLtHandle();
  std::cout << "getCurrentCUDABlasLtHandle: "
            << (blaslt_handle != nullptr ? "valid" : "null") << std::endl;

  cusparseHandle_t sparse_handle = at::cuda::getCurrentCUDASparseHandle();
  std::cout << "getCurrentCUDASparseHandle: "
            << (sparse_handle != nullptr ? "valid" : "null") << std::endl;

  cusolverDnHandle_t solver_handle = at::cuda::getCurrentCUDASolverDnHandle();
  std::cout << "getCurrentCUDASolverDnHandle: "
            << (solver_handle != nullptr ? "valid" : "null") << std::endl;
#endif

  // --- Stream management ---
  // Create and set a custom stream.
  auto stream = c10::cuda::getStreamFromPool();
  c10::cuda::setCurrentCUDAStream(stream);
  std::cout << "Current stream id after setCurrentCUDAStream: "
            << c10::cuda::getCurrentCUDAStream().id() << std::endl;

  // Get the default stream.
  auto default_stream = c10::cuda::getDefaultCUDAStream();
  std::cout << "Default stream id: " << default_stream.id() << std::endl;

  // Restore the default stream before doing more work.
  c10::cuda::setCurrentCUDAStream(default_stream);

  // --- Synchronization ---
  // Do some work on the GPU, then block until the device has finished.
  auto tensor = torch::randn({256, 256}, torch::device(torch::kCUDA));
  auto result = torch::matmul(tensor, tensor);
  torch::cuda::synchronize();   // Synchronize the current device.
  torch::cuda::synchronize(0);  // Synchronize device 0 explicitly.
  std::cout << "After torch::cuda::synchronize: matmul result sum = "
            << result.sum().item<float>() << std::endl;

  std::cout << "utilities example finished" << std::endl;
  return 0;
}
