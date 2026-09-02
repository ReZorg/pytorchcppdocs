// CUDA device and stream guards: c10::cuda::CUDAGuard,
// c10::cuda::CUDAStreamGuard, c10::cuda::OptionalCUDAGuard,
// c10::cuda::OptionalCUDAStreamGuard, and c10::cuda::CUDAMultiStreamGuard.
//
// Adapted from docs: api/cuda/guards.md

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <iostream>

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "CUDA not available, skipping" << std::endl;
    return 0;
  }

  const int64_t num_devices =
      static_cast<int64_t>(torch::cuda::device_count());
  std::cout << "CUDA device count: " << num_devices << std::endl;
  // Use device 1 as the "second" device when present, else reuse device 0.
  const int64_t second = num_devices > 1 ? 1 : 0;

  // --- c10::cuda::CUDAGuard ---
  // RAII guard that sets the current CUDA device and restores the previous
  // device when it goes out of scope.
  {
    c10::cuda::CUDAGuard guard(second);  // Switch to device `second`.
    // All CUDA operations here run on device `second`.
    auto tensor = torch::zeros({2, 2}, torch::device(torch::kCUDA));
    std::cout << "Inside CUDAGuard: current device = "
              << c10::cuda::current_device()
              << ", tensor device = " << tensor.device() << std::endl;
  }
  // Previous device is restored.
  std::cout << "After CUDAGuard: current device = "
            << c10::cuda::current_device() << std::endl;

  // --- c10::cuda::CUDAStreamGuard ---
  // RAII guard that sets the current stream (and its device) and restores
  // the previous stream on scope exit.
  auto stream = c10::cuda::getStreamFromPool();
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    // Operations here use the specified stream.
    std::cout << "Inside CUDAStreamGuard: current stream id = "
              << c10::cuda::getCurrentCUDAStream().id()
              << " (pool stream id = " << stream.id() << ")" << std::endl;
  }
  // Previous stream is restored.
  std::cout << "After CUDAStreamGuard: current stream id = "
            << c10::cuda::getCurrentCUDAStream().id() << std::endl;

  // --- c10::cuda::OptionalCUDAGuard ---
  // The guard only switches device if a device was set.
  const bool use_cuda = true;
  c10::cuda::OptionalCUDAGuard optional_guard;
  if (use_cuda) {
    optional_guard.set_device(c10::Device(c10::kCUDA, 0));
  }
  std::cout << "OptionalCUDAGuard active device = "
            << c10::cuda::current_device() << std::endl;
  optional_guard.reset();

  // --- c10::cuda::OptionalCUDAStreamGuard ---
  // The stream is switched only if one was explicitly set.
  c10::cuda::OptionalCUDAStreamGuard optional_stream_guard;
  optional_stream_guard.reset_stream(stream);
  std::cout << "OptionalCUDAStreamGuard current stream id = "
            << c10::cuda::getCurrentCUDAStream().id() << std::endl;
  optional_stream_guard.reset();

  // --- c10::cuda::CUDAMultiStreamGuard ---
  // Sets the current stream on several devices at once. It does not change
  // the current device index, only the stream on each stream's device.
  c10::cuda::CUDAStream stream0 = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStream stream1 = c10::cuda::getStreamFromPool(false, second);
  {
    c10::cuda::CUDAMultiStreamGuard multi_guard({stream0, stream1});
    // stream0 is current on device 0, stream1 on device `second`.
    std::cout << "CUDAMultiStreamGuard: stream on device 0 = "
              << c10::cuda::getCurrentCUDAStream(0).id()
              << ", stream on device " << second << " = "
              << c10::cuda::getCurrentCUDAStream(second).id() << std::endl;
  }
  // Both streams restored.

  std::cout << "guards example finished" << std::endl;
  return 0;
}
