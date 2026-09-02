// CUDA streams for asynchronous GPU execution: c10::cuda::CUDAStream,
// getDefaultCUDAStream, getCurrentCUDAStream, setCurrentCUDAStream,
// getStreamFromPool, stream guards, and stream synchronization.
//
// Adapted from docs: api/cuda/streams.md

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "CUDA not available, skipping" << std::endl;
    return 0;
  }

  // --- Acquiring CUDA streams ---
  // Default stream for the current device, where most computation occurs.
  c10::cuda::CUDAStream default_stream = c10::cuda::getDefaultCUDAStream();
  std::cout << "Default stream id: " << default_stream.id() << std::endl;

  // Normal-priority stream from the pool (round-robin allocation).
  c10::cuda::CUDAStream pool_stream = c10::cuda::getStreamFromPool();
  std::cout << "Pool stream id: " << pool_stream.id() << std::endl;

  // High-priority stream from the pool.
  c10::cuda::CUDAStream high_prio =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true);
  std::cout << "High-priority stream id: " << high_prio.id() << std::endl;

  // Stream for a specific device.
  c10::cuda::CUDAStream dev0_stream =
      c10::cuda::getStreamFromPool(false, /*device=*/0);
  std::cout << "Pool stream for device 0: device index = "
            << dev0_stream.device_index() << std::endl;

  // Current stream (may differ from the default if changed with guards).
  c10::cuda::CUDAStream current_stream = c10::cuda::getCurrentCUDAStream();
  std::cout << "Current stream id: " << current_stream.id() << std::endl;

  // --- Setting CUDA streams with setCurrentCUDAStream ---
  torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(torch::kCUDA));

  // Get a new stream and set it as current.
  c10::cuda::CUDAStream my_stream = c10::cuda::getStreamFromPool();
  c10::cuda::setCurrentCUDAStream(my_stream);

  // Operations now use my_stream.
  tensor0.sum();
  std::cout << "After setCurrentCUDAStream: current stream id = "
            << c10::cuda::getCurrentCUDAStream().id() << std::endl;

  // Restore the default stream.
  c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());

  // --- Setting CUDA streams with CUDAStreamGuard (recommended) ---
  {
    c10::cuda::CUDAStreamGuard guard(my_stream);
    // Operations use my_stream within this scope.
    tensor0.sum();
    std::cout << "Inside CUDAStreamGuard: current stream id = "
              << c10::cuda::getCurrentCUDAStream().id() << std::endl;
  }
  // Stream automatically restored to the default.

  // --- Multi-device stream management ---
  const int64_t num_devices = static_cast<int64_t>(torch::cuda::device_count());
  const int64_t second = num_devices > 1 ? 1 : 0;
  std::cout << "Using devices 0 and " << second
            << " (device count: " << num_devices << ")" << std::endl;

  // Acquire streams for different devices.
  c10::cuda::CUDAStream stream0 = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStream stream1 = c10::cuda::getStreamFromPool(false, second);

  // Set the current streams on each device.
  c10::cuda::setCurrentCUDAStream(stream0);
  c10::cuda::setCurrentCUDAStream(stream1);

  // Create tensors on device 0.
  tensor0 = torch::ones({2, 2}, torch::device(torch::kCUDA));
  tensor0.sum();  // Uses stream0.

  // Switch to the second device.
  {
    c10::cuda::CUDAGuard device_guard(second);
    torch::Tensor tensor1 = torch::ones({2, 2}, torch::device(torch::kCUDA));
    tensor1.sum();  // Uses stream1.
  }

  // --- CUDAMultiStreamGuard: set streams on multiple devices at once ---
  {
    // Note: CUDAMultiStreamGuard does not change the current device index.
    c10::cuda::CUDAMultiStreamGuard multi_guard({stream0, stream1});
    tensor0.sum();  // Uses stream0 on device 0.
    std::cout << "CUDAMultiStreamGuard: current device = "
              << c10::cuda::current_device() << ", stream on device 0 = "
              << c10::cuda::getCurrentCUDAStream(0).id() << std::endl;
  }
  // Both streams restored to their previous values.

  // --- Multi-device stream handling patterns ---
  // Create stream vectors on device 0.
  std::vector<c10::cuda::CUDAStream> streams0 = {
      c10::cuda::getDefaultCUDAStream(), c10::cuda::getStreamFromPool()};
  c10::cuda::setCurrentCUDAStream(streams0[0]);

  // Create a stream vector on the second device using a CUDAGuard.
  std::vector<c10::cuda::CUDAStream> streams1;
  {
    c10::cuda::CUDAGuard device_guard(second);
    streams1.push_back(c10::cuda::getDefaultCUDAStream());
    streams1.push_back(c10::cuda::getStreamFromPool());
  }
  c10::cuda::setCurrentCUDAStream(streams1[0]);

  // Pattern 1: CUDAGuard changes the current device only, not streams.
  {
    c10::cuda::CUDAGuard device_guard(second);
    // Current device is `second`, current stream there is still streams1[0].
    std::cout << "Pattern 1 (CUDAGuard): device = "
              << c10::cuda::current_device()
              << ", stream = " << c10::cuda::getCurrentCUDAStream().id()
              << std::endl;
  }

  // Pattern 2: CUDAStreamGuard changes both current device and stream.
  {
    c10::cuda::CUDAStreamGuard stream_guard(streams1[1]);
    // Current device is `second`, current stream is streams1[1].
    std::cout << "Pattern 2 (CUDAStreamGuard): device = "
              << c10::cuda::current_device()
              << ", stream = " << c10::cuda::getCurrentCUDAStream().id()
              << std::endl;
  }
  // Restored to device 0, stream streams0[0].

  // Pattern 3: CUDAMultiStreamGuard sets streams on multiple devices at once.
  {
    c10::cuda::CUDAMultiStreamGuard multi_guard({streams0[1], streams1[1]});
    // Current device unchanged (still 0); stream on device 0 is streams0[1],
    // stream on device `second` is streams1[1].
    std::cout << "Pattern 3 (CUDAMultiStreamGuard): device = "
              << c10::cuda::current_device() << ", stream on device 0 = "
              << c10::cuda::getCurrentCUDAStream(0).id() << std::endl;
  }
  // Streams restored to streams0[0] and streams1[0].

  // --- Stream synchronization ---
  // Queue asynchronous work on a non-default stream, then block until the
  // stream has completed all queued operations.
  {
    c10::cuda::CUDAStreamGuard guard(my_stream);
    auto result = torch::matmul(tensor0, tensor0);
    my_stream.synchronize();
    std::cout << "After synchronize: matmul result sum = "
              << result.sum().item<float>() << std::endl;
  }
  default_stream.synchronize();

  std::cout << "streams example finished" << std::endl;
  return 0;
}
