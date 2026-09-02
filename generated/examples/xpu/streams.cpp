// XPU streams for asynchronous execution on Intel GPUs:
// c10::xpu::XPUStream, getCurrentXPUStream, setCurrentXPUStream,
// getStreamFromPool, and syncStreamsOnDevice.
//
// Adapted from docs: api/xpu/streams.md

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>

#include <iostream>

int main() {
  // All XPU runtime usage is guarded behind an availability check so this
  // example compiles and runs (exiting 0) on machines without an Intel GPU.
  if (!torch::xpu::is_available()) {
    std::cout << "XPU not available, skipping" << std::endl;
    return 0;
  }

  // --- c10::xpu::XPUStream ---
  // Get the current XPU stream.
  c10::xpu::XPUStream current = c10::xpu::getCurrentXPUStream();
  std::cout << "Current XPU stream id: " << current.id()
            << ", device index: " << current.device_index() << std::endl;

  // --- c10::xpu::getStreamFromPool ---
  // Create a new stream from the pool.
  c10::xpu::XPUStream new_stream = c10::xpu::getStreamFromPool();
  std::cout << "Pool stream id: " << new_stream.id() << std::endl;

  // High-priority stream for a specific device.
  c10::xpu::XPUStream high_prio =
      c10::xpu::getStreamFromPool(/*isHighPriority=*/true, /*device=*/0);
  std::cout << "High-priority pool stream id: " << high_prio.id()
            << ", priority: " << high_prio.priority() << std::endl;

  // --- c10::xpu::setCurrentXPUStream ---
  // Make the pool stream current; queue some work on it.
  c10::xpu::setCurrentXPUStream(new_stream);
  auto tensor = torch::ones({128, 128}, torch::device(torch::kXPU));
  auto result = torch::matmul(tensor, tensor);
  std::cout << "Current stream id after setCurrentXPUStream: "
            << c10::xpu::getCurrentXPUStream().id() << std::endl;

  // --- Scoped stream switch ---
  // Save/restore the current stream around a scope that uses another stream.
  c10::xpu::XPUStream saved_stream = c10::xpu::getCurrentXPUStream();
  c10::xpu::setCurrentXPUStream(high_prio);
  std::cout << "On high-priority stream: current stream id = "
            << c10::xpu::getCurrentXPUStream().id() << std::endl;
  c10::xpu::setCurrentXPUStream(saved_stream);
  std::cout << "Restored stream id = "
            << c10::xpu::getCurrentXPUStream().id() << std::endl;

  // --- Stream synchronization ---
  // Block until a single stream has completed all queued operations.
  new_stream.synchronize();
  std::cout << "After new_stream.synchronize(): matmul result sum = "
            << result.sum().item<float>() << std::endl;

  // --- c10::xpu::syncStreamsOnDevice ---
  // Synchronize all streams on a device at once.
  c10::xpu::syncStreamsOnDevice(0);
  std::cout << "syncStreamsOnDevice(0) completed" << std::endl;

  std::cout << "streams example finished" << std::endl;
  return 0;
}
