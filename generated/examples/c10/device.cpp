// Examples for the "Device and DeviceType" page (api/c10/device).
// Covers c10::Device construction and inspection, plus the DeviceType
// enumerators and their convenience constants.
#include <torch/torch.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <utility>
#include <vector>

int main() {
  // Example from the docs: construct devices and query them.
  c10::Device cpu_device(c10::kCPU);
  c10::Device cuda_device(c10::kCUDA, 0);  // CUDA device 0

  std::cout << "cpu_device: " << cpu_device << std::endl;
  std::cout << "cpu_device.is_cpu(): " << cpu_device.is_cpu() << std::endl;
  std::cout << "cuda_device.is_cuda(): " << cuda_device.is_cuda() << std::endl;

  if (torch::cuda::is_available()) {
    c10::Device real_cuda(c10::kCUDA, 0);
    std::cout << "Using CUDA device " << real_cuda.index() << std::endl;
    std::cout << "has_index(): " << real_cuda.has_index() << std::endl;
    std::cout << "type() == c10::kCUDA: " << (real_cuda.type() == c10::kCUDA)
              << std::endl;
    // Tensors can be created directly on the device.
    torch::Tensor on_gpu = torch::zeros({2, 2}, real_cuda);
    std::cout << "tensor device: " << on_gpu.device() << std::endl;
  } else {
    std::cout << "CUDA not available; constructing Device(kCUDA, 0) is fine "
                 "but tensors cannot be moved there."
              << std::endl;
  }

  // Device without an explicit index.
  c10::Device indexless(c10::kCPU);
  std::cout << "Device(kCPU).has_index(): " << indexless.has_index()
            << std::endl;

  // DeviceType enumerators and their convenience constants.
  std::vector<std::pair<c10::DeviceType, const char*>> device_types = {
      {c10::kCPU, "CPU"},
      {c10::kCUDA, "CUDA"},
      {c10::kHIP, "HIP"},
      {c10::kXLA, "XLA"},
      {c10::kVulkan, "Vulkan"},
      {c10::kMetal, "Metal"},
      {c10::kXPU, "XPU"},
      {c10::kMPS, "MPS"},
      {c10::kMeta, "Meta"},
      {c10::kHPU, "HPU"},
      {c10::kLazy, "Lazy"},
      {c10::kIPU, "IPU"},
      {c10::kMTIA, "MTIA"},
      {c10::kPrivateUse1, "PrivateUse1"},
  };
  for (const auto& entry : device_types) {
    std::cout << "DeviceType " << entry.second
              << " -> " << c10::Device(entry.first) << std::endl;
  }

  // Meta tensors carry shape information but no data.
  torch::Tensor meta = torch::empty({2, 3}, c10::Device(c10::kMeta));
  std::cout << "meta tensor sizes: " << meta.sizes()
            << ", is_meta: " << meta.is_meta() << std::endl;

  return 0;
}
