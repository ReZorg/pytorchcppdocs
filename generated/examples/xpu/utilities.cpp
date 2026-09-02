// XPU utility functions: torch::xpu::is_available, torch::xpu::device_count,
// torch::xpu::synchronize, torch::xpu::manual_seed, and
// torch::xpu::manual_seed_all.
//
// Adapted from docs: api/xpu/utilities.md

#include <torch/torch.h>

#include <iostream>

int main() {
  // --- Device management ---
  // Availability and device-count queries are safe on any machine.
  if (torch::xpu::is_available()) {
    size_t num_devices = torch::xpu::device_count();
    std::cout << "Found " << num_devices << " XPU device(s)" << std::endl;

    // --- Random number generation ---
    // Set seed for reproducibility on the current XPU device.
    torch::xpu::manual_seed(42);
    // Set seed for all XPU devices.
    torch::xpu::manual_seed_all(42);
    std::cout << "Set XPU manual seed (42) on current device and all devices"
              << std::endl;

    // Do some work on the XPU, then synchronize all streams on device 0.
    auto tensor = torch::randn({64, 64}, torch::device(torch::kXPU));
    auto result = torch::matmul(tensor, tensor);
    torch::xpu::synchronize(0);
    std::cout << "After torch::xpu::synchronize(0): matmul result sum = "
              << result.sum().item<float>() << std::endl;
  } else {
    std::cout << "XPU not available, skipping" << std::endl;
  }

  std::cout << "utilities example finished" << std::endl;
  return 0;
}
