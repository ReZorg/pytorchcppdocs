// Examples for the "Device Guards" page (api/c10/guards).
// Covers c10::DeviceGuard and c10::OptionalDeviceGuard, the device-agnostic
// RAII guards that restore the previous device when they go out of scope.
#include <torch/torch.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <iostream>

int main() {
  // The guards work on any backend; use CUDA when it is available and CPU
  // otherwise. On CPU this exercises the same guard code paths.
  const bool use_cuda = torch::cuda::is_available();
  c10::Device device = use_cuda ? c10::Device(c10::kCUDA, 0) : c10::kCPU;

  if (use_cuda) {
    // DeviceGuard: sets the current device for its scope and restores the
    // previous device on destruction.
    {
      c10::DeviceGuard guard(c10::Device(c10::kCUDA, 0));
      // All operations here run on CUDA device 0.
      torch::Tensor inside = torch::zeros({2, 2});
      std::cout << "inside DeviceGuard: " << inside.device() << std::endl;
    }
    // Previous device is restored.
    std::cout << "DeviceGuard scope exited; previous device restored."
              << std::endl;
  } else {
    c10::DeviceGuard guard(c10::kCPU);
    torch::Tensor inside = torch::zeros({2, 2});
    std::cout << "DeviceGuard on CPU device: " << inside.device()
              << std::endl;
  }

  // OptionalDeviceGuard: only restores the device if one was set.
  const bool use_gpu = use_cuda;
  {
    c10::OptionalDeviceGuard opt_guard;
    if (use_gpu) {
      opt_guard.reset_device(c10::Device(c10::kCUDA, 0));
      torch::Tensor inside = torch::zeros({2, 2});
      std::cout << "inside OptionalDeviceGuard: " << inside.device()
                << std::endl;
    }
    // Guard restores the device only if reset_device was called.
  }
  std::cout << "OptionalDeviceGuard (use_gpu=" << use_gpu << ") done."
            << std::endl;

  // OptionalDeviceGuard can also be constructed directly from a device.
  {
    c10::OptionalDeviceGuard direct_guard(device);
    std::cout << "OptionalDeviceGuard constructed with " << device
              << std::endl;
  }

  return 0;
}
