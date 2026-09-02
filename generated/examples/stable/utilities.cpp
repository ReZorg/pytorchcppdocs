// Examples adapted from _sources/api/stable/utilities.md.txt
//
// Demonstrates the stable-ABI utility functions and types:
//   - torch::stable::accelerator::DeviceGuard and getCurrentDeviceIndex()
//   - torch::stable::accelerator::Stream and getCurrentStream()/nativeHandle()
//   - The ABI-stable C shim API (aoti_torch_get_current_cuda_stream) with
//     TORCH_ERROR_CODE_CHECK
//   - CUDA error-checking macros (STD_CUDA_CHECK /
//   STD_CUDA_KERNEL_LAUNCH_CHECK)
//   - Header-only core types (ScalarType / DeviceType / MemoryFormat / Layout)
//   - STD_TORCH_CHECK (header-only assertion macro)
//   - torch::headeronly::TensorAccessor (HeaderOnlyTensorAccessor)
//   - Header-only dispatch macros (THO_DISPATCH_V2 / THO_DISPATCH_SWITCH /
//     THO_DISPATCH_CASE) and the header-only AT_* type-collection macros
//   - torch::stable::parallel_for and torch::stable::get_num_threads
//
// NOTE on CUDA: this example builds against a CPU-only LibTorch. The CUDA-only
// macros and stream calls are shown guarded so the program compiles and runs
// without a GPU; they are exercised only when a CUDA device is present.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace ho = torch::headeronly;
namespace stable_acc = torch::stable::accelerator;
using torch::stable::Device;
using torch::stable::Tensor;

int main() {
  // --- DeviceGuard + getCurrentDeviceIndex + Stream ---
  // These query the active *accelerator* (e.g. CUDA). On a CPU-only build no
  // accelerator is present, so the underlying shim throws. We wrap them to
  // demonstrate the API while keeping the example runnable without a GPU.
  try {
    std::cout << "current device index: " << stable_acc::getCurrentDeviceIndex()
              << std::endl;
    {
      stable_acc::DeviceGuard guard(0);
      // Operations here run with device 0 current.
      std::cout << "inside DeviceGuard, current device index: "
                << stable_acc::getCurrentDeviceIndex() << std::endl;
    }
    // Previous device is restored here.

    // Get the current stream for the current device and read its id.
    auto stream =
        stable_acc::getCurrentStream(stable_acc::getCurrentDeviceIndex());
    std::cout << "current stream id: " << stream.id() << std::endl;
    // nativeHandle() (PyTorch 2.13+) returns the backend stream handle (a
    // cudaStream_t on CUDA builds).
    std::cout << "current stream nativeHandle: " << stream.nativeHandle()
              << std::endl;

#ifdef USE_CUDA
    // ABI-stable C shim for the current CUDA stream (the 2.9-2.12 path),
    // checked with TORCH_ERROR_CODE_CHECK. Only declared when USE_CUDA.
    void* cuda_stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
        stable_acc::getCurrentDeviceIndex(), &cuda_stream_ptr));
    std::cout << "cuda stream via C shim: " << cuda_stream_ptr << std::endl;
#endif  // USE_CUDA
  } catch (const std::exception& e) {
    std::cout << "accelerator utilities (DeviceGuard / getCurrentDeviceIndex / "
                 "Stream / aoti_torch_get_current_cuda_stream) require a CUDA "
                 "device; none available on this build. ("
              << e.what() << ")" << std::endl;
  }

  // --- CUDA error-checking macros (STD_CUDA_CHECK /
  // STD_CUDA_KERNEL_LAUNCH_CHECK) These wrap CUDA API calls and kernel
  // launches. They require cuda_runtime.h and a CUDA build, so on CPU-only we
  // only document them (they are guarded by TORCH_FEATURE_VERSION >= 2.10 and
  // only meaningful with CUDA).
  std::cout << "STD_CUDA_CHECK / STD_CUDA_KERNEL_LAUNCH_CHECK are CUDA-only; "
               "shown in source, not executed on CPU-only build."
            << std::endl;

  // --- Header-only core types ---
  auto dtype = ho::ScalarType::Float;
  auto device_type = ho::DeviceType::CPU;
  auto memory_format = ho::MemoryFormat::Contiguous;
  auto layout = ho::Layout::Strided;
  std::cout << "header-only types: dtype=" << ho::toString(dtype)
            << " device_type=" << static_cast<int>(device_type)
            << " layout=" << static_cast<int>(layout) << std::endl;
  (void)memory_format;

  // --- STD_TORCH_CHECK (header-only assertion, throws std::runtime_error) ---
  int some_value = 5;
  STD_TORCH_CHECK(some_value == 5, "Error message with ", some_value,
                  " interpolation");
  std::cout << "STD_TORCH_CHECK passed for value " << some_value << std::endl;

  // --- TensorAccessor (HeaderOnlyTensorAccessor) ---
  // Build a 2-D float tensor and access its elements through an accessor.
  auto t =
      torch::stable::empty({2, 3}, ho::ScalarType::Float, ho::Layout::Strided,
                           Device(ho::DeviceType::CPU));
  torch::stable::fill_(t, 0.0);
  auto sizes = t.sizes();
  auto strides = t.strides();
  ho::HeaderOnlyTensorAccessor<float, 2> accessor(
      static_cast<float*>(t.mutable_data_ptr()), sizes.data(), strides.data());
  accessor[1][2] = 42.0f;
  std::cout << "TensorAccessor[1][2] = " << accessor[1][2] << std::endl;

  // --- Header-only dispatch macros ---
  // THO_DISPATCH_V2 resolves scalar_t from a dtype and dispatches to the
  // matching case. Here we read element [0] with the resolved C++ type.
  double first = 0.0;
  THO_DISPATCH_V2(t.scalar_type(),  // resolved as scalar_t
                  "read_first", AT_WRAP(([&]() {
                    auto* data = static_cast<scalar_t*>(t.mutable_data_ptr());
                    first = static_cast<double>(data[0]);
                  })),
                  AT_EXPAND(AT_ALL_TYPES));
  std::cout << "THO_DISPATCH_V2 read first element: " << first << std::endl;

  // THO_DISPATCH_SWITCH / THO_DISPATCH_CASE (v1-style) over floating types.
  std::string resolved;
  THO_DISPATCH_SWITCH(
      t.scalar_type(), "dtype_name",
      THO_DISPATCH_CASE(ho::ScalarType::Float, [&] { resolved = "Float"; })
          THO_DISPATCH_CASE(ho::ScalarType::Double,
                            [&] { resolved = "Double"; }));
  std::cout << "THO_DISPATCH_SWITCH resolved dtype: " << resolved << std::endl;

  // --- parallel_for / get_num_threads ---
  std::cout << "get_num_threads: " << torch::stable::get_num_threads()
            << std::endl;
  auto par =
      torch::stable::empty({8}, ho::ScalarType::Float, ho::Layout::Strided,
                           Device(ho::DeviceType::CPU));
  torch::stable::parallel_for(0, 8, 1, [&par](int64_t begin, int64_t end) {
    auto* data = static_cast<float*>(par.mutable_data_ptr());
    for (int64_t i = begin; i < end; ++i) {
      data[i] = static_cast<float>(i * i);
    }
  });
  const float* pp = par.const_data_ptr<float>();
  std::cout << "parallel_for squares: [";
  for (int64_t i = 0; i < par.numel(); ++i) {
    std::cout << pp[i] << (i + 1 < par.numel() ? ", " : "");
  }
  std::cout << "]" << std::endl;

  return 0;
}
