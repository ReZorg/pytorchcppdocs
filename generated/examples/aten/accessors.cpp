// Examples for the "Tensor Accessors" page (api/aten/accessors).
// Covers CPU accessors and CUDA packed accessors for element-wise access
// without dynamic dispatch overhead.
#include <torch/torch.h>

#include <iostream>

int main() {
  // CPU accessor: validates element type and dimensionality once, then
  // provides efficient []-style access.
  torch::Tensor foo = torch::rand({12, 12});
  torch::TensorAccessor<float, 2> foo_a = foo.accessor<float, 2>();

  float trace = 0;
  for (int64_t i = 0; i < foo_a.size(0); i++) {
    trace += foo_a[i][i];
  }
  std::cout << "trace via accessor: " << trace << std::endl;
  std::cout << "accessor sizes: [" << foo_a.size(0) << ", " << foo_a.size(1)
            << "]" << std::endl;

  // Accessors work for other scalar types and dimensionalities too.
  torch::Tensor ints = torch::arange(24, torch::kInt64).reshape({2, 3, 4});
  auto ints_a = ints.accessor<int64_t, 3>();
  std::cout << "ints_a[1][2][3]: " << ints_a[1][2][3] << std::endl;

  if (torch::cuda::is_available()) {
    // CUDA packed accessors copy the sizes/strides metadata by value so a
    // kernel launched on the device can use them. From host code we can
    // still read the metadata; element access happens inside the kernel.
    torch::Tensor gpu_foo = torch::rand({12, 12}).cuda();
    auto gpu_packed = gpu_foo.packed_accessor64<float, 2>();
    std::cout << "packed_accessor64 sizes: [" << gpu_packed.size(0) << ", "
              << gpu_packed.size(1) << "]" << std::endl;

    // PackedTensorAccessor32/packed_accessor32 use 32-bit indexing, which
    // is faster on CUDA but may overflow for very large tensors.
    auto gpu_packed32 = gpu_foo.packed_accessor32<float, 2>();
    std::cout << "packed_accessor32 sizes: [" << gpu_packed32.size(0) << ", "
              << gpu_packed32.size(1) << "]" << std::endl;
  } else {
    std::cout << "CUDA not available; skipping packed accessor example."
              << std::endl;
  }

  return 0;
}
