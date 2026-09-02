// Examples for the "Tensor Class" page (api/aten/tensor).
// Covers at::Tensor methods, TensorOptions, Scalar, ScalarType, Layout,
// and c10::DeviceGuard.
#include <torch/torch.h>

#include <iostream>
#include <utility>
#include <vector>

int main() {
  // Default constructor: creates an undefined tensor.
  at::Tensor undefined;
  std::cout << "undefined.defined(): " << undefined.defined() << std::endl;

  // Example from the docs.
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2});
  at::Tensor c = a + b.to(at::kInt);
  std::cout << "a + b.to(kInt):\n" << c << std::endl;

  at::Tensor t = at::randn({2, 3, 4});

  // Shape and metadata accessors.
  std::cout << "dim(): " << t.dim() << std::endl;
  std::cout << "size(1): " << t.size(1) << std::endl;
  std::cout << "sizes(): " << t.sizes() << std::endl;
  std::cout << "strides(): " << t.strides() << std::endl;
  std::cout << "scalar_type() == at::kFloat: "
            << (t.scalar_type() == at::kFloat) << std::endl;
  std::cout << "device(): " << t.device() << std::endl;
  std::cout << "is_cuda(): " << t.is_cuda() << std::endl;
  std::cout << "is_cpu(): " << t.is_cpu() << std::endl;
  std::cout << "requires_grad(): " << t.requires_grad() << std::endl;
  std::cout << "data_ptr() != nullptr: " << (t.data_ptr() != nullptr)
            << std::endl;

  // requires_grad_() sets gradient tracking in place and returns the tensor.
  at::Tensor leaf = at::ones({2, 2}).requires_grad_(true);
  std::cout << "requires_grad after requires_grad_(true): "
            << leaf.requires_grad() << std::endl;
  leaf.requires_grad_(false);
  std::cout << "requires_grad after requires_grad_(false): "
            << leaf.requires_grad() << std::endl;

  // to(dtype) converts the data type.
  at::Tensor doubles = t.to(at::kDouble);
  std::cout << "to(kDouble).scalar_type() == at::kDouble: "
            << (doubles.scalar_type() == at::kDouble) << std::endl;

  // contiguous() returns a tensor with contiguous memory layout.
  at::Tensor transposed = t.transpose(0, 1);
  std::cout << "transposed.is_contiguous(): " << transposed.is_contiguous()
            << std::endl;
  at::Tensor contig = transposed.contiguous();
  std::cout << "contiguous().is_contiguous(): " << contig.is_contiguous()
            << std::endl;

  // TensorOptions: bundles dtype, device, layout, and requires_grad.
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(at::kFloat)
                                  .device(at::kCPU)
                                  .layout(at::kStrided)
                                  .requires_grad(false);
  at::Tensor configured = at::zeros({3, 4}, options);
  std::cout << "TensorOptions-configured tensor: " << configured.sizes()
            << " " << configured.dtype() << " on " << configured.device()
            << std::endl;

  if (torch::cuda::is_available()) {
    // to(device) moves a tensor between devices.
    at::Tensor gpu = t.to(at::Device(at::kCUDA, 0));
    std::cout << "to(kCUDA).is_cuda(): " << gpu.is_cuda() << std::endl;

    // DeviceGuard sets the current device for its scope.
    {
      c10::DeviceGuard guard(at::Device(at::kCUDA, 0));
      at::Tensor on_guard_device = at::zeros({2, 2});
      std::cout << "inside guard: " << on_guard_device.device() << std::endl;
    }
    // Previous device is restored here.
  }

  // at::Scalar: a dynamically typed single number.
  at::Scalar int_scalar(int64_t(42));
  at::Scalar double_scalar(3.5);
  std::cout << "Scalar(int64_t).isIntegral(false): "
            << int_scalar.isIntegral(/*includeBool=*/false)
            << ", to<int64_t>: " << int_scalar.to<int64_t>() << std::endl;
  std::cout << "Scalar(double).isFloatingPoint(): "
            << double_scalar.isFloatingPoint()
            << ", to<double>: " << double_scalar.to<double>() << std::endl;
  std::cout << "isIntegral(/*includeBool=*/true) on double: "
            << double_scalar.isIntegral(true) << std::endl;

  // ScalarType enumerators and their convenience constants.
  std::vector<std::pair<at::ScalarType, const char*>> scalar_types = {
      {at::kByte, "Byte"},   {at::kChar, "Char"},     {at::kShort, "Short"},
      {at::kInt, "Int"},     {at::kLong, "Long"},     {at::kHalf, "Half"},
      {at::kFloat, "Float"}, {at::kDouble, "Double"}, {at::kBool, "Bool"},
      {at::kBFloat16, "BFloat16"},
  };
  for (const auto& entry : scalar_types) {
    at::Tensor scalar_typed = at::empty({1}, entry.first);
    std::cout << "ScalarType " << entry.second
              << " -> itemsize: " << scalar_typed.element_size() << std::endl;
  }

  // Layout enumerators.
  std::cout << "at::kStrided: " << at::kStrided << std::endl;
  at::Tensor sparse =
      at::empty({2, 3}, at::TensorOptions().layout(at::kSparse));
  std::cout << "sparse tensor layout: " << sparse.layout() << std::endl;
  at::Tensor sparse_csr = at::sparse_csr_tensor(
      at::tensor({0, 1, 1}, at::kLong), at::tensor({0}, at::kLong),
      at::tensor({1.0f}), /*size=*/{2, 3},
      at::TensorOptions().dtype(at::kFloat));
  std::cout << "sparse CSR layout == at::kSparseCsr: "
            << (sparse_csr.layout() == at::kSparseCsr) << std::endl;
  std::cout << "at::kSparseCsc enumerator: " << at::kSparseCsc << std::endl;

  return 0;
}
