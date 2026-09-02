// Examples for the "Tensor Creation" page (api/aten/creation).
// Covers the factory functions, TensorOptions configuration, from_blob,
// to() conversion, Scalars, and zero-dimensional tensors.
#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  // Specifying a size: a bare size creates a vector, an IntArrayRef in
  // curly braces creates a multi-dimensional tensor.
  torch::Tensor vector = torch::ones(5);
  std::cout << "torch::ones(5): " << vector << std::endl;

  torch::Tensor three_d = torch::randn({3, 4, 5});
  assert(three_d.sizes() == std::vector<int64_t>({3, 4, 5}));
  std::cout << "randn({3, 4, 5}).size(2): " << three_d.size(2) << std::endl;

  // An std::vector<int64_t> also works as the size.
  std::vector<int64_t> shape = {2, 6};
  torch::Tensor from_vec_shape = torch::zeros(shape);
  std::cout << "zeros(vector{2, 6}).sizes(): " << from_vec_shape.sizes()
            << std::endl;

  // The available factory functions.
  std::cout << "zeros:\n" << torch::zeros({2, 3}) << std::endl;
  std::cout << "ones:\n" << torch::ones({2, 3}) << std::endl;
  std::cout << "empty sizes: " << torch::empty({2, 3}).sizes() << std::endl;
  std::cout << "full:\n" << torch::full({2, 3}, /*value=*/7.5) << std::endl;
  std::cout << "rand[0, 1):\n" << torch::rand({2, 3}) << std::endl;
  std::cout << "randn (standard normal):\n" << torch::randn({2, 3})
            << std::endl;

  // Function-specific parameters precede the size.
  torch::Tensor randint_high = torch::randint(/*high=*/10, {5, 5});
  torch::Tensor randint_range = torch::randint(/*low=*/3, /*high=*/10, {5, 5});
  std::cout << "randint(10) max < 10: "
            << (randint_high.max().item<int64_t>() < 10) << std::endl;
  std::cout << "randint(3, 10) min >= 3: "
            << (randint_range.min().item<int64_t>() >= 3) << std::endl;

  // arange/linspace/logspace/eye/randperm need no size argument; the size
  // is fully determined by the function-specific arguments.
  std::cout << "arange: " << torch::arange(0, 10, 2) << std::endl;
  std::cout << "linspace: " << torch::linspace(0, 1, 5) << std::endl;
  std::cout << "logspace: " << torch::logspace(0, 2, 3) << std::endl;
  std::cout << "eye:\n" << torch::eye(3) << std::endl;
  std::cout << "randperm: " << torch::randperm(6) << std::endl;

  // Configuring properties with TensorOptions. The CPU equivalent of the
  // CUDA example on the doc page:
  torch::TensorOptions options = torch::TensorOptions()
                                     .dtype(torch::kFloat64)
                                     .layout(torch::kStrided)
                                     .device(torch::kCPU)
                                     .requires_grad(true);
  torch::Tensor configured = torch::full({3, 4}, /*value=*/123, options);
  assert(configured.dtype() == torch::kFloat64);
  assert(configured.layout() == torch::kStrided);
  assert(configured.device().type() == torch::kCPU);
  assert(configured.requires_grad());
  std::cout << "full({3, 4}, 123, options) dtype: " << configured.dtype()
            << std::endl;

  if (torch::cuda::is_available()) {
    torch::Tensor cuda_tensor =
        torch::full({3, 4}, /*value=*/123,
                    torch::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(torch::kCUDA, 0)
                        .requires_grad(true));
    std::cout << "CUDA tensor device index: " << cuda_tensor.device().index()
              << std::endl;
  }

  // Defaults: a 32-bit float, strided, CPU tensor without gradients.
  torch::Tensor defaulted = torch::randn({3, 4});
  std::cout << "default dtype == kFloat32: "
            << (defaulted.dtype() == torch::kFloat32) << std::endl;

  // Shorthand free functions returning refined TensorOptions.
  torch::Tensor a =
      torch::ones(10, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor b = torch::ones(10, torch::dtype(torch::kFloat32));
  torch::Tensor chained =
      torch::ones(10, torch::dtype(torch::kFloat32).layout(torch::kStrided));
  // Implicit construction from a single ScalarType value.
  torch::Tensor implicit = torch::ones(10, torch::kFloat32);
  std::cout << "shorthand dtypes equal: "
            << (a.dtype() == b.dtype() && b.dtype() == chained.dtype() &&
                chained.dtype() == implicit.dtype())
            << std::endl;

  // torch::requires_grad() / torch::device() shorthand builders.
  torch::Tensor grad_tensor = torch::randn({2, 2}, torch::requires_grad(true));
  std::cout << "requires_grad(true) shorthand: "
            << grad_tensor.requires_grad() << std::endl;
  torch::Tensor cpu_device = torch::ones({2}, torch::device(torch::kCPU));
  std::cout << "device(kCPU) shorthand: " << cpu_device.device() << std::endl;

  // Rust-style dtype shorthands such as kF32.
  torch::Tensor rust_style = torch::ones({2}, torch::kF32);
  std::cout << "kF32 == kFloat32: " << (rust_style.dtype() == torch::kFloat32)
            << std::endl;

  // Using externally created data with from_blob. ATen does not own the
  // memory, so the tensor cannot be resized.
  float data[] = {1, 2, 3, 4, 5, 6};
  torch::Tensor blob = torch::from_blob(data, {2, 3});
  std::cout << "from_blob:\n" << blob << std::endl;
  std::cout << "from_blob shares memory: "
            << (blob.data_ptr() == static_cast<void*>(data)) << std::endl;

  // Tensor conversion with to(): creates a new tensor, never in-place.
  torch::Tensor source = torch::randn({2, 3}).to(torch::kInt64);
  torch::Tensor float_tensor = source.to(torch::kFloat32);
  std::cout << "to(kFloat32) dtype: " << float_tensor.dtype() << std::endl;
  torch::Tensor back = float_tensor.to(torch::kCPU, /*non_blocking=*/false);
  std::cout << "round-trip device: " << back.device() << std::endl;
  if (torch::cuda::is_available()) {
    torch::Tensor gpu_tensor = float_tensor.to(torch::kCUDA);
    torch::Tensor gpu0_tensor =
        float_tensor.to(torch::Device(torch::kCUDA, 0));
    torch::Tensor async_tensor =
        gpu_tensor.to(torch::kCPU, /*non_blocking=*/true);
    std::cout << "gpu tensor: " << gpu_tensor.device()
              << ", device-0 copy: " << gpu0_tensor.device()
              << ", async copy back: " << async_tensor.device() << std::endl;
  }

  // Scalars: dynamically typed numbers, implicitly constructed from C++
  // number types; used by functions such as addmm and sum.
  torch::Tensor mat = torch::randn({2, 2});
  torch::Tensor mat1 = torch::randn({2, 2});
  torch::Tensor mat2 = torch::randn({2, 2});
  // Note: in LibTorch the generated signature puts the tensors first
  // and the Scalar arguments (beta, alpha) last.
  torch::Tensor r = torch::addmm(mat, mat1, mat2, /*beta=*/1.0, /*alpha=*/0.5);
  std::cout << "addmm(mat, mat1, mat2, 1.0, 0.5):\n" << r << std::endl;
  // sum() returns a zero-dimensional tensor holding the scalar value.
  torch::Tensor total = torch::sum(mat);
  std::cout << "torch::sum (0-dim tensor): " << total.item<double>()
            << std::endl;

  // Zero-dimensional tensors hold a single value and can reference
  // elements in larger tensors.
  torch::Tensor matrix = torch::rand({10, 20});
  matrix[1][2] = 4;  // matrix[1][2] is a zero-dimensional tensor
  std::cout << "matrix[1][2] dim: " << matrix[1][2].dim()
            << ", value: " << matrix[1][2].item<float>() << std::endl;

  return 0;
}
