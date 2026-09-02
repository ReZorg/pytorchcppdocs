// Examples adapted from _sources/api/stable/registration.md.txt
//
// Demonstrates the stable-ABI operator registration macros, which are the
// binary-compatible equivalents of TORCH_LIBRARY / TORCH_LIBRARY_IMPL:
//   - STABLE_TORCH_LIBRARY: define operator schemas under a namespace.
//   - STABLE_TORCH_LIBRARY_IMPL: register a boxed kernel for a dispatch key.
//   - STABLE_TORCH_LIBRARY_FRAGMENT: extend an existing namespace.
//   - TORCH_BOX: wrap an unboxed kernel into the stable boxed calling
//     convention required by STABLE_TORCH_LIBRARY_IMPL.
//
// Kernels registered via the stable ABI operate on torch::stable::Tensor.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <cstdint>
#include <iostream>

using torch::stable::Tensor;

// Unboxed CPU kernel for `cppdocs_stable::reshape_op`: reshape the input to a
// 1-D tensor of `size` elements via the stable reshape op.
static Tensor my_reshape_kernel(const Tensor& input, int64_t size) {
  return torch::stable::reshape(input, {size});
}

// Unboxed CPU kernel for `cppdocs_stable::combine_op`: return a clone of `a`.
static Tensor my_combine_kernel(const Tensor& a, const Tensor& b) {
  (void)b;
  return torch::stable::clone(a);
}

// STABLE_TORCH_LIBRARY defines the schemas in the `cppdocs_stable` namespace.
STABLE_TORCH_LIBRARY(cppdocs_stable, m) {
  m.def("reshape_op(Tensor input, int size) -> Tensor");
  m.def("combine_op(Tensor a, Tensor b) -> Tensor");
}

// STABLE_TORCH_LIBRARY_IMPL registers the CPU implementation. Every kernel
// registered here must be boxed with TORCH_BOX.
STABLE_TORCH_LIBRARY_IMPL(cppdocs_stable, CPU, m) {
  m.impl("reshape_op", TORCH_BOX(&my_reshape_kernel));
}

// STABLE_TORCH_LIBRARY_FRAGMENT extends the existing namespace with an
// additional definition (as if from another translation unit).
STABLE_TORCH_LIBRARY_FRAGMENT(cppdocs_stable, m) {
  m.def("scale_op(Tensor a, Tensor b) -> Tensor");
}

// A second IMPL block provides the CPU kernel for the fragment-defined op.
STABLE_TORCH_LIBRARY_IMPL(cppdocs_stable, CPU, m) {
  m.impl("scale_op", TORCH_BOX(&my_combine_kernel));
}

int main() {
  // Build an input tensor with the stable API (2.9+): a 2x3 float tensor on
  // CPU, filled with a known value.
  auto input = torch::stable::empty(
      {2, 3}, torch::headeronly::ScalarType::Float,
      torch::headeronly::Layout::Strided,
      torch::stable::Device(torch::headeronly::DeviceType::CPU), false,
      torch::headeronly::MemoryFormat::Contiguous);
  torch::stable::fill_(input, 1.5);

  // Invoke the kernels (these are the same functions registered via TORCH_BOX
  // with STABLE_TORCH_LIBRARY_IMPL above).
  auto reshaped = my_reshape_kernel(input, 6);
  std::cout << "reshape_op: numel=" << reshaped.numel()
            << " dim=" << reshaped.dim() << std::endl;

  auto combined = my_combine_kernel(input, input);
  std::cout << "combine_op: sizes=[" << combined.sizes()[0] << ", "
            << combined.sizes()[1] << "]" << std::endl;

  std::cout << "STABLE_TORCH_LIBRARY / STABLE_TORCH_LIBRARY_IMPL / "
               "STABLE_TORCH_LIBRARY_FRAGMENT / TORCH_BOX registered and ran."
            << std::endl;
  return 0;
}
