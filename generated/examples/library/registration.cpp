// Examples adapted from _sources/api/library/registration.md.txt
//
// Demonstrates operator registration with the Torch Library API:
//   - TORCH_LIBRARY: define op schemas and (optionally) implementations.
//   - TORCH_LIBRARY_IMPL: register a backend-specific (CPU) implementation.
//   - TORCH_LIBRARY_FRAGMENT: add more definitions to an existing namespace.
//   - Dispatch keys (torch::kCPU / torch::kCUDA / torch::kAutograd /
//     torch::kMeta) and calling a registered op via torch::ops.

#include <torch/library.h>
#include <torch/torch.h>

#include <iostream>

// Kernel implementations for our custom operators.
static torch::Tensor add_impl(const torch::Tensor& self,
                              const torch::Tensor& other) {
  return self + other;
}

static torch::Tensor mul_cpu_impl(const torch::Tensor& self,
                                  const torch::Tensor& other) {
  // CPU kernel for `mul`.
  return self * other;
}

static torch::Tensor mul_cuda_impl(const torch::Tensor& self,
                                   const torch::Tensor& other) {
  // In a real extension this would contain a CUDA kernel launch.
  return self * other;
}

// TORCH_LIBRARY defines the `cppdocs_myops` namespace. Within the block, `m`
// is a torch::Library. We define a schema with its implementation ("add"),
// and a schema-only operator ("mul") whose backend implementations are
// provided separately with m.impl and dispatch keys.
TORCH_LIBRARY(cppdocs_myops, m) {
  // Define with implementation.
  m.def("add(Tensor self, Tensor other) -> Tensor", &add_impl);

  // Define schema only, then provide backend-specific implementations.
  m.def("mul(Tensor self, Tensor other) -> Tensor");
  m.impl("mul", torch::kCPU, &mul_cpu_impl);
  m.impl("mul", torch::kCUDA, &mul_cuda_impl);
}

// TORCH_LIBRARY_IMPL registers an implementation for a dispatch key outside
// the defining TORCH_LIBRARY block. Here we register the CPU kernel for "sub".
TORCH_LIBRARY_IMPL(cppdocs_myops, CPU, m) {
  m.impl("sub", [](const torch::Tensor& self, const torch::Tensor& other) {
    return self - other;
  });
}

// TORCH_LIBRARY_FRAGMENT adds more definitions to the same namespace from a
// separate block (as if from another translation unit). Here we define the
// "sub" schema that the TORCH_LIBRARY_IMPL(CPU) block above implements.
TORCH_LIBRARY_FRAGMENT(cppdocs_myops, m) {
  m.def("sub(Tensor self, Tensor other) -> Tensor");
}

int main() {
  const auto a = torch::tensor({1.0, 2.0, 3.0});
  const auto b = torch::tensor({10.0, 20.0, 30.0});

  // Look up each registered op in the dispatcher and call it through its
  // typed handle (this is how custom ops are invoked from C++).
  auto& dispatcher = c10::Dispatcher::singleton();
  auto add_op =
      dispatcher.findSchemaOrThrow("cppdocs_myops::add", "")
          .typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>();
  auto mul_op =
      dispatcher.findSchemaOrThrow("cppdocs_myops::mul", "")
          .typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>();
  auto sub_op =
      dispatcher.findSchemaOrThrow("cppdocs_myops::sub", "")
          .typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>();

  const auto add_out = add_op.call(a, b);
  const auto mul_out = mul_op.call(a, b);
  const auto sub_out = sub_op.call(a, b);

  std::cout << "add: " << add_out << std::endl;
  std::cout << "mul (dispatched to CPU impl): " << mul_out << std::endl;
  std::cout << "sub (dispatched to CPU impl): " << sub_out << std::endl;

  // The dispatch keys name the backends an implementation can target.
  std::cout << "Dispatch keys: torch::kCPU=" << static_cast<int>(torch::kCPU)
            << " torch::kCUDA=" << static_cast<int>(torch::kCUDA)
            << " torch::kAutograd=" << static_cast<int>(torch::kAutograd)
            << " torch::kMeta=" << static_cast<int>(torch::kMeta) << std::endl;
  return 0;
}
