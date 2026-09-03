// Installing the C++ distribution: the tutorial's minimal example-app,
// which creates a torch::Tensor and prints it, plus a check that the
// documented CMake pattern found LibTorch correctly.
//
// Adapted from docs: installing.md

#include <torch/torch.h>

#include <iostream>

int main() {
  // The example program from installing.md: create a tensor and print it.
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  // The tutorial's CMake configuration (find_package(Torch REQUIRED),
  // ${TORCH_CXX_FLAGS}, C++ standard, MSVC DLL copy) is exactly what
  // generated/CMakeLists.txt and examples/minimal/CMakeLists.txt implement.
  std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR << "."
            << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;
  std::cout << "installing example finished" << std::endl;
  return 0;
}
