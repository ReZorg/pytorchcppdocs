// Examples adapted from _sources/api/library/versioning.md.txt
//
// Demonstrates the LibTorch version macros. These are available in
// PyTorch >= 1.8.0 and identify the version of LibTorch in use:
//   - TORCH_VERSION_MAJOR / TORCH_VERSION_MINOR / TORCH_VERSION_PATCH
//   - TORCH_VERSION (full version string, e.g. "2.14.0")

#include <torch/torch.h>

#include <iostream>

int main() {
  // Reconstruct the version from its numeric parts.
  std::cout << "PyTorch version from parts: " << TORCH_VERSION_MAJOR << "."
            << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;

  // The full version string literal.
  std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
  return 0;
}
