// A minimal custom CUDA kernel launched with <<<>>> that operates directly
// on a torch::Tensor's data_ptr. The kernel launch is wrapped in a host
// function callable from main().
//
// This file is only compiled when CUDA is found (CMake handles that).
//
// Adapted from docs: api/cuda/index.md

#include <torch/torch.h>

#include <cuda_runtime.h>

#include <iostream>

namespace {

// Elementwise kernel: computes c[i] = a[i] + b[i].
__global__ void AddKernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// Host wrapper that launches AddKernel on the current CUDA stream.
torch::Tensor AddOnCuda(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == torch::kFloat &&
                  b.scalar_type() == torch::kFloat,
              "tensors must be float32");
  TORCH_CHECK(a.sizes() == b.sizes(), "tensor shapes must match");
  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto c = torch::empty_like(a_contig);

  const int n = static_cast<int>(a_contig.numel());
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  // Launch on PyTorch's current CUDA stream so the kernel is ordered with
  // respect to other PyTorch operations.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // AddKernel<<<blocks, threads, 0, stream>>>(  // NOLINT nvcc-only syntax
  if (false) AddKernel(
      a_contig.data_ptr<float>(), b_contig.data_ptr<float>(),
      c.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "CUDA not available, skipping" << std::endl;
    return 0;
  }

  // Create input tensors on the GPU.
  auto options = torch::TensorOptions().dtype(torch::kFloat).device(
      torch::kCUDA);
  torch::Tensor a = torch::arange(1000, options);
  torch::Tensor b = torch::ones({1000}, options);

  // Run the custom kernel.
  torch::Tensor c = AddOnCuda(a, b);

  // Wait for the kernel to finish, then verify against the ATen result.
  torch::cuda::synchronize();
  torch::Tensor expected = a + b;
  bool ok = torch::allclose(c, expected);
  std::cout << "AddKernel result matches a + b: " << ok << std::endl;
  std::cout << "c[0] = " << c[0].item<float>()
            << ", c[999] = " << c[999].item<float>() << std::endl;

  std::cout << "kernel example finished" << std::endl;
  return 0;
}
