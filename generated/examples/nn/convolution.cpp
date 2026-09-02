// Convolution layers: Conv1d/2d/3d and ConvTranspose1d/2d/3d with their
// Options (stride, padding, dilation, groups, bias) and forward shapes.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  // --- Conv1d: [N, C_in, L] -> [N, C_out, L_out] ---
  auto conv1 = torch::nn::Conv1d(
      torch::nn::Conv1dOptions(4, 8, /*kernel_size=*/3)
          .stride(1)
          .padding(1)
          .bias(true));
  auto input1d = torch::randn({2, 4, 16});
  PrintSizes("Conv1d output", conv1->forward(input1d));

  // --- Conv2d: [N, C_in, H, W] -> [N, C_out, H_out, W_out] ---
  auto conv2 = torch::nn::Conv2d(
      torch::nn::Conv2dOptions(3, 64, /*kernel_size=*/3)
          .stride(1)
          .padding(1)
          .bias(true));
  auto input2d = torch::randn({2, 3, 8, 8});
  PrintSizes("Conv2d output", conv2->forward(input2d));

  // Conv2d with stride 2, dilation and grouped convolution.
  auto conv2_strided = torch::nn::Conv2d(
      torch::nn::Conv2dOptions(8, 16, 3).stride(2).padding(1).dilation(1));
  PrintSizes("Conv2d stride=2 output",
             conv2_strided->forward(torch::randn({2, 8, 8, 8})));

  // Depthwise convolution: groups == in_channels.
  auto conv2_depthwise = torch::nn::Conv2d(
      torch::nn::Conv2dOptions(8, 8, 3).padding(1).groups(8));
  PrintSizes("Conv2d depthwise output",
             conv2_depthwise->forward(torch::randn({2, 8, 8, 8})));

  // --- Conv3d: [N, C_in, D, H, W] ---
  auto conv3 = torch::nn::Conv3d(
      torch::nn::Conv3dOptions(2, 4, /*kernel_size=*/3).padding(1));
  auto input3d = torch::randn({1, 2, 6, 6, 6});
  PrintSizes("Conv3d output", conv3->forward(input3d));

  // --- ConvTranspose1d: upsampling over length ---
  auto convt1 = torch::nn::ConvTranspose1d(
      torch::nn::ConvTranspose1dOptions(8, 4, /*kernel_size=*/4)
          .stride(2)
          .padding(1));
  PrintSizes("ConvTranspose1d output",
             convt1->forward(torch::randn({2, 8, 8})));

  // --- ConvTranspose2d: upsampling over 2D spatial dims ---
  auto convt2 = torch::nn::ConvTranspose2d(
      torch::nn::ConvTranspose2dOptions(64, 32, /*kernel_size=*/4)
          .stride(2)
          .padding(1));
  PrintSizes("ConvTranspose2d output",
             convt2->forward(torch::randn({2, 64, 4, 4})));

  // --- ConvTranspose3d: upsampling over 3D volume ---
  auto convt3 = torch::nn::ConvTranspose3d(
      torch::nn::ConvTranspose3dOptions(4, 2, /*kernel_size=*/4)
          .stride(2)
          .padding(1));
  PrintSizes("ConvTranspose3d output",
             convt3->forward(torch::randn({1, 4, 4, 4, 4})));

  // Learnable parameters of a convolution layer are ->weight / ->bias.
  std::cout << "Conv2d weight sizes: " << conv2->weight.sizes()
            << ", bias sizes: " << conv2->bias.sizes() << std::endl;

  return 0;
}
