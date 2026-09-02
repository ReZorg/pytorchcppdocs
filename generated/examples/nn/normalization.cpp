// Normalization layers: BatchNorm1d/2d/3d, InstanceNorm1d/2d/3d, LayerNorm,
// GroupNorm and LocalResponseNorm.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  // --- BatchNorm1d/2d/3d: normalize across the batch dim ---
  auto bn1 = torch::nn::BatchNorm1d(
      torch::nn::BatchNorm1dOptions(8)
          .eps(1e-5)
          .momentum(0.1)
          .affine(true)
          .track_running_stats(true));
  bn1->train();  // training mode updates running stats
  PrintSizes("BatchNorm1d output", bn1->forward(torch::randn({4, 8})));

  auto bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3));
  bn2->train();
  PrintSizes("BatchNorm2d output", bn2->forward(torch::randn({2, 3, 8, 8})));

  auto bn3 = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(3));
  bn3->train();
  PrintSizes("BatchNorm3d output",
             bn3->forward(torch::randn({1, 3, 4, 4, 4})));

  // eval() uses the running stats instead of batch statistics.
  bn1->eval();
  PrintSizes("BatchNorm1d eval output", bn1->forward(torch::randn({4, 8})));

  // --- InstanceNorm1d/2d/3d: normalize each sample/channel independently ---
  auto in1 = torch::nn::InstanceNorm1d(torch::nn::InstanceNorm1dOptions(4));
  PrintSizes("InstanceNorm1d output", in1->forward(torch::randn({2, 4, 8})));

  auto in2 = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(4));
  PrintSizes("InstanceNorm2d output",
             in2->forward(torch::randn({2, 4, 8, 8})));

  auto in3 = torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(4));
  PrintSizes("InstanceNorm3d output",
             in3->forward(torch::randn({1, 4, 4, 4, 4})));

  // --- LayerNorm: normalize over the given trailing shape ---
  auto ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions({16}));
  PrintSizes("LayerNorm output", ln->forward(torch::randn({2, 5, 16})));

  // --- GroupNorm: normalize within groups of channels ---
  auto gn = torch::nn::GroupNorm(
      torch::nn::GroupNormOptions(/*num_groups=*/4, /*num_channels=*/8));
  PrintSizes("GroupNorm output", gn->forward(torch::randn({2, 8, 8, 8})));

  // --- LocalResponseNorm: cross-channel lateral inhibition ---
  auto lrn = torch::nn::LocalResponseNorm(
      torch::nn::LocalResponseNormOptions(/*size=*/5));
  PrintSizes("LocalResponseNorm output",
             lrn->forward(torch::randn({2, 8, 8, 8})));

  return 0;
}
