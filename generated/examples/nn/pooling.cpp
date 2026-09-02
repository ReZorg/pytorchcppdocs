// Pooling layers: MaxPool/AvgPool/AdaptiveMaxPool/AdaptiveAvgPool in
// 1d/2d/3d variants, plus FractionalMaxPool, MaxUnpool and LPPool.
#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  // --- MaxPool 1d/2d/3d ---
  auto maxpool1 =
      torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(2).stride(2));
  PrintSizes("MaxPool1d output",
             maxpool1->forward(torch::randn({2, 3, 8})));

  auto maxpool2 =
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
  PrintSizes("MaxPool2d output",
             maxpool2->forward(torch::randn({2, 3, 8, 8})));

  auto maxpool3 =
      torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2).stride(2));
  PrintSizes("MaxPool3d output",
             maxpool3->forward(torch::randn({1, 3, 4, 4, 4})));

  // --- AvgPool 1d/2d/3d ---
  auto avgpool1 = torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(2).stride(2));
  PrintSizes("AvgPool1d output",
             avgpool1->forward(torch::randn({2, 3, 8})));

  auto avgpool2 = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2));
  PrintSizes("AvgPool2d output",
             avgpool2->forward(torch::randn({2, 3, 8, 8})));

  auto avgpool3 = torch::nn::AvgPool3d(torch::nn::AvgPool3dOptions(2).stride(2));
  PrintSizes("AvgPool3d output",
             avgpool3->forward(torch::randn({1, 3, 4, 4, 4})));

  // --- AdaptiveAvgPool 1d/2d/3d: fixed output size regardless of input ---
  auto adap_avg1 =
      torch::nn::AdaptiveAvgPool1d(torch::nn::AdaptiveAvgPool1dOptions(4));
  PrintSizes("AdaptiveAvgPool1d output",
             adap_avg1->forward(torch::randn({2, 3, 9})));

  // Output will always be 7x7 regardless of input size.
  auto adap_avg2 =
      torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({7, 7}));
  PrintSizes("AdaptiveAvgPool2d output",
             adap_avg2->forward(torch::randn({2, 3, 11, 13})));

  auto adap_avg3 = torch::nn::AdaptiveAvgPool3d(
      torch::nn::AdaptiveAvgPool3dOptions({2, 3, 3}));
  PrintSizes("AdaptiveAvgPool3d output",
             adap_avg3->forward(torch::randn({1, 3, 5, 6, 7})));

  // --- AdaptiveMaxPool 1d/2d/3d ---
  auto adap_max1 =
      torch::nn::AdaptiveMaxPool1d(torch::nn::AdaptiveMaxPool1dOptions(4));
  PrintSizes("AdaptiveMaxPool1d output",
             adap_max1->forward(torch::randn({2, 3, 9})));

  auto adap_max2 = torch::nn::AdaptiveMaxPool2d(
      torch::nn::AdaptiveMaxPool2dOptions({3, 3}));
  PrintSizes("AdaptiveMaxPool2d output",
             adap_max2->forward(torch::randn({2, 3, 8, 8})));

  auto adap_max3 = torch::nn::AdaptiveMaxPool3d(
      torch::nn::AdaptiveMaxPool3dOptions({2, 2, 2}));
  PrintSizes("AdaptiveMaxPool3d output",
             adap_max3->forward(torch::randn({1, 3, 4, 5, 6})));

  // --- FractionalMaxPool2d / FractionalMaxPool3d ---
  auto frac2 = torch::nn::FractionalMaxPool2d(
      torch::nn::FractionalMaxPool2dOptions(2).output_size(
          std::vector<int64_t>({4, 4})));
  PrintSizes("FractionalMaxPool2d output",
             frac2->forward(torch::randn({1, 3, 8, 8})));

  auto frac3 = torch::nn::FractionalMaxPool3d(
      torch::nn::FractionalMaxPool3dOptions(2).output_size(
          std::vector<int64_t>({2, 2, 2})));
  PrintSizes("FractionalMaxPool3d output",
             frac3->forward(torch::randn({1, 3, 4, 4, 4})));

  // --- MaxUnpool 1d/2d/3d: partial inverse of MaxPool via indices ---
  // forward_with_indices returns pooled values plus the argmax indices,
  // which MaxUnpool uses to place values back into unpooled positions.
  auto pool1 =
      torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(2).stride(2));
  auto unpool1 =
      torch::nn::MaxUnpool1d(torch::nn::MaxUnpool1dOptions(2).stride(2));
  auto input1 = torch::randn({2, 3, 8});
  auto p1 = pool1->forward_with_indices(input1);
  PrintSizes("MaxUnpool1d output",
             unpool1->forward(std::get<0>(p1), std::get<1>(p1)));

  auto pool2 =
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
  auto unpool2 =
      torch::nn::MaxUnpool2d(torch::nn::MaxUnpool2dOptions(2).stride(2));
  auto input2 = torch::randn({2, 3, 8, 8});
  auto p2 = pool2->forward_with_indices(input2);
  PrintSizes("MaxUnpool2d output",
             unpool2->forward(std::get<0>(p2), std::get<1>(p2)));

  auto pool3 =
      torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2).stride(2));
  auto unpool3 =
      torch::nn::MaxUnpool3d(torch::nn::MaxUnpool3dOptions(2).stride(2));
  auto input3 = torch::randn({1, 3, 4, 4, 4});
  auto p3 = pool3->forward_with_indices(input3);
  PrintSizes("MaxUnpool3d output",
             unpool3->forward(std::get<0>(p3), std::get<1>(p3)));

  // --- LPPool 1d/2d/3d: power-average pooling ---
  auto lp1 = torch::nn::LPPool1d(torch::nn::LPPool1dOptions(2.0, 2).stride(2));
  PrintSizes("LPPool1d output", lp1->forward(torch::randn({2, 3, 8})));

  auto lp2 = torch::nn::LPPool2d(torch::nn::LPPool2dOptions(2.0, 2).stride(2));
  PrintSizes("LPPool2d output", lp2->forward(torch::randn({2, 3, 8, 8})));

  auto lp3 = torch::nn::LPPool3d(torch::nn::LPPool3dOptions(2.0, 2).stride(2));
  PrintSizes("LPPool3d output", lp3->forward(torch::randn({1, 3, 4, 4, 4})));

  return 0;
}
