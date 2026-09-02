// Linear layers: Linear, Bilinear, Identity, Flatten and Unflatten.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  // --- Linear: y = xW^T + b over the last dimension ---
  auto linear =
      torch::nn::Linear(torch::nn::LinearOptions(784, 256).bias(true));
  auto flattened = torch::randn({4, 784});
  PrintSizes("Linear output", linear->forward(flattened));
  std::cout << "Linear weight sizes: " << linear->weight.sizes()
            << ", bias sizes: " << linear->bias.sizes() << std::endl;

  // Works on higher-rank inputs too: only the last dim is transformed.
  auto linear32 = torch::nn::Linear(torch::nn::LinearOptions(16, 8));
  PrintSizes("Linear on [2, 3, 16] output",
             linear32->forward(torch::randn({2, 3, 16})));

  // --- Bilinear: y = x1^T W x2 + b ---
  auto bilinear = torch::nn::Bilinear(
      torch::nn::BilinearOptions(/*in1_features=*/8, /*in2_features=*/6,
                                 /*out_features=*/4));
  PrintSizes("Bilinear output",
             bilinear->forward(torch::randn({2, 8}), torch::randn({2, 6})));

  // --- Identity: pass-through layer (useful for skip connections) ---
  auto identity = torch::nn::Identity();
  auto anything = torch::randn({2, 3, 4});
  PrintSizes("Identity output", identity->forward(anything));

  // --- Flatten: collapse dims [start_dim, end_dim) into one ---
  auto flatten =
      torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1));
  auto conv_out = torch::randn({2, 3, 8, 8});
  auto flat_out = flatten->forward(conv_out);
  PrintSizes("Flatten output", flat_out);

  // --- Unflatten: expand one dim back into a shape ---
  auto unflatten =
      torch::nn::Unflatten(torch::nn::UnflattenOptions(1, {3, 8, 8}));
  PrintSizes("Unflatten output", unflatten->forward(flat_out));

  // Typical pattern: conv features -> Flatten -> Linear.
  torch::nn::Sequential head(
      torch::nn::Flatten(),
      torch::nn::Linear(torch::nn::LinearOptions(3 * 8 * 8, 10)));
  PrintSizes("Conv->Flatten->Linear output", head->forward(conv_out));

  return 0;
}
