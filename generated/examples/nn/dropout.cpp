// Dropout layers: Dropout, Dropout2d, Dropout3d, AlphaDropout,
// FeatureAlphaDropout, and train()/eval() behavior.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  // --- Dropout: randomly zeros elements with probability p (train only) ---
  auto dropout = torch::nn::Dropout(torch::nn::DropoutOptions(0.5));
  auto flat = torch::ones({4, 8});

  dropout->train();
  auto dropped_train = dropout->forward(flat);
  PrintSizes("Dropout train output", dropped_train);
  std::cout << "Dropout train zero count: "
            << (dropped_train == 0).sum().item<int64_t>() << std::endl;

  // In eval mode dropout is disabled: output equals input.
  dropout->eval();
  auto dropped_eval = dropout->forward(flat);
  std::cout << "Dropout eval equals input: "
            << (dropped_eval.allclose(flat) ? "true" : "false") << std::endl;

  // --- Dropout2d / Dropout3d: zero entire channels (spatial dropout) ---
  auto dropout2d = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.3));
  auto dropout3d = torch::nn::Dropout3d(torch::nn::Dropout3dOptions(0.3));
  dropout2d->train();
  dropout3d->train();
  PrintSizes("Dropout2d output",
             dropout2d->forward(torch::ones({2, 4, 8, 8})));
  PrintSizes("Dropout3d output",
             dropout3d->forward(torch::ones({2, 4, 4, 4, 4})));

  // --- AlphaDropout: keeps mean/variance (use with SELU activations) ---
  auto alpha = torch::nn::AlphaDropout(torch::nn::AlphaDropoutOptions(0.2));
  alpha->train();
  PrintSizes("AlphaDropout output", alpha->forward(torch::randn({4, 8})));

  // --- FeatureAlphaDropout: alpha dropout over whole feature channels ---
  auto feature_alpha = torch::nn::FeatureAlphaDropout(
      torch::nn::FeatureAlphaDropoutOptions(0.2));
  feature_alpha->train();
  PrintSizes("FeatureAlphaDropout output",
             feature_alpha->forward(torch::randn({2, 4, 8, 8})));

  // Dropout inside a Sequential model toggles with model->train()/eval().
  torch::nn::Sequential model(torch::nn::Linear(8, 8), torch::nn::Dropout(0.5),
                              torch::nn::Linear(8, 4));
  model->eval();
  PrintSizes("Sequential with Dropout eval output",
             model->forward(torch::randn({2, 8})));

  return 0;
}
