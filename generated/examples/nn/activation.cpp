// Activation functions: ReLU family, ELU family, GELU/SiLU/Mish,
// Sigmoid/Tanh, Softmax variants, GLU, shrinkages and Threshold.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  auto x = torch::randn({2, 8});  // shared 2D input
  auto x4 = torch::randn({2, 3, 8, 8});  // shared 4D input

  // --- ReLU family ---
  auto relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(false));
  PrintSizes("ReLU output", relu->forward(x));

  auto leaky = torch::nn::LeakyReLU(
      torch::nn::LeakyReLUOptions().negative_slope(0.01));
  PrintSizes("LeakyReLU output", leaky->forward(x));

  // PReLU learns its negative slope (num_parameters slopes).
  auto prelu = torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(8));
  PrintSizes("PReLU output", prelu->forward(x));

  auto rrelu = torch::nn::RReLU(torch::nn::RReLUOptions().lower(0.125).upper(0.333));
  rrelu->train();  // randomized slope only in training mode
  PrintSizes("RReLU output", rrelu->forward(x));

  auto relu6 = torch::nn::ReLU6();
  PrintSizes("ReLU6 output", relu6->forward(x * 10));  // capped at 6

  // --- ELU family ---
  auto elu = torch::nn::ELU(torch::nn::ELUOptions().alpha(1.0));
  PrintSizes("ELU output", elu->forward(x));

  auto selu = torch::nn::SELU();
  PrintSizes("SELU output", selu->forward(x));

  auto celu = torch::nn::CELU(torch::nn::CELUOptions().alpha(1.0));
  PrintSizes("CELU output", celu->forward(x));

  // --- Modern smooth activations ---
  auto gelu = torch::nn::GELU();
  PrintSizes("GELU output", gelu->forward(x));

  auto silu = torch::nn::SiLU();
  PrintSizes("SiLU output", silu->forward(x));

  auto mish = torch::nn::Mish();
  PrintSizes("Mish output", mish->forward(x));

  // --- Classic bounded activations ---
  auto sigmoid = torch::nn::Sigmoid();
  auto sig = sigmoid->forward(x);
  PrintSizes("Sigmoid output", sig);
  std::cout << "Sigmoid in (0, 1): "
            << ((sig.min().item<float>() > 0 && sig.max().item<float>() < 1)
                    ? "true"
                    : "false")
            << std::endl;

  auto tanh = torch::nn::Tanh();
  PrintSizes("Tanh output", tanh->forward(x));

  // --- GLU: splits input in half along dim, a * sigmoid(b) ---
  auto glu = torch::nn::GLU(torch::nn::GLUOptions().dim(-1));
  PrintSizes("GLU output", glu->forward(x));  // [2, 4]

  // --- LogSigmoid: numerically stable log(sigmoid(x)) ---
  auto logsigmoid = torch::nn::LogSigmoid();
  PrintSizes("LogSigmoid output", logsigmoid->forward(x));

  // --- Softmax variants ---
  auto softmax = torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim=*/1));
  auto probs = softmax->forward(x);
  PrintSizes("Softmax output", probs);
  std::cout << "Softmax row sums to 1: "
            << (probs.sum(1).allclose(torch::ones({2})) ? "true" : "false")
            << std::endl;

  // Softmax over channels at each spatial location of a 4D input.
  auto softmax2d = torch::nn::Softmax2d();
  PrintSizes("Softmax2d output", softmax2d->forward(x4));

  auto logsoftmax = torch::nn::LogSoftmax(/*dim=*/1);
  PrintSizes("LogSoftmax output", logsoftmax->forward(x));

  auto softmin = torch::nn::Softmin(torch::nn::SoftminOptions(/*dim=*/1));
  PrintSizes("Softmin output", softmin->forward(x));

  // --- Softplus / shrinkages / sign-like ---
  auto softplus = torch::nn::Softplus(
      torch::nn::SoftplusOptions().beta(1).threshold(20));
  PrintSizes("Softplus output", softplus->forward(x));

  auto softshrink = torch::nn::Softshrink(0.5);
  PrintSizes("Softshrink output", softshrink->forward(x));

  auto softsign = torch::nn::Softsign();
  PrintSizes("Softsign output", softsign->forward(x));

  auto hardshrink = torch::nn::Hardshrink(0.5);
  PrintSizes("Hardshrink output", hardshrink->forward(x));

  auto hardtanh =
      torch::nn::Hardtanh(torch::nn::HardtanhOptions().min_val(-1).max_val(1));
  PrintSizes("Hardtanh output", hardtanh->forward(x * 5));

  auto tanhshrink = torch::nn::Tanhshrink();
  PrintSizes("Tanhshrink output", tanhshrink->forward(x));

  // --- Threshold: replace values <= threshold with value ---
  auto threshold = torch::nn::Threshold(
      torch::nn::ThresholdOptions(/*threshold=*/0.0, /*value=*/0.0));
  PrintSizes("Threshold output", threshold->forward(x));

  return 0;
}
