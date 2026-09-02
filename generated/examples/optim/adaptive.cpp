// Adaptive learning rate optimizers: torch::optim::Adam, AdamW, Adagrad, and
// RMSprop with their corresponding Options structs.
//
// Adapted from docs: api/optim/adaptive.md

#include <torch/torch.h>

#include <iostream>
#include <memory>

namespace {

struct TinyNet : torch::nn::Module {
  TinyNet() {
    fc1 = register_module("fc1", torch::nn::Linear(4, 8));
    fc2 = register_module("fc2", torch::nn::Linear(8, 2));
  }

  torch::Tensor forward(torch::Tensor x) {
    return fc2->forward(torch::relu(fc1->forward(x)));
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

double TrainSteps(torch::optim::Optimizer& optimizer,
                  TinyNet& model, const torch::Tensor& data,
                  const torch::Tensor& target, int steps) {
  double loss_value = 0.0;
  for (int step = 0; step < steps; ++step) {
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::mse_loss(output, target);
    loss.backward();
    optimizer.step();
    loss_value = loss.item<double>();
  }
  return loss_value;
}

}  // namespace

int main() {
  torch::manual_seed(42);

  auto data = torch::randn({16, 4});
  auto true_w = torch::tensor({{1.0, -1.0, 0.5, 0.5}, {0.0, 2.0, -0.5, 1.0}});
  auto target = torch::matmul(data, true_w.t());

  // --- Adam ---
  // Standard Adam configuration (from the docs).
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-3)   // learning rate
            .betas({0.9, 0.999})          // momentum terms
            .eps(1e-8)                    // numerical stability
            .weight_decay(0));            // L2 penalty
    auto loss = TrainSteps(optimizer, *model, data, target, 3);
    std::cout << "Adam final loss: " << loss << std::endl;
  }

  // Adam for transformers: lower learning rate, beta2=0.98 (from the docs).
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-4).betas({0.9, 0.98}));
    auto loss = TrainSteps(optimizer, *model, data, target, 3);
    std::cout << "Adam (lr=1e-4, betas={0.9, 0.98}) final loss: " << loss
              << std::endl;
  }

  // --- AdamW ---
  // AdamW with decoupled weight decay - preferred for transformers.
  // In AdamW: weight = weight - lr * grad - lr * weight_decay * weight
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(1e-4)
            .betas({0.9, 0.999})
            .weight_decay(0.01));  // Decoupled regularization
    auto loss = TrainSteps(optimizer, *model, data, target, 3);
    std::cout << "AdamW (decoupled weight_decay=0.01) final loss: " << loss
              << std::endl;
  }

  // --- RMSprop ---
  // RMSprop for RNN training (from the docs).
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::RMSprop optimizer(
        model->parameters(),
        torch::optim::RMSpropOptions(1e-3)
            .alpha(0.99)       // smoothing constant
            .momentum(0.9)     // optional momentum
            .centered(true));  // normalize by variance
    auto loss = TrainSteps(optimizer, *model, data, target, 3);
    std::cout << "RMSprop (centered, momentum=0.9) final loss: " << loss
              << std::endl;
  }

  // --- Adagrad ---
  // Adagrad for sparse NLP features (from the docs).
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::Adagrad optimizer(
        model->parameters(),
        torch::optim::AdagradOptions(0.01)
            .lr_decay(0)  // learning rate decay
            .weight_decay(0)
            .initial_accumulator_value(0));
    auto loss = TrainSteps(optimizer, *model, data, target, 3);
    std::cout << "Adagrad final loss: " << loss << std::endl;
  }

  std::cout << "adaptive example finished" << std::endl;
  return 0;
}
