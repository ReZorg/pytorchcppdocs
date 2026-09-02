// Gradient descent optimizers: torch::optim::SGD with momentum, Nesterov
// momentum, weight decay, SGDOptions, and per-layer optimizer parameter
// groups.
//
// Adapted from docs: api/optim/gradient_descent.md

#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <vector>

namespace {

// A tiny model with two layers so parameter groups are meaningful.
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

// Runs a few SGD training steps and returns the final loss.
double TrainSteps(torch::optim::SGD& optimizer, TinyNet& model,
                  const torch::Tensor& data, const torch::Tensor& target,
                  int steps) {
  double loss_value = 0.0;
  for (int step = 0; step < steps; ++step) {
    optimizer.zero_grad();  // Clear gradients
    auto output = model.forward(data);
    auto loss = torch::mse_loss(output, target);
    loss.backward();   // Compute gradients
    optimizer.step();  // Update parameters
    loss_value = loss.item<double>();
  }
  return loss_value;
}

}  // namespace

int main() {
  torch::manual_seed(42);

  // Synthetic regression data: y = x * W^T with fixed random W.
  auto data = torch::randn({16, 4});
  auto true_w = torch::tensor({{1.0, -1.0, 0.5, 0.5}, {0.0, 2.0, -0.5, 1.0}});
  auto target = torch::matmul(data, true_w.t());

  // --- SGD with momentum, weight decay, and Nesterov momentum ---
  // Standard SGD with momentum - good for CNNs (from the docs).
  auto model = std::make_shared<TinyNet>();
  torch::optim::SGD optimizer(model->parameters(),
                              torch::optim::SGDOptions(0.01)  // learning rate
                                  .momentum(0.9)        // momentum factor
                                  .weight_decay(1e-4)   // L2 regularization
                                  .nesterov(true));     // Nesterov momentum

  auto loss = TrainSteps(optimizer, *model, data, target, 3);
  std::cout << "SGD (momentum=0.9, nesterov, weight_decay=1e-4) final loss: "
            << loss << std::endl;

  // --- Plain SGD without momentum for comparison ---
  auto plain_model = std::make_shared<TinyNet>();
  torch::optim::SGD plain_optimizer(plain_model->parameters(),
                                    torch::optim::SGDOptions(0.01));
  loss = TrainSteps(plain_optimizer, *plain_model, data, target, 3);
  std::cout << "Plain SGD (no momentum) final loss: " << loss << std::endl;

  // --- Optimizer parameter groups (OptimizerParamGroup) ---
  // Different layers can use different hyperparameters via parameter groups.
  auto grouped_model = std::make_shared<TinyNet>();
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(grouped_model->fc1->parameters(),
                      std::make_unique<torch::optim::SGDOptions>(0.01));
  groups.emplace_back(grouped_model->fc2->parameters(),
                      std::make_unique<torch::optim::SGDOptions>(0.001));
  torch::optim::SGD grouped_optimizer(groups, torch::optim::SGDOptions(0.01));
  loss = TrainSteps(grouped_optimizer, *grouped_model, data, target, 3);
  std::cout << "SGD with per-layer param groups (lr 0.01 / 0.001) final loss: "
            << loss << std::endl;
  std::cout << "Number of param groups: "
            << grouped_optimizer.param_groups().size() << std::endl;

  std::cout << "gradient_descent example finished" << std::endl;
  return 0;
}
