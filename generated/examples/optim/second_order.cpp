// Second-order optimizer: torch::optim::LBFGS, which requires a closure
// function that recomputes the loss on every internal iteration.
//
// Adapted from docs: api/optim/second_order.md

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

}  // namespace

int main() {
  torch::manual_seed(42);

  auto model = std::make_shared<TinyNet>();

  // Full-batch synthetic regression data. LBFGS is intended for full-batch
  // training, not mini-batches.
  auto data = torch::randn({16, 4});
  auto true_w = torch::tensor({{1.0, -1.0, 0.5, 0.5}, {0.0, 2.0, -0.5, 1.0}});
  auto target = torch::matmul(data, true_w.t());

  // Learning rate is often 1.0 for LBFGS; max_iter limits the internal
  // iterations per step and history_size controls how many past gradients
  // are used to approximate the inverse Hessian.
  torch::optim::LBFGS optimizer(model->parameters(),
                                torch::optim::LBFGSOptions(1.0)
                                    .max_iter(20)
                                    .history_size(10));

  // LBFGS requires a closure that recomputes the model (from the docs).
  const int num_epochs = 3;
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto closure = [&]() -> torch::Tensor {
      optimizer.zero_grad();
      auto output = model->forward(data);
      auto loss = torch::mse_loss(output, target);
      loss.backward();
      return loss;
    };
    auto loss = optimizer.step(closure);
    std::cout << "LBFGS epoch " << epoch
              << " loss: " << loss.item<double>() << std::endl;
  }

  std::cout << "second_order example finished" << std::endl;
  return 0;
}
