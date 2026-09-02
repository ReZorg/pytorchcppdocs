// Learning rate schedulers: torch::optim::StepLR (derived from the
// LRScheduler base class) and ReduceLROnPlateauScheduler, plus minimal
// Exponential-decay and lambda-style schedulers implemented directly on the
// LRScheduler base to mirror the ExponentialLR / LambdaLR policies described
// in the docs.
//
// Adapted from docs: api/optim/schedulers.md

#include <torch/torch.h>

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

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

// One tiny synthetic "epoch" of training; returns the loss value.
double TrainOneEpoch(TinyNet& model, torch::optim::Optimizer& optimizer,
                     const torch::Tensor& data, const torch::Tensor& target) {
  model.train();
  optimizer.zero_grad();
  auto output = model.forward(data);
  auto loss = torch::mse_loss(output, target);
  loss.backward();
  optimizer.step();
  return loss.item<double>();
}

double CurrentLr(const torch::optim::Optimizer& optimizer) {
  return optimizer.param_groups().front().options().get_lr();
}

// Exponential decay policy (the docs' ExponentialLR): multiply the initial
// learning rate by gamma each epoch, implemented on the LRScheduler base.
class ExponentialDecayLR : public torch::optim::LRScheduler {
 public:
  ExponentialDecayLR(torch::optim::Optimizer& optimizer, double gamma)
      : LRScheduler(optimizer), gamma_(gamma) {}

 private:
  std::vector<double> get_lrs() override {
    std::vector<double> lrs;
    lrs.reserve(get_current_lrs().size());
    for (double lr : get_current_lrs()) {
      lrs.push_back(lr * gamma_);
    }
    return lrs;
  }

  const double gamma_;
};

// Lambda policy (the docs' LambdaLR): scale the initial learning rate by a
// user-provided function of the epoch count, on the LRScheduler base.
class LambdaLR : public torch::optim::LRScheduler {
 public:
  LambdaLR(torch::optim::Optimizer& optimizer,
           std::function<double(int64_t)> lr_lambda)
      : LRScheduler(optimizer),
        base_lrs_(get_current_lrs()),
        lr_lambda_(std::move(lr_lambda)) {}

 private:
  std::vector<double> get_lrs() override {
    std::vector<double> lrs;
    lrs.reserve(base_lrs_.size());
    const double scale = lr_lambda_(static_cast<int64_t>(step_count_));
    for (double base_lr : base_lrs_) {
      lrs.push_back(base_lr * scale);
    }
    return lrs;
  }

  const std::vector<double> base_lrs_;
  std::function<double(int64_t)> lr_lambda_;
};

}  // namespace

int main() {
  torch::manual_seed(42);

  auto data = torch::randn({16, 4});
  auto true_w = torch::tensor({{1.0, -1.0, 0.5, 0.5}, {0.0, 2.0, -0.5, 1.0}});
  auto target = torch::matmul(data, true_w.t());

  // --- StepLR: decays LR by gamma every step_size epochs (from the docs) ---
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::SGD optimizer(model->parameters(),
                                torch::optim::SGDOptions(0.1));

    // Reduce LR by 10x every 30 epochs (step_size=2 here to keep it quick).
    torch::optim::StepLR scheduler(optimizer, /*step_size=*/2, /*gamma=*/0.1);

    // LRScheduler is the polymorphic base class of all schedulers.
    torch::optim::LRScheduler* base = &scheduler;
    for (int epoch = 0; epoch < 5; ++epoch) {
      auto loss = TrainOneEpoch(*model, optimizer, data, target);
      std::cout << "StepLR epoch " << epoch << " loss: " << loss
                << " lr: " << CurrentLr(optimizer) << std::endl;
      base->step();
    }
  }

  // --- Exponential decay (docs' ExponentialLR): gamma each epoch ---
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::Adam optimizer(model->parameters(),
                                 torch::optim::AdamOptions(1e-3));

    // Reduce LR by 5% each epoch.
    ExponentialDecayLR scheduler(optimizer, /*gamma=*/0.95);
    torch::optim::LRScheduler* base = &scheduler;
    for (int epoch = 0; epoch < 3; ++epoch) {
      auto loss = TrainOneEpoch(*model, optimizer, data, target);
      std::cout << "ExponentialLR epoch " << epoch << " loss: " << loss
                << " lr: " << CurrentLr(optimizer) << std::endl;
      base->step();
    }
  }

  // --- ReduceLROnPlateau: reacts to a metric instead of a fixed schedule ---
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::SGD optimizer(model->parameters(),
                                torch::optim::SGDOptions(0.1));
    torch::optim::ReduceLROnPlateauScheduler scheduler(
        optimizer, torch::optim::ReduceLROnPlateauScheduler::min,
        /*factor=*/0.5, /*patience=*/0);
    // Feed the current loss as the monitored metric; a plateau (no
    // improvement) triggers an LR reduction after `patience` epochs.
    for (int epoch = 0; epoch < 3; ++epoch) {
      auto loss = TrainOneEpoch(*model, optimizer, data, target);
      scheduler.step(static_cast<float>(loss));
      std::cout << "ReduceLROnPlateau epoch " << epoch << " loss: " << loss
                << " lr: " << CurrentLr(optimizer) << std::endl;
    }
  }

  // --- LambdaLR: LR is scaled by a user-provided function of the epoch ---
  {
    auto model = std::make_shared<TinyNet>();
    torch::optim::Adam optimizer(model->parameters(),
                                 torch::optim::AdamOptions(1e-3));
    // Simple decay: lr_lambda(epoch) = 1 / (epoch + 1).
    LambdaLR scheduler(optimizer,
                       [](int64_t epoch) { return 1.0 / (epoch + 1); });
    torch::optim::LRScheduler* base = &scheduler;
    for (int epoch = 0; epoch < 3; ++epoch) {
      auto loss = TrainOneEpoch(*model, optimizer, data, target);
      std::cout << "LambdaLR epoch " << epoch << " loss: " << loss
                << " lr: " << CurrentLr(optimizer) << std::endl;
      base->step();
    }
  }

  std::cout << "schedulers example finished" << std::endl;
  return 0;
}
