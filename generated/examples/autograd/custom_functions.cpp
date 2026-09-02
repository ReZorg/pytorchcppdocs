// Examples for the "Custom Autograd Functions" page
// (api/autograd/custom_functions).
// Covers torch::autograd::Function, AutogradContext (save_for_backward,
// saved_data, get_saved_variables), and at::AutoDispatchBelowADInplaceOrView.
#include <torch/torch.h>

#include <torch/csrc/autograd/function.h>

#include <iostream>

// Custom autograd function from the docs: inherit from
// torch::autograd::Function<T> and implement static forward/backward.
class MyReLU : public torch::autograd::Function<MyReLU> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor input) {
    ctx->save_for_backward({input});
    return input.clamp_min(0);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output * (input > 0).to(grad_output.dtype());
    return {grad_input};
  }
};

// Custom kernels that redispatch below the Autograd dispatch key use
// at::AutoDispatchBelowADInplaceOrView (not InferenceMode) inside forward,
// and stash non-tensor arguments in ctx->saved_data.
class ScaledSquare : public torch::autograd::Function<ScaledSquare> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      double scale) {
    ctx->saved_data["scale"] = scale;
    ctx->save_for_backward({input});

    at::AutoDispatchBelowADInplaceOrView guard;
    // The "custom kernel": a plain ATen op redispatched below Autograd.
    auto result = at::mul(input, input) * scale;
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto scale = ctx->saved_data["scale"].toDouble();
    auto grad_input = grad_outputs[0] * 2 * input * scale;
    return {grad_input, torch::Tensor()};
  }
};

int main() {
  // Usage of a custom Function: apply() runs forward and wires up backward.
  torch::Tensor input =
      torch::tensor({-1.0, 0.5, 2.0, -3.0}, torch::requires_grad());
  torch::Tensor output = MyReLU::apply(input);
  std::cout << "MyReLU forward: " << output << std::endl;

  output.sum().backward();
  std::cout << "MyReLU grad (0 for x<=0, 1 for x>0): " << input.grad()
            << std::endl;

  // Custom function with saved_data and AutoDispatchBelowADInplaceOrView.
  torch::Tensor x =
      torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
  torch::Tensor y = ScaledSquare::apply(x, /*scale=*/3.0)[0];
  std::cout << "ScaledSquare forward (3*x^2): " << y << std::endl;
  y.sum().backward();
  std::cout << "ScaledSquare grad (6x): " << x.grad() << std::endl;

  return 0;
}
