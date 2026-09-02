// Examples for the "Gradient Modes" page (api/autograd/modes).
// Covers torch::NoGradGuard, c10::InferenceMode (including nesting and
// inference tensors), and at::AutoDispatchBelowADInplaceOrView for custom
// autograd kernels that redispatch below the Autograd dispatch key.
#include <torch/torch.h>

#include <iostream>

// The modern replacement for a legacy AutoNonVariableTypeMode-based custom
// kernel: redispatch below the Autograd key inside forward.
class CubeFunction : public torch::autograd::Function<CubeFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input) {
    ctx->save_for_backward({input});
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = at::mul(at::mul(input, input), input);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    return {grad_outputs[0] * 3 * input * input};
  }
};

int main() {
  torch::Tensor x = torch::randn({2, 2}, torch::requires_grad());

  // NoGradGuard: RAII guard that disables gradient computation in its
  // scope and restores the previous gradient mode on destruction.
  {
    torch::NoGradGuard no_grad;
    torch::Tensor result = x * 2;  // No gradients computed in this scope
    std::cout << "inside NoGradGuard, (x * 2).requires_grad(): "
              << result.requires_grad() << std::endl;
  }
  torch::Tensor tracked = x * 2;
  std::cout << "outside NoGradGuard, (x * 2).requires_grad(): "
            << tracked.requires_grad() << std::endl;

  // InferenceMode: more efficient than NoGradGuard for inference-only
  // workloads; also disables view tracking and version counter bumps.
  torch::Tensor inference_tensor;
  {
    c10::InferenceMode guard;
    torch::Tensor result = x * 2;
    std::cout << "inside InferenceMode, (x * 2).requires_grad(): "
              << result.requires_grad() << std::endl;
    // Newly allocated non-view tensors become inference tensors.
    inference_tensor = torch::randn({2, 2});
    std::cout << "allocated tensor is_inference(): "
              << inference_tensor.is_inference() << std::endl;
    result.sum();  // ops do not record grad_fn even for requires_grad input
  }
  std::cout << "after InferenceMode, is_inference() persists: "
            << inference_tensor.is_inference() << std::endl;

  // Inference tensors are immutable outside InferenceMode; clone to get a
  // normal tensor before mutating.
  torch::Tensor normal_copy = inference_tensor.clone();
  normal_copy.add_(1);
  std::cout << "clone of inference tensor is_inference(): "
            << normal_copy.is_inference() << std::endl;

  // InferenceMode can be nested with different enabled/disabled states.
  {
    c10::InferenceMode guard(true);
    std::cout << "nested outer: is_inference of new tensor: "
              << torch::empty({1}).is_inference() << std::endl;
    {
      c10::InferenceMode guard(false);  // InferenceMode is off
      std::cout << "nested inner (disabled): is_inference of new tensor: "
                << torch::empty({1}).is_inference() << std::endl;
    }
    // InferenceMode is on again here.
    std::cout << "back to outer scope: is_inference of new tensor: "
              << torch::empty({1}).is_inference() << std::endl;
  }
  // InferenceMode is off here.
  std::cout << "outside all guards: is_inference of new tensor: "
            << torch::empty({1}).is_inference() << std::endl;

  // Custom autograd kernels redispatch below the Autograd dispatch key
  // with AutoDispatchBelowADInplaceOrView (the renamed, safe
  // AutoNonVariableTypeMode).
  torch::Tensor c = torch::tensor({2.0, 3.0}, torch::requires_grad());
  torch::Tensor cubed = CubeFunction::apply(c)[0];
  cubed.sum().backward();
  std::cout << "CubeFunction forward: " << cubed << std::endl;
  std::cout << "CubeFunction grad (3x^2): " << c.grad() << std::endl;

  return 0;
}
