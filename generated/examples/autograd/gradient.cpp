// Examples for the "Gradient Computation" page (api/autograd/gradient).
// Covers torch::autograd::backward, torch::autograd::grad, and the
// tensor gradient methods requires_grad_, requires_grad, grad, and detach.
#include <torch/torch.h>

#include <iostream>

int main() {
  // Basic usage from the docs: build a small graph and backpropagate.
  auto x = torch::randn({2, 2}, torch::requires_grad());
  auto y = x * x;
  auto z = y.sum();

  // Tensor::backward() calls torch::autograd::backward on a scalar tensor.
  z.backward();
  std::cout << "z.backward(); x.grad() (= 2x):\n" << x.grad() << std::endl;

  // torch::autograd::grad computes gradients of outputs w.r.t. specific
  // inputs and returns them instead of accumulating into .grad().
  auto x2 = torch::randn({2, 2}, torch::requires_grad());
  auto z2 = (x2 * x2).sum();
  torch::autograd::variable_list grads = torch::autograd::grad({z2}, {x2});
  std::cout << "autograd::grad({z2}, {x2})[0]:\n" << grads[0] << std::endl;

  // torch::autograd::backward with an explicit grad_tensors argument:
  // for a non-scalar output, grad_tensors is the vector in the
  // Jacobian-vector product.
  auto x3 = torch::randn({3}, torch::requires_grad());
  auto out = x3 * x3;
  torch::autograd::backward({out}, {torch::ones_like(out)});
  std::cout << "backward with grad_tensors; x3.grad() (= 2x):\n"
            << x3.grad() << std::endl;

  // Gradients accumulate into the leaves, so zero them when needed.
  // Backward frees the saved graph values by default; rebuild the graph
  // (the usual efficient pattern) or pass retain_graph=true to traverse it
  // a second time.
  x3.grad().zero_();
  out = x3 * x3;
  torch::autograd::backward({out}, {torch::ones_like(out)},
                            /*retain_graph=*/true);
  torch::autograd::backward({out}, {torch::ones_like(out)});
  std::cout << "accumulated grad after two backward calls (= 4x):\n"
            << x3.grad() << std::endl;

  // create_graph builds a graph of the derivative itself, enabling
  // higher-order derivatives. dz/dx = 2x, d2z/dx2 = 2.
  auto x4 = torch::randn({2, 2}, torch::requires_grad());
  auto z4 = (x4 * x4).sum();
  auto first = torch::autograd::grad({z4}, {x4}, /*grad_outputs=*/{},
                                     /*retain_graph=*/std::nullopt,
                                     /*create_graph=*/true)[0];
  auto second = torch::autograd::grad({first.sum()}, {x4})[0];
  std::cout << "second derivative (should be all 2s):\n" << second
            << std::endl;

  // The inputs argument restricts which leaves receive gradients.
  auto a = torch::randn({2}, torch::requires_grad());
  auto b = torch::randn({2}, torch::requires_grad());
  auto total = (a * b).sum();
  torch::autograd::backward({total}, /*grad_tensors=*/{},
                            /*retain_graph=*/std::nullopt,
                            /*create_graph=*/false, /*inputs=*/{a});
  std::cout << "grad w.r.t. a only; a.grad() defined: " << a.grad().defined()
            << ", b.grad() defined: " << b.grad().defined() << std::endl;

  // Tensor gradient methods.
  auto t = torch::randn({2, 2}).requires_grad_(true);  // enable tracking
  bool needs_grad = t.requires_grad();
  std::cout << "t.requires_grad(): " << needs_grad << std::endl;
  (t * t).sum().backward();
  auto grad = t.grad();  // access the accumulated gradient
  std::cout << "t.grad() defined: " << grad.defined() << std::endl;

  // detach() returns a new tensor sharing storage but disconnected from
  // the computation graph.
  auto t_detached = t.detach();
  std::cout << "t.detach().requires_grad(): " << t_detached.requires_grad()
            << std::endl;

  return 0;
}
