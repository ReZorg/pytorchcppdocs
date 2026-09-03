// The PyTorch C++ API in five parts: ATen tensors, Autograd, the C++
// Frontend (nn modules + optimizers), TorchScript, and C++ extensions.
//
// Adapted from docs: index.md

#include <torch/torch.h>

#include <iostream>

// A tiny C++ Frontend module, as sketched on the index page.
struct NetImpl : torch::nn::Module {
  NetImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    return torch::log_softmax(fc2->forward(x), /*dim=*/1);
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

TORCH_MODULE(Net);

int main() {
  // --- ATen: the core tensor library (at::) ---
  at::Tensor a = at::ones({2, 2}, at::TensorOptions().dtype(at::kFloat));
  at::Tensor b = at::randn({2, 2});
  auto c = a + b.to(at::kFloat);
  std::cout << "ATen add result:\n" << c << std::endl;

  // --- Autograd: requires_grad + backward() ---
  auto x = torch::ones({2, 2}, torch::requires_grad());
  auto y = (x * x).sum();
  y.backward();
  std::cout << "Autograd grad of sum(x^2) at x=1:\n" << x.grad() << std::endl;

  // --- C++ Frontend: torch::nn modules + torch::optim ---
  Net net;
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
  auto data = torch::randn({4, 784});
  auto target = torch::randint(0, 10, {4});
  optimizer.zero_grad();
  auto loss = torch::nll_loss(net->forward(data), target);
  loss.backward();
  optimizer.step();
  std::cout << "One frontend training step, loss = " << loss.item<float>()
            << std::endl;

  // --- TorchScript: load/save scripted modules (round-trip via pickle) ---
  // Serializing a module to an archive exercises the same OutputArchive /
  // InputArchive path that TorchScript's save/load is built on.
  torch::serialize::OutputArchive archive;
  net->save(archive);
  archive.save_to("index_net.pt");
  Net loaded;
  torch::serialize::InputArchive input_archive;
  input_archive.load_from("index_net.pt");
  loaded->load(input_archive);
  std::cout << "TorchScript-style serialization round-trip succeeded"
            << std::endl;

  std::cout << "index example finished" << std::endl;
  return 0;
}
