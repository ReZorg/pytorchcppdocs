// Saving and loading: torch::save / torch::load for single tensors, vectors
// of tensors, a small nn::Module, and an optimizer's state. All files are
// written under std::filesystem::temp_directory_path() and removed at the
// end.
//
// Adapted from docs: api/serialize/save_load.md

#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace {

struct Net : torch::nn::Module {
  Net() {
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

  // Write all artifacts into the system temp directory.
  const auto temp_dir = std::filesystem::temp_directory_path();
  const auto tensor_path = (temp_dir / "libtorch_tensor.pt").string();
  const auto tensor_vec_path = (temp_dir / "libtorch_tensor_vec.pt").string();
  const auto model_path = (temp_dir / "libtorch_model.pt").string();
  const auto optimizer_path = (temp_dir / "libtorch_optimizer.pt").string();

  // --- Saving and loading a tensor (from the docs) ---
  torch::Tensor tensor = torch::randn({2, 3});
  torch::save(tensor, tensor_path);

  torch::Tensor loaded;
  torch::load(loaded, tensor_path);
  std::cout << "Tensor round-trip equal: "
            << (torch::allclose(tensor, loaded) ? "yes" : "no") << std::endl;

  // --- Saving and loading a vector of tensors ---
  std::vector<torch::Tensor> tensors = {torch::ones({2}),
                                        torch::zeros({2}),
                                        torch::full({2}, 3.0)};
  torch::save(tensors, tensor_vec_path);
  std::vector<torch::Tensor> loaded_tensors;
  torch::load(loaded_tensors, tensor_vec_path);
  std::cout << "Loaded " << loaded_tensors.size() << " tensors; last: "
            << loaded_tensors.back() << std::endl;

  // --- Saving and loading a module (from the docs) ---
  auto model = std::make_shared<Net>();
  torch::save(model, model_path);

  auto loaded_model = std::make_shared<Net>();
  torch::load(loaded_model, model_path);

  auto probe = torch::randn({1, 4});
  std::cout << "Module round-trip equal: "
            << (torch::allclose(model->forward(probe),
                                loaded_model->forward(probe))
                    ? "yes"
                    : "no")
            << std::endl;

  // --- Saving and loading optimizer state (from the docs) ---
  auto optimizer =
      std::make_shared<torch::optim::Adam>(model->parameters(), 0.001);
  // Take one training step so the optimizer holds non-trivial state.
  optimizer->zero_grad();
  auto loss = torch::mse_loss(model->forward(probe), torch::zeros({1, 2}));
  loss.backward();
  optimizer->step();

  torch::save(*optimizer, optimizer_path);

  auto loaded_optimizer =
      std::make_shared<torch::optim::Adam>(loaded_model->parameters(), 0.001);
  torch::load(*loaded_optimizer, optimizer_path);
  std::cout << "Optimizer state saved and reloaded" << std::endl;

  // Cleanup: remove every file this example wrote.
  for (const auto& path :
       {tensor_path, tensor_vec_path, model_path, optimizer_path}) {
    std::filesystem::remove(path);
  }
  std::cout << "Cleaned up temporary files" << std::endl;

  std::cout << "save_load example finished" << std::endl;
  return 0;
}
