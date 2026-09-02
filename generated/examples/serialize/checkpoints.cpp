// Checkpoints: save and resume complete training state (model parameters,
// optimizer state, epoch number) with a combined OutputArchive, following
// the save_checkpoint()/load_checkpoint() pattern. The checkpoint file is
// written under std::filesystem::temp_directory_path() and removed at the
// end.
//
// Adapted from docs: api/serialize/checkpoints.md

#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

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

// --- Creating a checkpoint (from the docs) ---
// A checkpoint bundles model parameters, optimizer state (momentum buffers,
// learning rates), and bookkeeping such as the current epoch number.
void SaveCheckpoint(std::shared_ptr<Net> model,
                    torch::optim::Adam& optimizer, int epoch,
                    const std::string& path) {
  torch::serialize::OutputArchive archive;
  model->save(archive);
  archive.write("epoch", torch::tensor(epoch));
  optimizer.save(archive);
  archive.save_to(path);
}

// --- Loading a checkpoint (from the docs) ---
int LoadCheckpoint(std::shared_ptr<Net> model, torch::optim::Adam& optimizer,
                   const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  model->load(archive);
  torch::Tensor epoch_tensor;
  archive.read("epoch", epoch_tensor);
  optimizer.load(archive);
  return epoch_tensor.item<int>();
}

}  // namespace

int main() {
  torch::manual_seed(42);

  auto model = std::make_shared<Net>();
  torch::optim::Adam optimizer(model->parameters(), 1e-3);

  const auto checkpoint_path =
      (std::filesystem::temp_directory_path() / "libtorch_checkpoint.pt")
          .string();

  // Synthetic training data.
  auto data = torch::randn({16, 4});
  auto target = torch::randn({16, 2});

  // Resume from checkpoint if it exists (from the docs).
  int start_epoch = 0;
  if (std::filesystem::exists(checkpoint_path)) {
    std::cout << "Loading checkpoint..." << std::endl;
    start_epoch = LoadCheckpoint(model, optimizer, checkpoint_path);
    std::cout << "Resuming from epoch " << start_epoch << std::endl;
  }

  // Training loop with periodic checkpoints (shortened to 3 epochs so the
  // example terminates quickly; real runs would loop much longer).
  for (int epoch = start_epoch; epoch < 3; ++epoch) {
    optimizer.zero_grad();
    auto loss = torch::mse_loss(model->forward(data), target);
    loss.backward();
    optimizer.step();
    std::cout << "Epoch " << epoch << " loss: " << loss.item<double>()
              << std::endl;

    // Save a checkpoint every epoch (docs suggest e.g. every 10 epochs).
    SaveCheckpoint(model, optimizer, epoch + 1, checkpoint_path);
    std::cout << "Saved checkpoint at epoch " << epoch + 1 << std::endl;
  }

  // --- Verify the checkpoint: resume into a fresh model + optimizer ---
  auto resumed_model = std::make_shared<Net>();
  torch::optim::Adam resumed_optimizer(resumed_model->parameters(), 1e-3);
  int resumed_epoch = LoadCheckpoint(resumed_model, resumed_optimizer,
                                     checkpoint_path);
  std::cout << "Resumed checkpoint at epoch " << resumed_epoch << std::endl;

  // The resumed model must produce identical outputs.
  auto before = model->forward(data);
  auto after = resumed_model->forward(data);
  std::cout << "Resumed model outputs match: "
            << (torch::allclose(before, after) ? "yes" : "no") << std::endl;

  // Cleanup: remove the checkpoint file this example wrote.
  std::filesystem::remove(checkpoint_path);
  std::cout << "Cleaned up temporary files" << std::endl;

  std::cout << "checkpoints example finished" << std::endl;
  return 0;
}
