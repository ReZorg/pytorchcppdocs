// Archives: torch::serialize::OutputArchive / InputArchive for saving
// multiple named values (tensors and scalars) into a single file, including
// nested sub-archives. The file is written under
// std::filesystem::temp_directory_path() and removed at the end.
//
// Adapted from docs: api/serialize/archives.md

#include <torch/torch.h>

#include <filesystem>
#include <iostream>

int main() {
  torch::manual_seed(42);

  const auto path =
      (std::filesystem::temp_directory_path() / "libtorch_archive.pt")
          .string();

  // --- Writing multiple named values with OutputArchive (from the docs) ---
  torch::Tensor tensor1 = torch::randn({2, 3});
  torch::Tensor tensor2 = torch::randn({2, 3});
  {
    torch::serialize::OutputArchive archive;
    archive.write("tensor1", tensor1);
    archive.write("tensor2", tensor2);

    // Save multiple tensors and metadata (from the docs).
    auto model_weights = torch::randn({4, 8});
    auto model_biases = torch::randn({8});
    int current_epoch = 3;
    float best_loss = 0.25f;
    archive.write("weights", model_weights);
    archive.write("biases", model_biases);
    archive.write("epoch", torch::tensor(current_epoch));
    archive.write("loss", torch::tensor(best_loss));

    // Nested archives group related values under one key hierarchy:
    // write a child OutputArchive under a key of the parent.
    torch::serialize::OutputArchive nested;
    nested.write("name", torch::tensor({1, 2, 3}));
    archive.write("metadata", nested);

    archive.save_to(path);
    std::cout << "Wrote OutputArchive (tensors + metadata + nested) to "
              << path << std::endl;
  }

  // --- Reading values back with InputArchive (from the docs) ---
  {
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    torch::Tensor loaded1, loaded2;
    archive.read("tensor1", loaded1);
    archive.read("tensor2", loaded2);
    std::cout << "tensor1 round-trip equal: "
              << (torch::allclose(tensor1, loaded1) ? "yes" : "no")
              << std::endl;
    std::cout << "tensor2 round-trip equal: "
              << (torch::allclose(tensor2, loaded2) ? "yes" : "no")
              << std::endl;

    // Existence check with try_read before reading an optional key.
    torch::Tensor maybe;
    bool found = archive.try_read("weights", maybe);
    std::cout << "try_read('weights'): " << (found ? "found" : "missing")
              << std::endl;

    torch::Tensor weights, biases, epoch_tensor, loss_tensor;
    archive.read("weights", weights);
    archive.read("biases", biases);
    archive.read("epoch", epoch_tensor);
    archive.read("loss", loss_tensor);
    int epoch = epoch_tensor.item<int>();
    float loss = loss_tensor.item<float>();
    std::cout << "epoch: " << epoch << " loss: " << loss << std::endl;

    // Read the nested sub-archive.
    torch::serialize::InputArchive nested;
    archive.read("metadata", nested);
    torch::Tensor name;
    nested.read("name", name);
    std::cout << "nested metadata/name: " << name << std::endl;

    // InputArchive::keys() lists all top-level keys that were written.
    std::cout << "InputArchive holds " << archive.keys().size()
              << " top-level keys" << std::endl;
  }

  // Cleanup: remove the archive file this example wrote.
  std::filesystem::remove(path);
  std::cout << "Cleaned up temporary files" << std::endl;

  std::cout << "archives example finished" << std::endl;
  return 0;
}
