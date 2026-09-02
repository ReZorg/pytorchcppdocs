// End-to-end C++ frontend example: defining a torch::nn::Module, a data
// loader over synthetic (random) tensors, SGD and Adam optimizers, a
// training loop with loss printing, and model checkpointing with
// torch::save (written to a temp file and cleaned up afterwards).
//
// Adapted from docs: frontend.md (the MNIST dataset is replaced with
// randomly generated tensors so the example needs no downloads).

#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <memory>

// Synthetic map-style dataset that generates a random 1x28x28 image and a
// random digit target on the fly (no downloads required).
class RandomMNISTDataset
    : public torch::data::datasets::Dataset<RandomMNISTDataset> {
 public:
  explicit RandomMNISTDataset(size_t size) : size_(size) {}

  torch::data::Example<> get(size_t /*index*/) override {
    return {torch::randn({1, 28, 28}), torch::randint(0, 10, {})};
  }

  torch::optional<size_t> size() const override { return size_; }

 private:
  size_t size_;
};

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register three Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
  torch::manual_seed(42);

  // Create a new Net.
  auto net = std::make_shared<Net>();

  // Create a data loader that yields batches from the synthetic dataset.
  auto data_loader = torch::data::make_data_loader(
      RandomMNISTDataset(/*size=*/256).map(torch::data::transforms::Stack<>()),
      /*batch_size=*/64);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  // Checkpoint path in a temp location; removed again at the end.
  const std::filesystem::path checkpoint =
      std::filesystem::temp_directory_path() / "net.pt";

  for (size_t epoch = 1; epoch <= 2; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = net->forward(batch.data);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss for every batch.
      std::cout << "Epoch: " << epoch << " | Batch: " << ++batch_index
                << " | Loss: " << loss.item<float>() << std::endl;
    }
    // Serialize your model periodically as a checkpoint.
    torch::save(net, checkpoint.string());
    std::cout << "Saved checkpoint to " << checkpoint.string() << std::endl;
  }

  // An Adam optimizer can be used the same way as SGD.
  torch::optim::Adam adam(net->parameters(), /*lr=*/0.001);
  torch::Tensor images = torch::randn({64, 1, 28, 28});
  torch::Tensor targets = torch::randint(0, 10, {64});
  adam.zero_grad();
  torch::Tensor prediction = net->forward(images);
  torch::Tensor loss = torch::nll_loss(prediction, targets);
  loss.backward();
  adam.step();
  std::cout << "Adam step loss: " << loss.item<float>() << std::endl;

  // Load the checkpoint back into a fresh Net to verify the round-trip.
  auto loaded = std::make_shared<Net>();
  torch::load(loaded, checkpoint.string());
  std::cout << "Reloaded checkpoint with " << loaded->parameters().size()
            << " parameter tensors" << std::endl;

  // Clean up the temporary checkpoint file.
  std::filesystem::remove(checkpoint);
  std::cout << "Cleaned up " << checkpoint.string() << std::endl;

  std::cout << "end_to_end example finished" << std::endl;
  return 0;
}
