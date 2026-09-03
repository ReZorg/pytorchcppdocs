// Frequently asked C++ frontend patterns: device selection, NoGradGuard
// inference, module definition, datasets + data loaders, and checkpointing.
//
// Adapted from docs: faq.md

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <utility>

namespace {

// A minimal module, mirroring the FAQ's MyModel walkthrough.
struct MyModelImpl : torch::nn::Module {
  MyModelImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(784, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    return fc2->forward(x);
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

TORCH_MODULE(MyModel);

// The FAQ's CustomDataset pattern over an in-memory tensor pair.
class CustomDataset
    : public torch::data::datasets::Dataset<CustomDataset> {
 public:
  explicit CustomDataset(int64_t size)
      : data_(torch::randn({size, 784})),
        target_(torch::randint(0, 10, {size})) {}

  torch::data::Example<> get(size_t index) override {
    return {data_[static_cast<int64_t>(index)],
            target_[static_cast<int64_t>(index)]};
  }

  torch::optional<size_t> size() const override {
    return static_cast<size_t>(data_.size(0));
  }

 private:
  torch::Tensor data_;
  torch::Tensor target_;
};

}  // namespace

int main() {
  // FAQ: "How do I select the device?" Prefer CUDA when it is available.
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  std::cout << "Using device: " << device << std::endl;

  // FAQ: "How do I run inference?" eval mode + NoGradGuard.
  MyModel model;
  model->to(device);
  model->eval();
  auto input = torch::randn({8, 784}, device);
  {
    torch::NoGradGuard no_grad;
    auto output = model->forward(input);
    std::cout << "Inference output sizes: " << output.sizes() << std::endl;
  }
  model->train();

  // FAQ: "How do I use datasets and data loaders?"
  auto dataset = CustomDataset(/*size=*/64)
                     .map(torch::data::transforms::Stack<>());
  auto dataloader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(32).workers(0));
  int64_t batches = 0;
  for (auto& batch : *dataloader) {
    batches += 1;
  }
  std::cout << "Iterated " << batches << " batches" << std::endl;

  // FAQ: "How do I save and load a model?"
  const std::string path = "faq_model.pt";
  torch::save(model, path);
  MyModel loaded;
  torch::load(loaded, path);
  std::cout << "Checkpoint round-trip succeeded" << std::endl;

  // FAQ: "How do I control intra-op threading?"
  at::set_num_threads(4);
  std::cout << "Intra-op threads: " << at::get_num_threads() << std::endl;

  std::cout << "faq example finished" << std::endl;
  return 0;
}
