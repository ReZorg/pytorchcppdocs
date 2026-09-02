// DataLoader: torch::data::make_data_loader with DataLoaderOptions
// (batch_size, workers, shuffle, samplers), range-based iteration over
// batches, and the stateless vs. stateful data loader paths.
//
// Adapted from docs: api/data/dataloader.md

#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <utility>

namespace {

// Synthetic map-style dataset: feature i is filled with float(i), label is i.
class SyntheticDataset
    : public torch::data::datasets::Dataset<SyntheticDataset> {
 public:
  explicit SyntheticDataset(size_t size) : size_(size) {}

  torch::data::Example<> get(size_t index) override {
    auto data = torch::full({4}, static_cast<float>(index));
    auto target = torch::tensor(static_cast<int64_t>(index));
    return {data, target};
  }

  torch::optional<size_t> size() const override { return size_; }

 private:
  size_t size_;
};

// A stateful dataset manages its own batching internally: it produces whole
// batches directly from an internal position instead of single samples
// fetched by index through an external sampler.
class BatchStreamDataset
    : public torch::data::datasets::StatefulDataset<BatchStreamDataset> {
 public:
  BatchStreamDataset(size_t num_batches, size_t batch_size)
      : num_batches_(num_batches), batch_size_(batch_size) {}

  void reset() override { next_batch_ = 0; }

  torch::optional<std::vector<torch::data::Example<>>> get_batch(
      size_t /*batch_size*/) override {
    if (next_batch_ >= num_batches_) {
      return torch::nullopt;
    }
    std::vector<torch::data::Example<>> batch;
    batch.reserve(batch_size_);
    for (size_t i = 0; i < batch_size_; ++i) {
      auto value = static_cast<float>(next_batch_ * batch_size_ + i);
      batch.push_back({torch::full({4}, value),
                       torch::tensor(static_cast<int64_t>(value))});
    }
    ++next_batch_;
    return batch;
  }

  torch::optional<size_t> size() const override { return num_batches_; }

  void save(torch::serialize::OutputArchive& archive) const override {
    archive.write("next_batch", torch::tensor(static_cast<int64_t>(next_batch_)));
  }

  void load(torch::serialize::InputArchive& archive) override {
    auto value = torch::tensor(0);
    archive.read("next_batch", value);
    next_batch_ = static_cast<size_t>(value.item<int64_t>());
  }

 private:
  size_t num_batches_;
  size_t batch_size_;
  size_t next_batch_ = 0;
};

}  // namespace

int main() {
  torch::manual_seed(42);

  // --- StatelessDataLoader path (map-style dataset + external sampler) ---
  // Stack<> collates individual Examples into batched tensors, giving the
  // loader's iterator batches with .data and .target members.
  auto dataset = SyntheticDataset(12).map(torch::data::transforms::Stack<>());

  // Use make_data_loader to create a DataLoader from a dataset (docs).
  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions()
          .batch_size(4)
          .workers(2)  // parallel loading worker threads
          .max_jobs(2));

  std::cout << "StatelessDataLoader batches:" << std::endl;
  for (auto& batch : *data_loader) {
    std::cout << "  batch data sizes: " << batch.data.sizes()
              << " target sizes: " << batch.target.sizes() << std::endl;
  }

  // --- Shuffled loading with an explicit RandomSampler ---
  auto shuffled_dataset =
      SyntheticDataset(12).map(torch::data::transforms::Stack<>());
  auto shuffled_loader = torch::data::make_data_loader(
      std::move(shuffled_dataset),
      torch::data::samplers::RandomSampler(12),
      torch::data::DataLoaderOptions().batch_size(4).workers(0));
  auto first_batch = shuffled_loader->begin();
  std::cout << "RandomSampler first batch labels: "
            << (*first_batch).target.view({-1}) << std::endl;

  // --- StatefulDataLoader path (dataset produces batches itself) ---
  auto stateful_loader = torch::data::make_data_loader(
      BatchStreamDataset(3, 4),
      torch::data::DataLoaderOptions().batch_size(4).workers(0));
  int num_batches = 0;
  for (auto& batch : *stateful_loader) {
    ++num_batches;
  }
  std::cout << "StatefulDataLoader produced " << num_batches << " batches"
            << std::endl;

  std::cout << "dataloader example finished" << std::endl;
  return 0;
}
