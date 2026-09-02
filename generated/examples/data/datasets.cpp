// Datasets: the torch::data::datasets::Dataset base class with a custom
// get()/size() subclass, TensorDataset, map()-produced MapDataset,
// SharedBatchDataset, and ChunkDataset driven by a custom ChunkDataReader.
// MNIST usage is shown in comments because it downloads data from the
// internet.
//
// Adapted from docs: api/data/datasets.md

#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

// --- Custom Dataset (from the docs) ---
// All map-style datasets inherit from Dataset and implement get() and
// size(). torch::data::Example<> pairs a data tensor with a target tensor.
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
 public:
  explicit CustomDataset(const std::string& /*root*/) {
    // Synthesize data instead of loading it from the root directory.
    images_ = torch::arange(10 * 4, torch::kFloat32).reshape({10, 4});
    labels_ = torch::arange(10, torch::kInt64);
  }

  torch::data::Example<> get(size_t index) override {
    return {images_[index], labels_[index]};
  }

  torch::optional<size_t> size() const override { return images_.size(0); }

 private:
  torch::Tensor images_, labels_;
};

// --- Custom ChunkDataReader (from the docs) ---
// A ChunkDataReader reads a data source chunk by chunk; ChunkDataset uses it
// for large-scale data loading. Chunks here are vectors of scalar tensors.
class SyntheticChunkReader
    : public torch::data::datasets::ChunkDataReader<torch::Tensor> {
 public:
  // ChunkDataset batches the unwrapped ChunkType directly.
  using BatchType = ChunkType;

  explicit SyntheticChunkReader(size_t chunk_count)
      : chunk_count_(chunk_count) {}

  std::vector<torch::Tensor> read_chunk(size_t chunk_index) override {
    // Each chunk holds 4 scalar tensors derived from the chunk index.
    ChunkType chunk;
    for (size_t i = 0; i < 4; ++i) {
      chunk.push_back(
          torch::tensor(static_cast<int64_t>(chunk_index * 4 + i)));
    }
    return chunk;
  }

  size_t chunk_count() override { return chunk_count_; }

  void reset() override {}

 private:
  size_t chunk_count_;
};

}  // namespace

int main() {
  torch::manual_seed(42);

  // --- Custom Dataset: get() / size() ---
  CustomDataset custom("./data");
  std::cout << "CustomDataset size: " << *custom.size() << std::endl;
  auto example = custom.get(3);
  std::cout << "CustomDataset[3] data: " << example.data
            << " target: " << example.target << std::endl;

  // --- TensorDataset: a dataset of tensors, indexed inside get() ---
  // TensorDataset stores one tensor and returns single-tensor TensorExamples
  // (no separate target tensor).
  torch::data::datasets::TensorDataset tensor_dataset(
      torch::arange(8 * 4, torch::kFloat32).reshape({8, 4}));
  std::cout << "TensorDataset size: " << *tensor_dataset.size() << std::endl;
  std::cout << "TensorDataset[0] data: " << tensor_dataset.get(0).data
            << std::endl;

  // --- MapDataset: the result of calling .map() with a transform ---
  // Built-in MNIST usage from the docs (NOT run here because constructing
  // MNIST downloads the dataset from the internet):
  //
  //   auto dataset = torch::data::datasets::MNIST("./data")
  //       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
  //       .map(torch::data::transforms::Stack<>());
  //
  // The same pattern with the synthetic CustomDataset:
  auto mapped = CustomDataset("./data")
                    .map(torch::data::transforms::Normalize<>(0.0, 1.0))
                    .map(torch::data::transforms::Stack<>());
  std::cout << "MapDataset (Normalize + Stack) size: " << *mapped.size()
            << std::endl;
  std::vector<size_t> indices{0, 1, 2, 3};
  auto batch = mapped.get_batch(indices);
  std::cout << "MapDataset batch of 4 data sizes: " << batch.data.sizes()
            << std::endl;

  // --- SharedBatchDataset: wraps a BatchDataset so it can be shared ---
  using MappedDataset = torch::data::datasets::MapDataset<
      CustomDataset, torch::data::transforms::Stack<>>;
  auto shared = torch::data::datasets::make_shared_dataset<
      torch::data::datasets::SharedBatchDataset<MappedDataset>>(
      std::make_shared<MappedDataset>(
          CustomDataset("./data").map(torch::data::transforms::Stack<>())));
  auto shared_batch = shared.get_batch({0, 1});
  std::cout << "SharedBatchDataset batch of 2 data sizes: "
            << shared_batch.data.sizes() << std::endl;

  // --- ChunkDataset: stateful loading over a ChunkDataReader ---
  // A ChunkSampler picks which chunk to load next; an ExampleSampler orders
  // examples within each chunk.
  torch::data::datasets::ChunkDataset<
      SyntheticChunkReader, torch::data::samplers::SequentialSampler,
      torch::data::samplers::SequentialSampler>
      chunk_dataset(
          SyntheticChunkReader(2),
          torch::data::samplers::SequentialSampler(2),  // chunk sampler
          torch::data::samplers::SequentialSampler(4),  // example sampler
          torch::data::datasets::ChunkDatasetOptions(
              /*preloader_count=*/1, /*batch_size=*/2));
  // ChunkDataset is a StatefulDataset; because it owns worker threads it is
  // neither copyable nor movable, so drive it directly with get_batch()
  // (each request is a batch size) instead of a DataLoader.
  chunk_dataset.reset();
  size_t seen = 0;
  while (auto chunk_batch = chunk_dataset.get_batch(/*batch_size=*/2)) {
    seen += chunk_batch->size();
  }
  std::cout << "ChunkDataset yielded " << seen << " batched examples"
            << std::endl;

  std::cout << "datasets example finished" << std::endl;
  return 0;
}
