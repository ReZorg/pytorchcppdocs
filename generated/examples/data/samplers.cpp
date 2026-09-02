// Samplers: SequentialSampler, RandomSampler, DistributedRandomSampler,
// DistributedSequentialSampler (both derived from DistributedSampler), and
// StreamSampler, plus Sampler::save()/load() serialization and the
// CustomBatchRequest index container.
//
// Adapted from docs: api/data/samplers.md

#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

// Drains a sampler for one epoch and returns the yielded indices.
template <typename Sampler>
std::vector<size_t> Drain(Sampler& sampler, size_t batch_size) {
  torch::data::samplers::CustomBatchRequest request(batch_size);
  std::vector<size_t> indices;
  sampler.reset();
  while (auto next = sampler.next(request)) {
    indices.insert(indices.end(), next->begin(), next->end());
  }
  return indices;
}

}  // namespace

int main() {
  const size_t kSize = 10;

  // --- SequentialSampler: samples in order from 0 to N-1 ---
  torch::data::samplers::SequentialSampler sequential(kSize);
  std::cout << "SequentialSampler indices (batch 3): ";
  for (auto index : Drain(sequential, 3)) {
    std::cout << index << " ";
  }
  std::cout << std::endl;

  // --- RandomSampler: samples in random order ---
  torch::data::samplers::RandomSampler random(kSize);
  std::cout << "RandomSampler indices (batch 4): ";
  for (auto index : Drain(random, 4)) {
    std::cout << index << " ";
  }
  std::cout << std::endl;

  // --- DistributedRandomSampler: each process gets a disjoint subset ---
  // (num_replicas=2, rank=0 simulates the first of two processes).
  torch::data::samplers::DistributedRandomSampler dist_random(
      kSize, /*num_replicas=*/2, /*rank=*/0, /*allow_duplicates=*/false);
  std::cout << "DistributedRandomSampler (2 replicas, rank 0) indices: ";
  for (auto index : Drain(dist_random, 2)) {
    std::cout << index << " ";
  }
  std::cout << std::endl;

  // --- DistributedSequentialSampler: ordered disjoint subsets ---
  torch::data::samplers::DistributedSequentialSampler dist_sequential(
      kSize, /*num_replicas=*/2, /*rank=*/1, /*allow_duplicates=*/false);
  std::cout << "DistributedSequentialSampler (2 replicas, rank 1) indices: ";
  for (auto index : Drain(dist_sequential, 2)) {
    std::cout << index << " ";
  }
  std::cout << std::endl;

  // --- StreamSampler: draws from infinite streams (BatchSize requests) ---
  torch::data::samplers::StreamSampler stream_sampler(
      torch::data::samplers::BatchSize(8));
  torch::data::samplers::CustomBatchRequest stream_request(4);
  stream_sampler.reset();
  size_t stream_total = 0;
  while (auto next = stream_sampler.next(stream_request)) {
    stream_total += next->size();
  }
  std::cout << "StreamSampler yielded " << stream_total << " indices"
            << std::endl;

  // --- Sampler serialization: save()/load() via archives ---
  // All samplers derive from torch::data::samplers::Sampler, which supports
  // save/load through torch::serialize archives.
  std::stringstream buffer;
  {
    torch::serialize::OutputArchive archive;
    random.save(archive);
    archive.save_to(buffer);
  }
  torch::data::samplers::RandomSampler restored(kSize);
  {
    torch::serialize::InputArchive archive;
    archive.load_from(buffer);
    restored.load(archive);
  }
  std::cout << "Sampler state saved and restored (epoch index: "
            << restored.index() << ")" << std::endl;

  // --- CustomBatchRequest: the index container passed to Sampler::next ---
  // It behaves like a vector<size_t> and can carry an explicit index list.
  torch::data::samplers::CustomBatchRequest explicit_request{0, 2, 4};
  std::cout << "CustomBatchRequest holds " << explicit_request.size()
            << " indices" << std::endl;

  std::cout << "samplers example finished" << std::endl;
  return 0;
}
