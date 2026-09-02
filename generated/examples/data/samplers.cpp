// Samplers: SequentialSampler, RandomSampler, DistributedRandomSampler, and
// DistributedSequentialSampler (both derived from DistributedSampler), plus
// StreamSampler (which yields CustomBatchRequest objects), and
// Sampler::save()/load() serialization through archives.
//
// Adapted from docs: api/data/samplers.md

#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

namespace {

// Drains an index sampler for one epoch and returns the yielded indices.
// RandomSampler, SequentialSampler, and the distributed samplers all use
// std::vector<size_t> as their BatchRequest type.
template <typename Sampler>
std::vector<size_t> Drain(Sampler& sampler, size_t batch_size) {
  std::vector<size_t> indices;
  sampler.reset();
  // Sampler::next() takes the batch size and returns the next index batch.
  while (auto next = sampler.next(batch_size)) {
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

  // --- StreamSampler: draws from (possibly infinite) data streams ---
  // Instead of indices it returns BatchSize, a CustomBatchRequest describing
  // how many samples to fetch next from the stream.
  torch::data::samplers::StreamSampler stream_sampler(
      /*epoch_size=*/8);
  stream_sampler.reset();
  size_t stream_total = 0;
  while (auto batch_size = stream_sampler.next(4)) {
    // CustomBatchRequest exposes the request through size().
    stream_total += batch_size->size();
  }
  std::cout << "StreamSampler yielded " << stream_total << " batch slots"
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

  std::cout << "samplers example finished" << std::endl;
  return 0;
}
