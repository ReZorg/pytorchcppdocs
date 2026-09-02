// Embedding layers: Embedding (dense lookup) and EmbeddingBag (sum/mean of
// embeddings), including from_pretrained.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  const int64_t vocab_size = 100, embed_dim = 8;

  // --- Embedding: indices -> dense vectors ---
  auto embedding = torch::nn::Embedding(
      torch::nn::EmbeddingOptions(vocab_size, embed_dim)
          .padding_idx(0));  // index 0 always embeds to zeros
  auto indices = torch::tensor({1, 2, 3, 4});
  auto embedded = embedding->forward(indices);
  PrintSizes("Embedding output", embedded);  // [4, 8]

  // Batched lookup with padding: [batch, seq_len] -> [batch, seq_len, dim].
  auto batch_indices = torch::tensor({{5, 6, 7, 0}, {8, 0, 0, 0}});
  PrintSizes("Embedding batch output", embedding->forward(batch_indices));
  std::cout << "Padding row is zero: "
            << (embedding->forward(torch::tensor({0})).abs().sum()
                        .item<float>() == 0.0f
                    ? "true"
                    : "false")
            << std::endl;

  // --- Embedding from pretrained weights (frozen table) ---
  auto weights = torch::randn({vocab_size, embed_dim});
  auto pretrained = torch::nn::Embedding::from_pretrained(
      weights, torch::nn::EmbeddingFromPretrainedOptions().freeze(true));
  auto pre_out = pretrained->forward(indices);
  PrintSizes("Embedding from_pretrained output", pre_out);
  std::cout << "from_pretrained row 1 matches weight row 1: "
            << (pre_out[0].allclose(weights[1]) ? "true" : "false")
            << std::endl;

  // --- EmbeddingBag: bag of indices -> summed/averaged embedding ---
  // mode: kSum (default), kMean or kMax.
  auto bag = torch::nn::EmbeddingBag(
      torch::nn::EmbeddingBagOptions(vocab_size, embed_dim)
          .mode(torch::kMean));
  // 1D input: one bag containing indices {1, 5, 9, 17}.
  auto bag_input = torch::tensor({1, 5, 9, 17});
  PrintSizes("EmbeddingBag 1-bag output", bag->forward(bag_input));

  // Multiple bags via offsets: bag 0 = {1, 5}, bag 1 = {9, 17}.
  auto bag_inputs = torch::tensor({1, 5, 9, 17});
  auto offsets = torch::tensor({0, 2});
  PrintSizes("EmbeddingBag 2-bag output",
             bag->forward(bag_inputs, offsets));  // [2, 8]

  return 0;
}
