// Transforms: torch::data::transforms::Normalize, Stack, Lambda,
// TensorLambda, and BatchLambda, chained onto datasets with .map() and
// composed with torch::data::transforms::compose; also the Transform /
// BatchTransform / TensorTransform base classes.
//
// Adapted from docs: api/data/transforms.md

#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace {

// --- TensorTransform (base class) ---
// A TensorTransform maps a tensor to a tensor and can be applied directly
// or chained onto a dataset with .map().
struct SquareTransform
    : public torch::data::transforms::TensorTransform<> {
  torch::Tensor operator()(torch::Tensor input) override {
    return input * input;
  }
};

}  // namespace

int main() {
  torch::manual_seed(42);

  // --- TensorTransform applied directly to a tensor ---
  SquareTransform square;
  auto input = torch::tensor({1.0, 2.0, 3.0});
  std::cout << "TensorTransform square: " << input << " -> " << square(input)
            << std::endl;

  // --- Normalize: (tensor - mean) / std ---
  auto normalized =
      torch::data::transforms::Normalize<>(0.5, 0.5)(input);
  std::cout << "Normalize(0.5, 0.5): " << normalized << std::endl;

  // --- Lambda / TensorLambda: wrap arbitrary functions ---
  torch::data::transforms::TensorLambda<decltype([](torch::Tensor t) {
    return t + 1;
  })>
      add_one([](torch::Tensor t) { return t + 1; });
  std::cout << "TensorLambda (+1): " << add_one(input) << std::endl;

  // Lambda is the generic (non-tensor) form, here over an Example.
  auto relabel = torch::data::transforms::Lambda<
      torch::data::Example<>, torch::data::Example<>>(
      [](torch::data::Example<> ex) {
        ex.target = torch::tensor(7, ex.target.options());
        return ex;
      });
  auto relabeled = relabel({input, torch::tensor(0)});
  std::cout << "Lambda (relabel): target " << relabeled.target << std::endl;

  // --- Stack: collate a vector of Examples into one batched Example ---
  std::vector<torch::data::Example<>> examples;
  for (int i = 0; i < 3; ++i) {
    examples.push_back(
        {torch::full({4}, static_cast<float>(i)), torch::tensor(i)});
  }
  auto stacked = torch::data::transforms::Stack<>()(examples);
  std::cout << "Stack: data sizes " << stacked.data.sizes()
            << " target sizes " << stacked.target.sizes() << std::endl;

  // --- BatchLambda: a BatchTransform over a whole vector of Examples ---
  torch::data::transforms::BatchLambda<
      torch::data::Example<>, torch::data::Example<>>
      identity_batch([](std::vector<torch::data::Example<>> batch) {
        return batch;  // e.g. could drop or permute samples
      });
  std::cout << "BatchLambda over " << identity_batch(examples).size()
            << " examples" << std::endl;

  // --- compose: fuse transforms into one ---
  auto pipeline = torch::data::transforms::compose(
      torch::data::transforms::Normalize<>(0.0, 1.0),
      torch::data::transforms::TensorLambda<decltype(
          [](torch::Tensor t) { return t * 2; })>(
          [](torch::Tensor t) { return t * 2; }));
  std::cout << "compose(Normalize, *2): " << pipeline(input) << std::endl;

  // --- Chaining transforms on a dataset with .map() (from the docs) ---
  // Built-in MNIST usage from the docs (NOT run here because constructing
  // MNIST downloads the dataset from the internet):
  //
  //   auto dataset = torch::data::datasets::MNIST("./data")
  //       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
  //       .map(torch::data::transforms::Stack<>());
  //
  // The same chaining pattern with a synthetic TensorDataset:
  auto dataset =
      torch::data::datasets::TensorDataset(
          torch::arange(8 * 4, torch::kFloat32).reshape({8, 4}) / 32.0,
          torch::arange(8, torch::kInt64))
          .map(torch::data::transforms::Normalize<>(0.5, 0.5))
          .map(torch::data::transforms::Stack<>());
  auto batch = dataset.get_batch({0, 1, 2, 3});
  std::cout << "map()-chained dataset batch data sizes: "
            << batch.data.sizes() << std::endl;

  std::cout << "transforms example finished" << std::endl;
  return 0;
}
