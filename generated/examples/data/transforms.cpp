// Transforms: torch::data::transforms::Normalize, Stack, Lambda,
// TensorLambda, and BatchLambda, chained onto datasets with .map() and
// composed by nesting; also the Transform / BatchTransform / TensorTransform
// base classes.
//
// Adapted from docs: api/data/transforms.md

#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace {

// --- TensorTransform (base class) ---
// A TensorTransform maps a tensor to a tensor via operator() and can be
// applied directly or chained onto a dataset with .map().
struct SquareTransform : public torch::data::transforms::TensorTransform<> {
  torch::Tensor operator()(torch::Tensor input) override {
    return input * input;
  }
};

// A minimal map-style dataset of Example<> (data + target), like MNIST.
class ExampleDataset
    : public torch::data::datasets::Dataset<ExampleDataset> {
 public:
  torch::data::Example<> get(size_t index) override {
    return {torch::full({4}, static_cast<float>(index)),
            torch::tensor(static_cast<int64_t>(index))};
  }

  torch::optional<size_t> size() const override { return 8; }
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
  auto normalized = torch::data::transforms::Normalize<>(0.5, 0.5)(input);
  std::cout << "Normalize(0.5, 0.5): " << normalized << std::endl;

  // --- TensorLambda: a TensorTransform from a user-provided function ---
  torch::data::transforms::TensorLambda<> add_one(
      [](torch::Tensor t) { return t + 1; });
  std::cout << "TensorLambda (+1): " << add_one(input) << std::endl;

  // --- Lambda: a generic Transform over individual examples ---
  // Lambda::apply() maps one Input to one Output (here an Example).
  torch::data::transforms::Lambda<torch::data::Example<>,
                                  torch::data::Example<>>
      relabel([](torch::data::Example<> ex) {
        ex.target = torch::tensor(7, ex.target.options());
        return ex;
      });
  auto relabeled = relabel.apply({input, torch::tensor(0)});
  std::cout << "Lambda (relabel): target " << relabeled.target << std::endl;

  // --- Stack: a Collation reducing a vector of Examples to one batch ---
  std::vector<torch::data::Example<>> examples;
  for (int i = 0; i < 3; ++i) {
    examples.push_back(
        {torch::full({4}, static_cast<float>(i)), torch::tensor(i)});
  }
  torch::data::transforms::Stack<> stack;
  auto stacked = stack.apply_batch(examples);
  std::cout << "Stack: data sizes " << stacked.data.sizes()
            << " target sizes " << stacked.target.sizes() << std::endl;

  // --- BatchLambda: a BatchTransform mapping a whole batch to a batch ---
  // BatchTransform<Input, Output>::InputBatchType is the batch type itself,
  // so to transform a vector of Examples the template argument is the
  // vector type.
  torch::data::transforms::BatchLambda<std::vector<torch::data::Example<>>,
                                       std::vector<torch::data::Example<>>>
      identity_batch([](std::vector<torch::data::Example<>> batch) {
        return batch;  // e.g. could drop or permute samples
      });
  std::cout << "BatchLambda over "
            << identity_batch.apply_batch(examples).size() << " examples"
            << std::endl;

  // --- Composing transforms by nesting (the docs' "compose") ---
  // There is no compose() helper in this API; transforms are composed by
  // chaining .map() calls (below) or by nesting calls directly.
  auto composed = add_one(normalized);
  std::cout << "Composed (Normalize then +1): " << composed << std::endl;

  // --- Chaining transforms on a dataset with .map() (from the docs) ---
  // Built-in MNIST usage from the docs (NOT run here because constructing
  // MNIST downloads the dataset from the internet):
  //
  //   auto dataset = torch::data::datasets::MNIST("./data")
  //       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
  //       .map(torch::data::transforms::Stack<>());
  //
  // The same chaining pattern with a synthetic Example<> dataset:
  auto dataset = ExampleDataset()
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                     .map(torch::data::transforms::Stack<>());
  auto batch = dataset.get_batch({0, 1, 2, 3});
  std::cout << "map()-chained dataset batch data sizes: "
            << batch.data.sizes() << std::endl;

  std::cout << "transforms example finished" << std::endl;
  return 0;
}
