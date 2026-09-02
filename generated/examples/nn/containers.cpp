// Containers: Sequential, ModuleList, ModuleDict, ParameterList,
// ParameterDict, AnyModule, Functional; register_module, named_parameters,
// clone.
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace {

// A small custom module showing register_module and named_parameters.
struct MLPImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};

  explicit MLPImpl(int64_t in_features, int64_t hidden, int64_t out_features) {
    fc1 = register_module("fc1", torch::nn::Linear(in_features, hidden));
    fc2 = register_module("fc2", torch::nn::Linear(hidden, out_features));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    return fc2->forward(x);
  }
};
TORCH_MODULE(MLP);

void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}

}  // namespace

int main() {
  // --- Sequential ---
  torch::nn::Sequential seq(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 8, 3).padding(1)),
      torch::nn::ReLU(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).padding(1)),
      torch::nn::ReLU(), torch::nn::Flatten());
  auto image = torch::randn({2, 1, 8, 8});
  auto seq_out = seq->forward(image);
  PrintSizes("Sequential output", seq_out);

  torch::nn::Sequential mlp_seq(torch::nn::Linear(16 * 8 * 8, 32),
                                torch::nn::ReLU(), torch::nn::Linear(32, 10));
  auto logits = mlp_seq->forward(seq_out);
  PrintSizes("Sequential MLP output", logits);

  // --- ModuleList ---
  // Modules in a ModuleList are registered, but forward() is manual.
  torch::nn::ModuleList layers;
  layers->push_back(torch::nn::Linear(10, 20));
  layers->push_back(torch::nn::Linear(20, 30));
  layers->push_back(torch::nn::Linear(30, 5));
  torch::Tensor x = torch::randn({4, 10});
  for (const auto& layer : *layers) {
    x = layer->as<torch::nn::Linear>()->forward(x);
  }
  PrintSizes("ModuleList output", x);

  // --- ModuleDict: named access via update() / operator[] / at<T>() ---
  torch::nn::ModuleDict dict;
  dict->update(std::vector<std::pair<std::string, std::shared_ptr<torch::nn::Module>>>{
      {"encoder", torch::nn::Linear(10, 16).ptr()},
      {"decoder", torch::nn::Linear(16, 10).ptr()}});
  auto encoded =
      dict->at<torch::nn::LinearImpl>("encoder").forward(torch::randn({4, 10}));
  auto decoded = dict->at<torch::nn::LinearImpl>("decoder").forward(encoded);
  PrintSizes("ModuleDict decoded output", decoded);
  std::cout << "ModuleDict contains \"encoder\": "
            << (dict->contains("encoder") ? "true" : "false") << std::endl;

  // --- ParameterList: parameters stored without wrapping modules ---
  torch::nn::ParameterList param_list;
  param_list->append(torch::randn({3, 4}));
  param_list->append(torch::randn({3, 4}));
  torch::Tensor param_sum = param_list->at(0) + param_list->at(1);
  PrintSizes("ParameterList element sum", param_sum);

  // --- ParameterDict: named parameter storage ---
  torch::nn::ParameterDict param_dict;
  param_dict->insert("scale", torch::ones({3, 4}));
  param_dict->insert("bias", torch::zeros({3, 4}));
  auto affine = param_dict->get("scale") * torch::randn({3, 4}) +
                param_dict->get("bias");
  PrintSizes("ParameterDict affine result", affine);

  // --- AnyModule: type-erased module storage ---
  torch::nn::AnyModule any_linear(torch::nn::Linear(10, 5));
  auto any_out = any_linear.forward(torch::randn({2, 10}));
  PrintSizes("AnyModule output", any_out);

  // --- Functional: wrap a callable so it can live inside Sequential ---
  torch::nn::Sequential with_fn(
      torch::nn::Linear(8, 8),
      torch::nn::Functional([](torch::Tensor t) { return t * 2.0; }),
      torch::nn::ReLU());
  auto fn_out = with_fn->forward(torch::randn({2, 8}));
  PrintSizes("Sequential with Functional output", fn_out);

  // --- register_module / named_parameters ---
  auto model = MLP(784, 64, 10);
  auto out = model->forward(torch::randn({2, 784}));
  PrintSizes("MLP output", out);
  std::cout << "MLP named_parameters:" << std::endl;
  for (const auto& pair : model->named_parameters()) {
    std::cout << "  " << pair.key() << " " << pair.value().sizes()
              << std::endl;
  }

  // --- clone(): deep copy of a module ---
  auto model_copy = std::dynamic_pointer_cast<MLPImpl>(model->clone());
  auto copy_out = model_copy->forward(torch::randn({2, 784}));
  PrintSizes("Cloned MLP output", copy_out);
  std::cout << "Clone has " << model_copy->parameters().size()
            << " parameter tensors" << std::endl;

  return 0;
}
