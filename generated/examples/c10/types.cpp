// Examples for the "Core Types" page (api/c10/types).
// Covers c10::ArrayRef, OptionalArrayRef, optional, Half, Dict, List,
// IListRef, and IValue.
#include <torch/torch.h>

#include <ATen/core/Dict.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

// OptionalArrayRef: an optional non-owning array view.
void print_sizes(c10::OptionalArrayRef<int64_t> sizes = c10::nullopt) {
  if (sizes.has_value()) {
    for (auto s : sizes.value()) {
      std::cout << "  dim size: " << s << std::endl;
    }
  } else {
    std::cout << "  no sizes provided" << std::endl;
  }
}

}  // namespace

int main() {
  // ArrayRef: a non-owning view over a contiguous array.
  std::vector<int64_t> sizes = {3, 4, 5};
  c10::ArrayRef<int64_t> sizes_ref(sizes);
  std::cout << "ArrayRef size: " << sizes_ref.size()
            << ", front: " << sizes_ref.front() << std::endl;

  // Initializer lists implicitly convert to ArrayRef.
  at::Tensor tensor = at::zeros({3, 4, 5});
  std::cout << "at::zeros({3, 4, 5}).sizes(): " << tensor.sizes()
            << std::endl;

  // OptionalArrayRef.
  std::cout << "print_sizes({2, 3}):" << std::endl;
  print_sizes({2, 3});
  std::cout << "print_sizes(nullopt):" << std::endl;
  print_sizes();

  // c10::optional: a wrapper that may or may not contain a value.
  c10::optional<int64_t> maybe_dim = c10::nullopt;
  std::cout << "maybe_dim.has_value(): " << maybe_dim.has_value()
            << std::endl;
  std::cout << "maybe_dim.value_or(-1): " << maybe_dim.value_or(-1)
            << std::endl;
  maybe_dim = 7;
  std::cout << "maybe_dim.value(): " << maybe_dim.value() << std::endl;

  // c10::Half: 16-bit IEEE 754 half-precision float.
  c10::Half h = 3.14f;
  float f = static_cast<float>(h);
  std::cout << "c10::Half(3.14f) as float: " << f << std::endl;
  at::Tensor half_tensor = at::ones({2}, at::kHalf);
  std::cout << "half tensor dtype: " << half_tensor.dtype() << std::endl;

  // c10::Dict: an ordered hash map storing IValue elements; copies share
  // the same underlying storage.
  c10::Dict<std::string, at::Tensor> named_tensors;
  named_tensors.insert("weight", torch::randn({3, 3}));
  named_tensors.insert("bias", torch::zeros({3}));
  std::cout << "dict contains \"weight\": " << named_tensors.contains("weight")
            << std::endl;
  at::Tensor w = named_tensors.at("weight");
  std::cout << "weight sizes: " << w.sizes() << std::endl;
  for (const auto& entry : named_tensors) {
    std::cout << "dict entry " << entry.key() << ": " << entry.value().sizes()
              << std::endl;
  }

  // c10::List: a type-safe list container backed by IValue elements.
  c10::List<at::Tensor> tensor_list;
  tensor_list.push_back(torch::randn({2, 3}));
  tensor_list.push_back(torch::zeros({2, 3}));
  at::Tensor first = tensor_list.get(0);
  std::cout << "List size: " << tensor_list.size()
            << ", first sizes: " << first.sizes() << std::endl;

  c10::List<int64_t> int_list;
  int_list.push_back(1);
  int_list.push_back(2);
  int_list.push_back(3);
  std::cout << "int List: [" << int_list.get(0) << ", " << int_list.get(1)
            << ", " << int_list.get(2) << "]" << std::endl;

  // c10::IListRef: a lightweight unified reference over list-like types.
  std::vector<at::Tensor> vec = {torch::randn({2}), torch::randn({3})};
  c10::IListRef<at::Tensor> ref(vec);
  for (const auto& t : ref) {
    std::cout << "IListRef element sizes: " << t.sizes() << std::endl;
  }

  // c10::IValue: a type-erased container for tensors, scalars, lists,
  // dicts, tuples, and more.
  c10::IValue tensor_val = at::ones({2, 2});
  std::cout << "IValue isTensor(): " << tensor_val.isTensor() << std::endl;
  if (tensor_val.isTensor()) {
    at::Tensor t2 = tensor_val.toTensor();
    std::cout << "toTensor().sizes(): " << t2.sizes() << std::endl;
  }

  c10::IValue int_val(int64_t(42));
  std::cout << "IValue isInt(): " << int_val.isInt()
            << ", toInt(): " << int_val.toInt() << std::endl;

  c10::IValue double_val(2.5);
  std::cout << "IValue isDouble(): " << double_val.isDouble()
            << ", toDouble(): " << double_val.toDouble() << std::endl;

  c10::IValue bool_val(true);
  std::cout << "IValue isBool(): " << bool_val.isBool()
            << ", toBool(): " << bool_val.toBool() << std::endl;

  c10::IValue string_val("hello");
  std::cout << "IValue isString(): " << string_val.isString()
            << ", toString(): " << string_val.toStringRef() << std::endl;

  c10::List<int64_t> backing;
  backing.push_back(1);
  backing.push_back(2);
  c10::IValue list_val(backing);
  std::cout << "IValue isList(): " << list_val.isList()
            << ", list size: " << list_val.toList().size() << std::endl;

  c10::Dict<std::string, int64_t> dict_backing;
  dict_backing.insert("one", 1);
  c10::IValue dict_val(dict_backing);
  std::cout << "IValue isGenericDict(): " << dict_val.isGenericDict()
            << std::endl;
  if (dict_val.isGenericDict()) {
    std::cout << "dict[\"one\"]: "
              << dict_val.toGenericDict().at("one").toInt() << std::endl;
  }

  auto tuple_backing = std::make_tuple(int64_t(1), 2.0);
  c10::IValue tuple_val(tuple_backing);
  std::cout << "IValue isTuple(): " << tuple_val.isTuple()
            << ", tuple size: " << tuple_val.toTuple()->elements().size()
            << std::endl;

  c10::IValue none_val;
  std::cout << "IValue isNone(): " << none_val.isNone() << std::endl;

  return 0;
}
