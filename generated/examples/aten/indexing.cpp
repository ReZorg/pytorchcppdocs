// Examples for the "Tensor Indexing" page (api/aten/indexing).
// Covers Tensor::index, Tensor::index_put_, and the index types None,
// Ellipsis, integers, booleans, Slice, and tensor indices.
#include <torch/torch.h>

#include <iostream>

// Brings in None, Ellipsis, and Slice.
using namespace torch::indexing;  // NOLINT(build/namespaces)

int main() {
  // Full example from the doc page.
  torch::Tensor tensor = torch::arange(2 * 3 * 4).reshape({2, 3, 4});
  std::cout << "tensor:\n" << tensor << std::endl;

  // Getter operations: integers.
  torch::Tensor row = tensor.index({0});          // tensor[0]
  torch::Tensor elem = tensor.index({1, 2, 3});   // tensor[1, 2, 3]
  std::cout << "tensor.index({0}):\n" << row << std::endl;
  std::cout << "tensor.index({1, 2, 3}): " << elem.item<int64_t>()
            << std::endl;
  std::cout << "tensor.index({-1}) sizes: " << tensor.index({-1}).sizes()
            << std::endl;

  // Slicing with Slice. The constructor is
  // Slice(start = None, stop = None, step = None).
  torch::Tensor sliced = tensor.index({Slice(), Slice(0, 2)});  // [:, 0:2]
  std::cout << "tensor[:, 0:2] sizes: " << sliced.sizes() << std::endl;
  std::cout << "tensor[1, :, 3]: "
            << tensor.index({1, Slice(), 3}) << std::endl;
  std::cout << "tensor[1::2]: "
            << torch::arange(6).index({Slice(1, None, 2)}) << std::endl;
  // Slice syntax variants.
  std::cout << "Slice() == Slice(None, None): "
            << torch::arange(4).index({Slice(None, None)}).sizes()
            << std::endl;
  std::cout << "Slice(1, None): " << torch::arange(4).index({Slice(1, None)})
            << std::endl;
  std::cout << "Slice(None, 3): " << torch::arange(4).index({Slice(None, 3)})
            << std::endl;
  std::cout << "Slice(1, 3): " << torch::arange(4).index({Slice(1, 3)})
            << std::endl;
  std::cout << "Slice(1, 3, 2): " << torch::arange(4).index({Slice(1, 3, 2)})
            << std::endl;
  std::cout << "Slice(None, None, 2): "
            << torch::arange(6).index({Slice(None, None, 2)}) << std::endl;

  // None inserts a dimension (unsqueeze); Ellipsis covers full dimensions.
  torch::Tensor unsqueezed = tensor.index({None});         // tensor[None]
  torch::Tensor row_none = tensor.index({Slice(), None});  // tensor[:, None]
  std::cout << "tensor[None] sizes: " << unsqueezed.sizes() << std::endl;
  std::cout << "tensor[:, None] sizes: " << row_none.sizes() << std::endl;
  std::cout << "tensor[...] sizes: " << tensor.index({Ellipsis}).sizes()
            << std::endl;
  std::cout << "tensor[\"...\"] sizes: " << tensor.index({"..."}).sizes()
            << std::endl;
  std::cout << "tensor[..., 0]:\n" << tensor.index({Ellipsis, 0}) << std::endl;
  std::cout << "tensor[..., -1] sizes: "
            << tensor.index({Ellipsis, -1}).sizes() << std::endl;

  // Boolean indexing.
  torch::Tensor bool_indexed = torch::arange(3).index({true});
  std::cout << "arange(3)[True] sizes: " << bool_indexed.sizes() << std::endl;

  // Integer tensor (fancy) indexing.
  torch::Tensor idx = torch::tensor({0, 2});
  torch::Tensor gathered = tensor.index({Slice(), idx});  // tensor[:, [0, 2]]
  std::cout << "tensor[:, tensor({0, 2})] sizes: " << gathered.sizes()
            << std::endl;
  std::cout << "tensor.index({tensor({1, 2})}) sizes: "
            << torch::arange(5).index({torch::tensor({1, 2})}).sizes()
            << std::endl;
  // 2-D index tensor: tensor[:, [[0, 1], [4, 3]]] on a length-5 axis.
  torch::Tensor fancy2d =
      torch::arange(2 * 5)
          .reshape({2, 5})
          .index({Slice(), torch::tensor({{0, 1}, {4, 3}})});
  std::cout << "2-D fancy index result sizes: " << fancy2d.sizes()
            << std::endl;

  // Boolean mask indexing.
  torch::Tensor mask = tensor > 10;
  torch::Tensor selected = tensor.index({mask});  // tensor[tensor > 10]
  std::cout << "tensor[tensor > 10] numel: " << selected.numel() << std::endl;

  // Setter operations with index_put_.
  torch::Tensor setter = torch::arange(2 * 3 * 4).reshape({2, 3, 4});
  setter.index_put_({0}, 1);     // tensor[0] = 1
  setter.index_put_({1, 2}, 7);  // tensor[1, 2] = 7
  std::cout << "after index_put_({0}, 1):\n" << setter.index({0}) << std::endl;

  torch::Tensor row_assign = torch::zeros({3, 5});
  row_assign.index_put_({1}, torch::arange(5));  // tensor[1] = arange(5)
  std::cout << "index_put_({1}, arange(5)):\n" << row_assign << std::endl;

  torch::Tensor strided_set = torch::zeros({6});
  strided_set.index_put_({Slice(1, None, 2)}, 1);  // tensor[1::2] = 1
  std::cout << "index_put_({1::2}, 1): " << strided_set << std::endl;

  torch::Tensor mixed = torch::zeros({2, 4});
  // tensor[0, 1::2] = tensor([3., 4.])
  mixed.index_put_({0, Slice(1, None, 2)}, torch::tensor({3., 4.}));
  std::cout << "index_put_({0, 1::2}, {3., 4.}):\n" << mixed << std::endl;

  torch::Tensor zeroed = torch::ones({2, 2});
  zeroed.index_put_({Ellipsis}, 0);  // tensor[...] = 0
  std::cout << "index_put_({...}, 0) sum: " << zeroed.sum().item<int64_t>()
            << std::endl;

  torch::Tensor none_set = torch::zeros({3});
  none_set.index_put_({None}, torch::ones({1, 3}));  // tensor[None] = value
  std::cout << "index_put_({None}, value) sum: "
            << none_set.sum().item<int64_t>() << std::endl;

  torch::Tensor mask_set = torch::arange(6);
  mask_set.index_put_({mask_set > 2}, 0);  // tensor[tensor > 2] = 0
  std::cout << "masked index_put_: " << mask_set << std::endl;

  torch::Tensor fancy_set = torch::zeros({4});
  // tensor[tensor([0, 2])] = value
  fancy_set.index_put_({torch::tensor({0, 2})},
                       torch::tensor({9.0f, 9.0f}));
  std::cout << "fancy index_put_: " << fancy_set << std::endl;

  torch::Tensor slice_fancy = torch::zeros({3, 3}, torch::kInt64);
  // tensor[1:2, tensor([1, 2])] = 5
  slice_fancy.index_put_({Slice(1, 2), torch::tensor({1, 2})}, 5);
  std::cout << "slice+tensor index_put_:\n" << slice_fancy << std::endl;

  // The accumulate parameter adds values instead of replacing them.
  torch::Tensor accum = torch::zeros({4});
  torch::Tensor ones_idx = torch::tensor({1, 1, 3});
  accum.index_put_({ones_idx}, torch::ones({3}), /*accumulate=*/true);
  std::cout << "accumulate=true result: " << accum << std::endl;

  return 0;
}
