// Examples adapted from _sources/api/stable/operators.md.txt
//
// Demonstrates the stable-ABI tensor operations in torch::stable. These keep
// binary compatibility across PyTorch versions. Covered here:
//   - torch::stable::Tensor / torch::stable::Device classes
//   - Creation: empty, empty_like, new_empty, new_zeros, full, from_blob
//   - Manipulation: clone, contiguous, reshape, view, permute, flatten,
//     squeeze, unsqueeze, transpose, select, index_select, narrow, pad
//   - Device/type conversion: to (dtype/device overloads), is_pinned
//   - In-place: fill_, zero_, copy_
//   - Math: matmul, amax (both overloads), sum, sum_out, subtract,
//     bitwise_and, bitwise_or, bitwise_left_shift, bitwise_right_shift,
//     floor_divide
//
// Header-only types (torch::headeronly::ScalarType / Layout / DeviceType /
// MemoryFormat) are used for tensor options.

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace ho = torch::headeronly;
using torch::stable::Device;
using torch::stable::Tensor;

// Print a 1-D float stable tensor.
static void print_vec(const char* label, const Tensor& t) {
  auto flat = torch::stable::flatten(t);
  const float* p = flat.const_data_ptr<float>();
  std::cout << label << " [";
  for (int64_t i = 0; i < flat.numel(); ++i) {
    std::cout << p[i] << (i + 1 < flat.numel() ? ", " : "");
  }
  std::cout << "]" << std::endl;
}

static void print_sizes(const char* label, const Tensor& t) {
  auto s = t.sizes();
  std::cout << label << " [";
  for (size_t i = 0; i < s.size(); ++i) {
    std::cout << s[i] << (i + 1 < s.size() ? ", " : "");
  }
  std::cout << "]" << std::endl;
}

int main() {
  // --- torch::stable::Device ---
  Device cpu_device(ho::DeviceType::CPU);
  std::cout << "Device is_cpu: " << cpu_device.is_cpu()
            << " has_index: " << cpu_device.has_index() << std::endl;

  // --- Creation ---
  auto tensor =
      torch::stable::empty({3, 4}, ho::ScalarType::Float, ho::Layout::Strided,
                           cpu_device, false, ho::MemoryFormat::Contiguous);
  print_sizes("empty({3,4}) sizes:", tensor);

  auto like = torch::stable::empty_like(tensor);
  print_sizes("empty_like sizes:", like);

  auto ne = torch::stable::new_empty(tensor, {2, 5});
  print_sizes("new_empty sizes:", ne);

  auto nz = torch::stable::new_zeros(tensor, {2, 2});
  print_vec("new_zeros:", nz);

  auto full = torch::stable::full({2, 3}, 7.0, ho::ScalarType::Float);
  print_vec("full(7):", full);

  std::vector<float> host = {0, 1, 2, 3, 4, 5};
  auto blob = torch::stable::from_blob(host.data(), {2, 3}, {3, 1}, cpu_device,
                                       ho::ScalarType::Float);
  print_vec("from_blob:", blob);

  // --- Manipulation ---
  auto cloned = torch::stable::clone(full);
  print_vec("clone:", cloned);

  auto reshaped = torch::stable::reshape(full, {3, 2});
  print_sizes("reshape sizes:", reshaped);

  auto viewed = torch::stable::view(full, {6});
  print_sizes("view sizes:", viewed);

  auto perm = torch::stable::permute(full, {1, 0});
  print_sizes("permute sizes:", perm);

  auto flat = torch::stable::flatten(full);
  print_sizes("flatten sizes:", flat);

  auto unsq = torch::stable::unsqueeze(flat, 0);
  print_sizes("unsqueeze sizes:", unsq);

  auto sq = torch::stable::squeeze(unsq, 0);
  print_sizes("squeeze sizes:", sq);

  auto tr = torch::stable::transpose(full, 0, 1);
  print_sizes("transpose sizes:", tr);

  auto sel = torch::stable::select(full, 0, 1);
  print_sizes("select sizes:", sel);

  auto idx = torch::stable::empty({2}, ho::ScalarType::Long,
                                  ho::Layout::Strided, cpu_device);
  int64_t* idxp = static_cast<int64_t*>(idx.mutable_data_ptr());
  idxp[0] = 2;
  idxp[1] = 0;
  auto isel = torch::stable::index_select(flat, 0, idx);
  print_vec("index_select:", isel);

  auto nar = torch::stable::narrow(flat, 0, 1, 3);
  print_vec("narrow:", nar);

  auto padded = torch::stable::pad(nar, {1, 1});
  print_vec("pad:", padded);

  auto contig = torch::stable::contiguous(perm);
  std::cout << "contiguous is_contiguous: " << contig.is_contiguous()
            << std::endl;

  // --- Device and type conversion ---
  auto as_double = torch::stable::to(full, ho::ScalarType::Double);
  std::cout << "to(Double) scalar_type: "
            << torch::headeronly::toString(as_double.scalar_type())
            << std::endl;

  auto on_cpu = torch::stable::to(full, cpu_device);
  std::cout << "to(cpu) get_device_index: " << on_cpu.get_device_index()
            << std::endl;

  std::cout << "is_pinned: " << torch::stable::is_pinned(full) << std::endl;

  // --- In-place operations ---
  auto fillme = torch::stable::empty({3}, ho::ScalarType::Float);
  torch::stable::fill_(fillme, 9.0);
  print_vec("fill_:", fillme);

  torch::stable::zero_(fillme);
  print_vec("zero_:", fillme);

  auto dst = torch::stable::empty({3}, ho::ScalarType::Float);
  auto src = torch::stable::full({3}, 4.0, ho::ScalarType::Float);
  torch::stable::copy_(dst, src);
  print_vec("copy_:", dst);

  // --- Mathematical operations ---
  auto mm_a = torch::stable::full({2, 3}, 1.0, ho::ScalarType::Float);
  auto mm_b = torch::stable::full({3, 2}, 2.0, ho::ScalarType::Float);
  auto mm = torch::stable::matmul(mm_a, mm_b);
  print_vec("matmul:", mm);

  auto am1 = torch::stable::amax(mm, 1, false);
  print_vec("amax(dim=1):", am1);

  auto am_all = torch::stable::amax(mm, {0, 1}, false);
  print_vec("amax(dims={0,1}):", am_all);

  auto s = torch::stable::sum(mm);
  print_vec("sum:", s);

  auto sum_out = torch::stable::empty({2}, ho::ScalarType::Float);
  torch::stable::sum_out(sum_out, mm, std::vector<int64_t>{1}, false);
  print_vec("sum_out(dim=1):", sum_out);

  auto ones22 = torch::stable::full({2, 2}, 1.0, ho::ScalarType::Float);
  auto sub = torch::stable::subtract(mm, ones22, 1.0);
  print_vec("subtract:", sub);

  // Bitwise / shift / floor_divide operate on integer tensors.
  auto int_a = torch::stable::full({3}, 6.0, ho::ScalarType::Int);
  auto int_b = torch::stable::full({3}, 3.0, ho::ScalarType::Int);
  const int32_t* ia = int_a.const_data_ptr<int32_t>();
  std::cout
      << "bitwise_and: "
      << torch::stable::bitwise_and(int_a, int_b).const_data_ptr<int32_t>()[0]
      << std::endl;
  std::cout
      << "bitwise_or: "
      << torch::stable::bitwise_or(int_a, int_b).const_data_ptr<int32_t>()[0]
      << std::endl;
  std::cout << "bitwise_left_shift: "
            << torch::stable::bitwise_left_shift(int_a, int_b)
                   .const_data_ptr<int32_t>()[0]
            << std::endl;
  std::cout << "bitwise_right_shift: "
            << torch::stable::bitwise_right_shift(int_a, int_b)
                   .const_data_ptr<int32_t>()[0]
            << std::endl;
  std::cout
      << "floor_divide: "
      << torch::stable::floor_divide(int_a, int_b).const_data_ptr<int32_t>()[0]
      << std::endl;
  (void)ia;

  // --- view with dtype (reinterpret) ---
  auto as_int_bits =
      torch::stable::view(torch::stable::full({2}, 1.0f, ho::ScalarType::Float),
                          ho::ScalarType::Int);
  std::cout << "view(dtype=Int) scalar_type: "
            << torch::headeronly::toString(as_int_bits.scalar_type())
            << std::endl;

  return 0;
}
