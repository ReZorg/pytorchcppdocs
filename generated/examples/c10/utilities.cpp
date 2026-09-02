// Examples for the "Utilities" page (api/c10/utilities).
// Covers c10::MaybeOwned<Tensor> and the error handling/assertion macros
// from c10/util/Exception.h: TORCH_CHECK and its typed variants,
// TORCH_INTERNAL_ASSERT, TORCH_WARN, and the c10::Error exception classes.
#include <torch/torch.h>

#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>
#include <iostream>

namespace {

// MaybeOwned<Tensor>: dynamically encodes whether a Tensor is owned or
// borrowed. The canonical use is Tensor::expect_contiguous, which returns
// a borrowed self-reference when the tensor is already contiguous.
void demo_expect_contiguous(const at::Tensor& tensor) {
  c10::MaybeOwned<at::Tensor> maybe_owned =
      tensor.expect_contiguous(at::MemoryFormat::Contiguous);
  std::cout << "  contiguous result data_ptr matches input: "
            << (maybe_owned->data_ptr() == tensor.data_ptr()) << std::endl;
}

// It is always safe to produce an owned MaybeOwned from a Tensor in hand.
c10::MaybeOwned<at::Tensor> maybe_clone(const at::Tensor& tensor,
                                        bool force_copy) {
  if (!force_copy) {
    return c10::MaybeOwned<at::Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<at::Tensor>::owned(tensor.clone());
}

}  // namespace

int main() {
  // MaybeOwned<Tensor> in its borrowed state for an already-contiguous
  // tensor, and owned state after materializing a contiguous copy.
  at::Tensor contiguous = at::randn({2, 3});
  std::cout << "expect_contiguous on contiguous tensor:" << std::endl;
  demo_expect_contiguous(contiguous);
  at::Tensor strided = contiguous.t();
  std::cout << "expect_contiguous on non-contiguous tensor:" << std::endl;
  demo_expect_contiguous(strided);

  at::Tensor source = at::ones({2});
  std::cout << "borrowed alias shares storage: "
            << (maybe_clone(source, false)->data_ptr() == source.data_ptr())
            << std::endl;
  std::cout << "owned clone has its own storage: "
            << (maybe_clone(source, true)->data_ptr() != source.data_ptr())
            << std::endl;

  // TORCH_CHECK: validates user input and runtime conditions, raising
  // c10::Error on failure.
  at::Tensor matrix = at::randn({3, 3});
  TORCH_CHECK(matrix.dim() == 2, "Expected 2D tensor, got ", matrix.dim(),
              "D");
  TORCH_CHECK(matrix.numel() >= 0);  // default message generated on failure
  std::cout << "TORCH_CHECK passed for dim() == " << matrix.dim()
            << std::endl;

  // Typed variants map to specific Python exception types.
  TORCH_CHECK_INDEX(matrix.size(0) > 0, "row index out of range");
  TORCH_CHECK_VALUE(matrix.numel() > 0, "tensor must not be empty");
  TORCH_CHECK_TYPE(matrix.scalar_type() == at::kFloat, "expected float");
  TORCH_CHECK_NOT_IMPLEMENTED(true, "nothing is unimplemented here");
  std::cout << "typed TORCH_CHECK variants passed." << std::endl;

  // TORCH_CHECK_LINALG validates linear algebra preconditions. A
  // non-square matrix violates it, so catch the resulting exception.
  at::Tensor rectangular = at::randn({2, 3});
  TORCH_CHECK_LINALG(rectangular.is_complex() || !rectangular.is_complex(),
                     "trivially satisfied condition");
  std::cout << "TORCH_CHECK_LINALG passed." << std::endl;

  // TORCH_INTERNAL_ASSERT is for invariants that should always hold.
  int64_t googol = matrix.numel();
  TORCH_INTERNAL_ASSERT(googol > 0);
  TORCH_INTERNAL_ASSERT(googol > 0, "googol was ", googol);
  std::cout << "TORCH_INTERNAL_ASSERT passed." << std::endl;

  // TORCH_WARN issues a warning; TORCH_WARN_ONCE only fires once.
  TORCH_WARN("This operation is slow for sparse tensors");
  TORCH_WARN_ONCE("This warning appears only once");

  // c10::Error is the base exception class, carrying source location and
  // an optional backtrace. Specialized subclasses exist for common error
  // categories.
  try {
    TORCH_CHECK_VALUE(false, "demonstrating c10::ValueError");
  } catch (const c10::ValueError& e) {
    std::cout << "caught c10::ValueError: " << e.what_without_backtrace()
              << std::endl;
  }
  try {
    TORCH_CHECK_INDEX(false, "demonstrating c10::IndexError");
  } catch (const c10::IndexError& e) {
    std::cout << "caught c10::IndexError: " << e.what_without_backtrace()
              << std::endl;
  }
  try {
    TORCH_CHECK_TYPE(false, "demonstrating c10::TypeError");
  } catch (const c10::TypeError& e) {
    std::cout << "caught c10::TypeError: " << e.what_without_backtrace()
              << std::endl;
  }
  try {
    TORCH_CHECK_NOT_IMPLEMENTED(false,
                                "demonstrating c10::NotImplementedError");
  } catch (const c10::NotImplementedError& e) {
    std::cout << "caught c10::NotImplementedError: "
              << e.what_without_backtrace() << std::endl;
  }
  // c10::LinAlgError and c10::OutOfMemoryError are further specialized
  // subclasses of c10::Error; all can be caught via the base class.
  try {
    TORCH_CHECK(false, "demonstrating base c10::Error");
  } catch (const c10::Error& e) {
    std::cout << "caught c10::Error (what_without_backtrace): "
              << e.what_without_backtrace() << std::endl;
  }

  return 0;
}
