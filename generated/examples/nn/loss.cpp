// Loss functions: regression (L1, MSE, SmoothL1, Huber), classification
// (CrossEntropy, NLL, BCE, BCEWithLogits, margins), KLDiv, CTC, embedding
// and ranking losses.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintLoss(const std::string& name, const torch::Tensor& loss) {
  std::cout << name << ": " << loss.item<float>() << std::endl;
}
}  // namespace

int main() {
  const int64_t batch = 4, num_classes = 5;
  auto pred = torch::randn({batch, num_classes});
  auto target = torch::randn({batch, num_classes});

  // --- Regression losses ---
  auto l1 = torch::nn::L1Loss();
  PrintLoss("L1Loss", l1->forward(pred, target));

  auto mse = torch::nn::MSELoss();
  PrintLoss("MSELoss", mse->forward(pred, target));

  // Reduction options: kNone / kMean / kSum.
  auto mse_sum =
      torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kSum));
  PrintLoss("MSELoss (sum)", mse_sum->forward(pred, target));

  auto smooth_l1 = torch::nn::SmoothL1Loss();
  PrintLoss("SmoothL1Loss", smooth_l1->forward(pred, target));

  auto huber = torch::nn::HuberLoss(torch::nn::HuberLossOptions().delta(1.0));
  PrintLoss("HuberLoss", huber->forward(pred, target));

  // --- Classification losses ---
  // CrossEntropyLoss combines LogSoftmax + NLLLoss; targets are class ids.
  auto logits = torch::randn({batch, num_classes});
  auto class_ids = torch::randint(0, num_classes, {batch});
  auto cross_entropy = torch::nn::CrossEntropyLoss();
  PrintLoss("CrossEntropyLoss", cross_entropy->forward(logits, class_ids));

  // NLLLoss expects log-probabilities as input.
  auto log_probs = torch::log_softmax(logits, /*dim=*/1);
  auto nll = torch::nn::NLLLoss();
  PrintLoss("NLLLoss", nll->forward(log_probs, class_ids));

  // BCELoss expects probabilities in [0, 1].
  auto probs = torch::sigmoid(torch::randn({batch, 1}));
  auto binary_targets = torch::randint(0, 2, {batch, 1}).to(torch::kFloat);
  auto bce = torch::nn::BCELoss();
  PrintLoss("BCELoss", bce->forward(probs, binary_targets));

  // BCEWithLogitsLoss applies sigmoid internally (more stable).
  auto bce_logits = torch::nn::BCEWithLogitsLoss();
  PrintLoss("BCEWithLogitsLoss",
            bce_logits->forward(torch::randn({batch, 1}), binary_targets));

  // KLDivLoss: input is log-probabilities, target is probabilities.
  auto kl = torch::nn::KLDivLoss();
  PrintLoss("KLDivLoss", kl->forward(log_probs, torch::softmax(target, 1)));

  // --- CTCLoss: sequence-to-sequence without alignment ---
  auto ctc = torch::nn::CTCLoss();
  auto ctc_log_probs = torch::log_softmax(
      torch::randn({/*T=*/10, /*N=*/2, /*C=*/8}), /*dim=*/2);
  auto ctc_targets = torch::tensor({{1, 2, 3}, {4, 5, 0}}, torch::kInt64);
  auto input_lengths = torch::tensor({10, 10}, torch::kInt64);
  auto target_lengths = torch::tensor({3, 2}, torch::kInt64);
  PrintLoss("CTCLoss", ctc->forward(ctc_log_probs, ctc_targets, input_lengths,
                                    target_lengths));

  // --- PoissonNLLLoss: target is drawn from a Poisson distribution ---
  auto poisson = torch::nn::PoissonNLLLoss();
  PrintLoss("PoissonNLLLoss",
            poisson->forward(torch::randn({batch, 4}),
                             torch::randint(0, 5, {batch, 4}).to(torch::kFloat)));

  // --- Margin and embedding losses ---
  auto margin_ranking = torch::nn::MarginRankingLoss(
      torch::nn::MarginRankingLossOptions().margin(0.5));
  auto x1 = torch::randn({4});
  auto x2 = torch::randn({4});
  auto signs = torch::tensor({1.0, -1.0, 1.0, -1.0});
  PrintLoss("MarginRankingLoss", margin_ranking->forward(x1, x2, signs));

  auto hinge = torch::nn::HingeEmbeddingLoss(
      torch::nn::HingeEmbeddingLossOptions().margin(1.0));
  PrintLoss("HingeEmbeddingLoss", hinge->forward(torch::randn({4}), signs));

  auto cosine_embed = torch::nn::CosineEmbeddingLoss(
      torch::nn::CosineEmbeddingLossOptions().margin(0.0));
  PrintLoss("CosineEmbeddingLoss",
            cosine_embed->forward(torch::randn({4, 8}), torch::randn({4, 8}),
                                  signs));

  auto multi_margin = torch::nn::MultiMarginLoss(
      torch::nn::MultiMarginLossOptions().margin(1.0));
  PrintLoss("MultiMarginLoss", multi_margin->forward(logits, class_ids));

  // MultiLabelMarginLoss: targets are class indices padded with -1.
  auto ml_margin = torch::nn::MultiLabelMarginLoss();
  auto ml_targets = torch::tensor({{0, 1, -1, -1, -1}, {2, 3, 4, -1, -1},
                                   {1, -1, -1, -1, -1}, {0, 2, 4, -1, -1}});
  PrintLoss("MultiLabelMarginLoss", ml_margin->forward(logits, ml_targets));

  // MultiLabelSoftMarginLoss: multi-hot targets.
  auto ml_soft = torch::nn::MultiLabelSoftMarginLoss();
  auto multi_hot = torch::randint(0, 2, {batch, num_classes}).to(torch::kFloat);
  PrintLoss("MultiLabelSoftMarginLoss",
            ml_soft->forward(logits, multi_hot));

  auto soft_margin = torch::nn::SoftMarginLoss();
  PrintLoss("SoftMarginLoss", soft_margin->forward(torch::randn({4}), signs));

  // --- Triplet losses: anchor, positive, negative ---
  auto anchor = torch::randn({4, 8});
  auto positive = torch::randn({4, 8});
  auto negative = torch::randn({4, 8});
  auto triplet = torch::nn::TripletMarginLoss(
      torch::nn::TripletMarginLossOptions().margin(1.0).p(2));
  PrintLoss("TripletMarginLoss",
            triplet->forward(anchor, positive, negative));

  // Custom distance function variant.
  auto triplet_d = torch::nn::TripletMarginWithDistanceLoss(
      torch::nn::TripletMarginWithDistanceLossOptions()
          .margin(1.0)
          .distance_function([](const torch::Tensor& a, const torch::Tensor& b) {
            return torch::pairwise_distance(a, b, /*p=*/2);
          }));
  PrintLoss("TripletMarginWithDistanceLoss",
            triplet_d->forward(anchor, positive, negative));

  return 0;
}
