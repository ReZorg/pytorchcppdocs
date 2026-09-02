// Functional API: stateless torch::nn::functional operations. Weights are
// passed explicitly instead of being held by modules.
#include <torch/nn/functional.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace F = torch::nn::functional;

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
void PrintScalar(const std::string& name, const torch::Tensor& t) {
  std::cout << name << ": " << t.item<float>() << std::endl;
}
}  // namespace

int main() {
  auto x = torch::randn({2, 8});
  auto x4 = torch::randn({2, 3, 8, 8});

  // ==== Activation functions (stateless) ====
  PrintSizes("F::relu", F::relu(x));
  PrintSizes("F::relu6", F::relu6(x));
  PrintSizes("F::leaky_relu",
             F::leaky_relu(x, F::LeakyReLUFuncOptions().negative_slope(0.01)));
  PrintSizes("F::elu", F::elu(x));
  PrintSizes("F::selu", F::selu(x));
  PrintSizes("F::celu", F::celu(x));
  PrintSizes("F::gelu", F::gelu(x));
  PrintSizes("F::silu", F::silu(x));
  PrintSizes("F::mish", F::mish(x));
  PrintSizes("F::prelu", F::prelu(x, torch::full({8}, 0.25)));
  PrintSizes("F::rrelu",
             F::rrelu(x, F::RReLUFuncOptions().lower(0.125).upper(0.333)
                             .training(true)));
  PrintSizes("F::hardshrink", F::hardshrink(x));
  PrintSizes("F::hardtanh", F::hardtanh(x));
  PrintSizes("F::logsigmoid", F::logsigmoid(x));
  PrintSizes("F::glu", F::glu(x));  // halves last dim
  PrintSizes("F::softplus", F::softplus(x));
  PrintSizes("F::softshrink", F::softshrink(x));
  PrintSizes("F::softsign", F::softsign(x));
  PrintSizes("F::tanhshrink", F::tanhshrink(x));
  PrintSizes("F::threshold", F::threshold(x, F::ThresholdFuncOptions(0, 0)));
  PrintSizes("F::softmax", F::softmax(x, F::SoftmaxFuncOptions(/*dim=*/1)));
  PrintSizes("F::softmin", F::softmin(x, F::SoftminFuncOptions(/*dim=*/1)));
  PrintSizes("F::log_softmax",
             F::log_softmax(x, F::LogSoftmaxFuncOptions(/*dim=*/1)));
  PrintSizes("F::gumbel_softmax",
             F::gumbel_softmax(x, F::GumbelSoftmaxFuncOptions().tau(1.0)));

  // ==== Convolution functions (explicit weight/bias tensors) ====
  auto w1 = torch::randn({8, 4, 3});      // [out, in/groups, kL]
  auto w2 = torch::randn({8, 3, 3, 3});   // [out, in/groups, kH, kW]
  auto w3 = torch::randn({4, 2, 3, 3, 3});
  auto b8 = torch::randn({8});
  PrintSizes("F::conv1d",
             F::conv1d(torch::randn({2, 4, 16}), w1,
                       F::Conv1dFuncOptions().stride(1).padding(1).bias(b8)));
  PrintSizes("F::conv2d",
             F::conv2d(x4, w2, F::Conv2dFuncOptions().stride(1).padding(1)));
  PrintSizes("F::conv3d", F::conv3d(torch::randn({1, 2, 6, 6, 6}), w3,
                                    F::Conv3dFuncOptions().padding(1)));
  PrintSizes("F::conv_transpose1d",
             F::conv_transpose1d(torch::randn({2, 8, 8}),
                                 torch::randn({8, 4, 4}),
                                 F::ConvTranspose1dFuncOptions().stride(2)
                                     .padding(1)));
  PrintSizes("F::conv_transpose2d",
             F::conv_transpose2d(torch::randn({2, 8, 4, 4}),
                                 torch::randn({8, 4, 4, 4}),
                                 F::ConvTranspose2dFuncOptions().stride(2)
                                     .padding(1)));
  PrintSizes("F::conv_transpose3d",
             F::conv_transpose3d(torch::randn({1, 4, 4, 4, 4}),
                                 torch::randn({4, 2, 4, 4, 4}),
                                 F::ConvTranspose3dFuncOptions().stride(2)
                                     .padding(1)));

  // ==== Pooling functions ====
  auto x1d = torch::randn({2, 3, 8});
  auto x3d = torch::randn({1, 3, 4, 4, 4});
  PrintSizes("F::avg_pool1d",
             F::avg_pool1d(x1d, F::AvgPool1dFuncOptions(2).stride(2)));
  PrintSizes("F::avg_pool2d",
             F::avg_pool2d(x4, F::AvgPool2dFuncOptions(2).stride(2)));
  PrintSizes("F::avg_pool3d",
             F::avg_pool3d(x3d, F::AvgPool3dFuncOptions(2).stride(2)));
  PrintSizes("F::max_pool1d",
             F::max_pool1d(x1d, F::MaxPool1dFuncOptions(2).stride(2)));
  PrintSizes("F::max_pool2d",
             F::max_pool2d(x4, F::MaxPool2dFuncOptions(2).stride(2)));
  PrintSizes("F::max_pool3d",
             F::max_pool3d(x3d, F::MaxPool3dFuncOptions(2).stride(2)));

  // *_with_indices variants return pooled values plus argmax indices,
  // which max_unpool uses to place values back into unpooled positions.
  auto pi1 = F::max_pool1d_with_indices(x1d, F::MaxPool1dFuncOptions(2).stride(2));
  auto pi2 = F::max_pool2d_with_indices(x4, F::MaxPool2dFuncOptions(2).stride(2));
  auto pi3 = F::max_pool3d_with_indices(x3d, F::MaxPool3dFuncOptions(2).stride(2));
  PrintSizes("F::max_pool1d_with_indices output", std::get<0>(pi1));
  PrintSizes("F::max_pool2d_with_indices output", std::get<0>(pi2));
  PrintSizes("F::max_pool3d_with_indices output", std::get<0>(pi3));
  PrintSizes("F::max_unpool1d",
             F::max_unpool1d(std::get<0>(pi1), std::get<1>(pi1),
                             F::MaxUnpool1dFuncOptions(2).stride(2)));
  PrintSizes("F::max_unpool2d",
             F::max_unpool2d(std::get<0>(pi2), std::get<1>(pi2),
                             F::MaxUnpool2dFuncOptions(2).stride(2)));
  PrintSizes("F::max_unpool3d",
             F::max_unpool3d(std::get<0>(pi3), std::get<1>(pi3),
                             F::MaxUnpool3dFuncOptions(2).stride(2)));

  PrintSizes("F::adaptive_max_pool1d",
             F::adaptive_max_pool1d(x1d, F::AdaptiveMaxPool1dFuncOptions(4)));
  PrintSizes("F::adaptive_max_pool2d",
             F::adaptive_max_pool2d(x4, F::AdaptiveMaxPool2dFuncOptions({3, 3})));
  PrintSizes("F::adaptive_max_pool3d",
             F::adaptive_max_pool3d(x3d,
                                    F::AdaptiveMaxPool3dFuncOptions({2, 2, 2})));
  PrintSizes("F::adaptive_avg_pool1d",
             F::adaptive_avg_pool1d(x1d, F::AdaptiveAvgPool1dFuncOptions(4)));
  PrintSizes("F::adaptive_avg_pool2d",
             F::adaptive_avg_pool2d(x4, F::AdaptiveAvgPool2dFuncOptions({3, 3})));
  PrintSizes("F::adaptive_avg_pool3d",
             F::adaptive_avg_pool3d(x3d,
                                    F::AdaptiveAvgPool3dFuncOptions({2, 2, 2})));

  PrintSizes("F::fractional_max_pool2d",
             F::fractional_max_pool2d(
                 x4, F::FractionalMaxPool2dFuncOptions(2).output_size(
                         std::vector<int64_t>({4, 4}))));
  PrintSizes("F::fractional_max_pool3d",
             F::fractional_max_pool3d(
                 x3d, F::FractionalMaxPool3dFuncOptions(2).output_size(
                          std::vector<int64_t>({2, 2, 2}))));

  PrintSizes("F::lp_pool1d",
             F::lp_pool1d(x1d, F::LPPool1dFuncOptions(2.0, 2).stride(2)));
  PrintSizes("F::lp_pool2d",
             F::lp_pool2d(x4, F::LPPool2dFuncOptions(2.0, 2).stride(2)));
  PrintSizes("F::lp_pool3d",
             F::lp_pool3d(x3d, F::LPPool3dFuncOptions(2.0, 2).stride(2)));

  // ==== Linear functions ====
  PrintSizes("F::linear", F::linear(x, torch::randn({4, 8}), torch::randn({4})));
  PrintSizes("F::bilinear",
             F::bilinear(torch::randn({2, 8}), torch::randn({2, 6}),
                         torch::randn({4, 8, 6}), torch::randn({4})));

  // ==== Dropout functions (training flag passed explicitly) ====
  PrintSizes("F::dropout",
             F::dropout(x, F::DropoutFuncOptions().p(0.5).training(true)));
  PrintSizes("F::dropout2d",
             F::dropout2d(x4, F::Dropout2dFuncOptions().p(0.3).training(true)));
  PrintSizes("F::dropout3d",
             F::dropout3d(x3d, F::Dropout3dFuncOptions().p(0.3).training(true)));
  PrintSizes("F::alpha_dropout",
             F::alpha_dropout(x, F::AlphaDropoutFuncOptions().p(0.2)
                                     .training(true)));
  PrintSizes("F::feature_alpha_dropout",
             F::feature_alpha_dropout(x4, F::FeatureAlphaDropoutFuncOptions()
                                              .p(0.2)
                                              .training(true)));

  // ==== Embedding functions ====
  auto embed_weight = torch::randn({100, 8});
  auto ids = torch::tensor({1, 2, 3, 0});
  PrintSizes("F::one_hot", F::one_hot(ids, 100).to(torch::kFloat));
  PrintSizes("F::embedding", F::embedding(ids, embed_weight,
                                          F::EmbeddingFuncOptions()
                                              .padding_idx(0)));
  // 2D input: each row is one bag of indices.
  PrintSizes("F::embedding_bag",
             F::embedding_bag(torch::tensor({{1, 5, 9}, {2, 7, 0}}),
                              embed_weight,
                              F::EmbeddingBagFuncOptions().mode(torch::kMean)));

  // ==== Normalization functions (stats/affine params passed in) ====
  PrintSizes("F::batch_norm",
             F::batch_norm(x4, torch::zeros({3}), torch::ones({3}),
                           F::BatchNormFuncOptions()
                               .weight(torch::ones({3}))
                               .bias(torch::zeros({3}))
                               .training(true)
                               .momentum(0.1)
                               .eps(1e-5)));
  PrintSizes("F::instance_norm", F::instance_norm(x4));
  PrintSizes("F::layer_norm",
             F::layer_norm(x, F::LayerNormFuncOptions({8})
                                  .weight(torch::ones({8}))
                                  .bias(torch::zeros({8}))));
  PrintSizes("F::group_norm",
             F::group_norm(torch::randn({2, 8, 4, 4}),
                           F::GroupNormFuncOptions(/*num_groups=*/4)));
  PrintSizes("F::local_response_norm",
             F::local_response_norm(x4, F::LocalResponseNormFuncOptions(5)));
  PrintSizes("F::normalize",
             F::normalize(x, F::NormalizeFuncOptions().p(2).dim(1)));

  // ==== Loss functions ====
  auto target = torch::randn({2, 8});
  auto logits = torch::randn({4, 5});
  auto labels = torch::randint(0, 5, {4});
  auto log_probs = F::log_softmax(logits, F::LogSoftmaxFuncOptions(1));
  auto probs = torch::sigmoid(torch::randn({4, 1}));
  auto bin_targets = torch::randint(0, 2, {4, 1}).to(torch::kFloat);
  auto signs = torch::tensor({1.0, -1.0, 1.0, -1.0});

  PrintScalar("F::l1_loss", F::l1_loss(x, target));
  PrintScalar("F::mse_loss", F::mse_loss(x, target));
  PrintScalar("F::binary_cross_entropy",
              F::binary_cross_entropy(probs, bin_targets));
  PrintScalar("F::binary_cross_entropy_with_logits",
              F::binary_cross_entropy_with_logits(torch::randn({4, 1}),
                                                  bin_targets));
  PrintScalar("F::cross_entropy", F::cross_entropy(logits, labels));
  PrintScalar("F::nll_loss", F::nll_loss(log_probs, labels));
  PrintScalar("F::kl_div", F::kl_div(log_probs, torch::softmax(logits, 1)));
  PrintScalar("F::smooth_l1_loss",
              F::smooth_l1_loss(x, target, F::SmoothL1LossFuncOptions()));
  PrintScalar("F::huber_loss", F::huber_loss(x, target));
  PrintScalar("F::hinge_embedding_loss",
              F::hinge_embedding_loss(torch::randn({4}), signs));
  PrintScalar("F::multi_margin_loss", F::multi_margin_loss(logits, labels));
  PrintScalar("F::cosine_embedding_loss",
              F::cosine_embedding_loss(torch::randn({4, 8}),
                                       torch::randn({4, 8}), signs));
  PrintScalar("F::margin_ranking_loss",
              F::margin_ranking_loss(torch::randn({4}), torch::randn({4}),
                                     signs));
  PrintScalar("F::multilabel_margin_loss",
              F::multilabel_margin_loss(
                  logits, torch::tensor({{0, 1, -1, -1, -1}, {2, 3, -1, -1, -1},
                                         {1, 4, -1, -1, -1},
                                         {0, 2, 4, -1, -1}})));
  PrintScalar("F::soft_margin_loss",
              F::soft_margin_loss(torch::randn({4}), signs));
  PrintScalar("F::multilabel_soft_margin_loss",
              F::multilabel_soft_margin_loss(
                  logits, torch::randint(0, 2, {4, 5}).to(torch::kFloat)));
  PrintScalar("F::triplet_margin_loss",
              F::triplet_margin_loss(torch::randn({4, 8}), torch::randn({4, 8}),
                                     torch::randn({4, 8})));
  PrintScalar("F::triplet_margin_with_distance_loss",
              F::triplet_margin_with_distance_loss(
                  torch::randn({4, 8}), torch::randn({4, 8}),
                  torch::randn({4, 8})));
  PrintScalar("F::ctc_loss",
              F::ctc_loss(torch::log_softmax(torch::randn({10, 2, 8}), 2),
                          torch::tensor({{1, 2, 3}, {4, 5, 0}}),
                          torch::tensor({10, 10}), torch::tensor({3, 2})));
  PrintScalar("F::poisson_nll_loss",
              F::poisson_nll_loss(torch::randn({4, 4}),
                                  torch::randint(0, 5, {4, 4}).to(torch::kFloat)));

  // ==== Distance functions ====
  PrintSizes("F::cosine_similarity",
             F::cosine_similarity(torch::randn({2, 8}), torch::randn({2, 8}),
                                  F::CosineSimilarityFuncOptions().dim(1)));
  PrintSizes("F::pairwise_distance",
             F::pairwise_distance(torch::randn({2, 8}), torch::randn({2, 8}),
                                  F::PairwiseDistanceFuncOptions().p(2)));
  PrintSizes("F::pdist", F::pdist(torch::randn({5, 4})));

  // ==== Vision functions ====
  PrintSizes("F::interpolate",
             F::interpolate(x4, F::InterpolateFuncOptions()
                                    .size(std::vector<int64_t>({16, 16}))
                                    .mode(torch::kBilinear)
                                    .align_corners(false)));
  auto theta = torch::randn({2, 2, 3});
  auto grid = F::affine_grid(theta, {2, 3, 8, 8}, /*align_corners=*/false);
  PrintSizes("F::affine_grid", grid);
  PrintSizes("F::grid_sample",
             F::grid_sample(x4, grid, F::GridSampleFuncOptions()
                                          .mode(torch::kBilinear)
                                          .padding_mode(torch::kZeros)
                                          .align_corners(false)));
  PrintSizes("F::pad", F::pad(x4, F::PadFuncOptions({1, 1, 1, 1})
                                      .mode(torch::kConstant)
                                      .value(0.0)));
  PrintSizes("F::pixel_shuffle", F::pixel_shuffle(torch::randn({2, 12, 4, 4}), 2));
  PrintSizes("F::pixel_unshuffle", F::pixel_unshuffle(x4, 2));

  // ==== Fold / Unfold ====
  // Unfold extracts sliding local blocks: [N, C*kH*kW, L].
  auto unfolded = F::unfold(x4, F::UnfoldFuncOptions({2, 2}));
  PrintSizes("F::unfold", unfolded);
  // Fold combines blocks back into a tensor.
  auto folded =
      F::fold(unfolded, F::FoldFuncOptions(/*output_size=*/{8, 8},
                                           /*kernel_size=*/{2, 2}));
  PrintSizes("F::fold", folded);

  return 0;
}
