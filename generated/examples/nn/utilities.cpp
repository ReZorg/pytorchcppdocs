// Utilities: torch::nn::init (xavier/kaiming/uniform/normal/orthogonal),
// Cloneable, AnyModule, Functional, CosineSimilarity, PairwiseDistance,
// PackedSequence (pack/pad), grad clipping, parameters_to_vector,
// padding layers and vision layers.
#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  // ==== torch::nn::init: parameter initialization ====
  auto linear = torch::nn::Linear(64, 32);
  torch::nn::init::xavier_uniform_(linear->weight);
  torch::nn::init::xavier_normal_(linear->weight);
  torch::nn::init::zeros_(linear->bias);

  auto conv = torch::nn::Conv2d(8, 16, 3);
  torch::nn::init::kaiming_uniform_(conv->weight, /*a=*/0, torch::kFanIn,
                                    torch::kReLU);
  torch::nn::init::kaiming_normal_(conv->weight);

  auto bn = torch::nn::BatchNorm1d(8);
  torch::nn::init::ones_(bn->weight);
  torch::nn::init::constant_(linear->bias, 0.1);
  torch::nn::init::normal_(linear->weight, /*mean=*/0, /*std=*/0.01);
  torch::nn::init::uniform_(linear->weight, /*a=*/-0.1, /*b=*/0.1);

  auto rnn = torch::nn::RNN(8, 16);
  torch::nn::init::orthogonal_(rnn->named_parameters()["weight_hh_l0"]);
  std::cout << "init applied; linear bias[0]: "
            << linear->bias[0].item<float>() << std::endl;

  // ==== Cloneable: deep copy of a module ====
  auto model = torch::nn::Linear(10, 5);
  auto model_copy =
      std::dynamic_pointer_cast<torch::nn::LinearImpl>(model->clone());
  PrintSizes("Cloned Linear output",
             model_copy->forward(torch::randn({2, 10})));

  // ==== AnyModule: type-erased module storage ====
  torch::nn::AnyModule any_module(torch::nn::Linear(10, 5));
  PrintSizes("AnyModule output", any_module.forward(torch::randn({2, 10})));

  // ==== Functional: wrap a callable as a module ====
  torch::nn::Sequential with_fn(
      torch::nn::Linear(8, 8),
      torch::nn::Functional([](torch::Tensor t) { return torch::sigmoid(t); }));
  PrintSizes("Sequential with Functional output",
             with_fn->forward(torch::randn({2, 8})));

  // ==== CosineSimilarity / PairwiseDistance ====
  auto cos = torch::nn::CosineSimilarity(
      torch::nn::CosineSimilarityOptions().dim(1).eps(1e-6));
  auto a = torch::randn({2, 8});
  auto b = torch::randn({2, 8});
  PrintSizes("CosineSimilarity output", cos->forward(a, b));

  auto pdist = torch::nn::PairwiseDistance(
      torch::nn::PairwiseDistanceOptions().p(2).eps(1e-6));
  PrintSizes("PairwiseDistance output", pdist->forward(a, b));

  // ==== PackedSequence: pack_padded_sequence / pad_packed_sequence ====
  // Three sequences of lengths 4, 2, 1 stored batch-first.
  auto seqs = torch::randn({3, 4, 8});
  auto lengths = torch::tensor({4, 2, 1}, torch::kInt64);
  auto packed = torch::nn::utils::rnn::pack_padded_sequence(
      seqs, lengths, /*batch_first=*/true);
  PrintSizes("PackedSequence data", packed.data());
  PrintSizes("PackedSequence batch_sizes", packed.batch_sizes());

  // All RNN modules accept packed sequences as inputs.
  auto pack_rnn = torch::nn::RNN(8, 16);
  auto packed_out = pack_rnn->forward_with_packed_input(packed);
  auto padded = torch::nn::utils::rnn::pad_packed_sequence(
      std::get<0>(packed_out), /*batch_first=*/true);
  PrintSizes("pad_packed_sequence output", std::get<0>(padded));
  PrintSizes("pad_packed_sequence lengths", std::get<1>(padded));

  // ==== Gradient utilities: clip_grad_norm_ / clip_grad_value_ ====
  auto model2 = torch::nn::Linear(8, 8);
  auto loss = model2->forward(torch::randn({4, 8})).sum();
  loss.backward();
  auto params = model2->parameters();
  double total_norm =
      torch::nn::utils::clip_grad_norm_(params, /*max_norm=*/1.0);
  std::cout << "clip_grad_norm_ total norm: " << total_norm << std::endl;
  torch::nn::utils::clip_grad_value_(params, /*clip_value=*/0.5);
  std::cout << "clip_grad_value_ applied" << std::endl;

  // ==== parameters_to_vector / vector_to_parameters ====
  auto flat_params = torch::nn::utils::parameters_to_vector(params);
  PrintSizes("parameters_to_vector", flat_params);
  flat_params.add_(1.0);  // modify the flat copy...
  torch::nn::utils::vector_to_parameters(flat_params, params);
  std::cout << "vector_to_parameters wrote back; weight[0,0]: "
            << model2->weight[0][0].item<float>() << std::endl;

  // ==== Padding layers ====
  PrintSizes("ReflectionPad1d output",
             torch::nn::ReflectionPad1d(1)->forward(torch::randn({2, 3, 8})));
  PrintSizes(
      "ReflectionPad2d output",
      torch::nn::ReflectionPad2d(1)->forward(torch::randn({2, 3, 8, 8})));
  PrintSizes("ReflectionPad3d output",
             torch::nn::ReflectionPad3d(1)->forward(
                 torch::randn({1, 3, 4, 4, 4})));
  PrintSizes("ReplicationPad1d output",
             torch::nn::ReplicationPad1d(1)->forward(torch::randn({2, 3, 8})));
  PrintSizes(
      "ReplicationPad2d output",
      torch::nn::ReplicationPad2d(1)->forward(torch::randn({2, 3, 8, 8})));
  PrintSizes("ReplicationPad3d output",
             torch::nn::ReplicationPad3d(1)->forward(
                 torch::randn({1, 3, 4, 4, 4})));
  PrintSizes("ZeroPad1d output",
             torch::nn::ZeroPad1d(1)->forward(torch::randn({2, 3, 8})));
  PrintSizes("ZeroPad2d output",
             torch::nn::ZeroPad2d(1)->forward(torch::randn({2, 3, 8, 8})));
  PrintSizes("ZeroPad3d output", torch::nn::ZeroPad3d(1)->forward(
                                     torch::randn({1, 3, 4, 4, 4})));
  PrintSizes("ConstantPad1d output",
             torch::nn::ConstantPad1d(
                 torch::nn::ConstantPad1dOptions(1, 0.5))
                 ->forward(torch::randn({2, 3, 8})));
  PrintSizes("ConstantPad2d output",
             torch::nn::ConstantPad2d(
                 torch::nn::ConstantPad2dOptions({1, 1, 1, 1}, 0.5))
                 ->forward(torch::randn({2, 3, 8, 8})));
  PrintSizes("ConstantPad3d output",
             torch::nn::ConstantPad3d(
                 torch::nn::ConstantPad3dOptions(1, 0.5))
                 ->forward(torch::randn({1, 3, 4, 4, 4})));

  // ==== Vision layers ====
  auto pixel_shuffle = torch::nn::PixelShuffle(/*upscale_factor=*/2);
  PrintSizes("PixelShuffle output",
             pixel_shuffle->forward(torch::randn({2, 12, 4, 4})));

  auto pixel_unshuffle = torch::nn::PixelUnshuffle(/*downscale_factor=*/2);
  PrintSizes("PixelUnshuffle output",
             pixel_unshuffle->forward(torch::randn({2, 3, 8, 8})));

  auto upsample = torch::nn::Upsample(
      torch::nn::UpsampleOptions()
          .size(std::vector<int64_t>({16, 16}))
          .mode(torch::kBilinear)
          .align_corners(false));
  PrintSizes("Upsample output",
             upsample->forward(torch::randn({2, 3, 8, 8})));

  // Unfold extracts sliding blocks; Fold combines blocks back.
  auto unfold = torch::nn::Unfold(
      torch::nn::UnfoldOptions(/*kernel_size=*/{2, 2}));
  auto blocks = unfold->forward(torch::randn({2, 3, 8, 8}));
  PrintSizes("Unfold output", blocks);
  auto fold = torch::nn::Fold(
      torch::nn::FoldOptions(/*output_size=*/{8, 8}, /*kernel_size=*/{2, 2}));
  PrintSizes("Fold output", fold->forward(blocks));

  return 0;
}
