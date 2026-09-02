// Transformer layers: Transformer, TransformerEncoder/Decoder,
// TransformerEncoderLayer/DecoderLayer and MultiheadAttention.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  // Default layout is [seq_len, batch, d_model].
  const int64_t src_len = 7, tgt_len = 5, batch = 2, d_model = 16, nhead = 4;

  // --- TransformerEncoderLayer: single self-attention block ---
  auto encoder_layer = torch::nn::TransformerEncoderLayer(
      torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
          .dim_feedforward(32)
          .dropout(0.1));
  auto src = torch::randn({src_len, batch, d_model});
  PrintSizes("TransformerEncoderLayer output", encoder_layer->forward(src));

  // --- TransformerEncoder: stack of encoder layers ---
  auto encoder =
      torch::nn::TransformerEncoder(encoder_layer, /*num_layers=*/2);
  auto memory = encoder->forward(src);
  PrintSizes("TransformerEncoder output", memory);

  // --- TransformerDecoderLayer: self-attention + cross-attention block ---
  auto decoder_layer = torch::nn::TransformerDecoderLayer(
      torch::nn::TransformerDecoderLayerOptions(d_model, nhead)
          .dim_feedforward(32)
          .dropout(0.1));
  auto tgt = torch::randn({tgt_len, batch, d_model});
  PrintSizes("TransformerDecoderLayer output",
             decoder_layer->forward(tgt, memory));

  // --- TransformerDecoder: stack of decoder layers ---
  auto decoder =
      torch::nn::TransformerDecoder(decoder_layer, /*num_layers=*/2);
  PrintSizes("TransformerDecoder output", decoder->forward(tgt, memory));

  // --- Transformer: complete encoder-decoder architecture ---
  auto transformer = torch::nn::Transformer(
      torch::nn::TransformerOptions()
          .d_model(d_model)
          .nhead(nhead)
          .num_encoder_layers(2)
          .num_decoder_layers(2)
          .dim_feedforward(32)
          .dropout(0.1));
  PrintSizes("Transformer output", transformer->forward(src, tgt));

  // --- MultiheadAttention: scaled dot-product attention ---
  // forward returns (attn_output, attn_output_weights).
  auto mha = torch::nn::MultiheadAttention(
      torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(0.1));
  auto attn = mha->forward(src, src, src);  // self-attention
  PrintSizes("MultiheadAttention attn_output", std::get<0>(attn));
  PrintSizes("MultiheadAttention attn_output_weights", std::get<1>(attn));

  // Cross-attention: query from target, key/value from encoder output.
  auto cross = mha->forward(tgt, memory, memory);
  PrintSizes("MultiheadAttention cross attn_output", std::get<0>(cross));

  return 0;
}
