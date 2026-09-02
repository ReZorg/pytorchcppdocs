// Recurrent layers: RNN, LSTM, GRU (plus cell variants), forward passes,
// hidden state handling and RNNOptions.
#include <torch/torch.h>

#include <iostream>

namespace {
void PrintSizes(const std::string& name, const torch::Tensor& t) {
  std::cout << name << " sizes: " << t.sizes() << std::endl;
}
}  // namespace

int main() {
  const int64_t batch = 4, seq_len = 6, input_size = 8, hidden_size = 16;

  // --- RNN ---
  // RNNOptions(input_size, hidden_size): num_layers, batch_first,
  // bidirectional and dropout are configurable.
  auto rnn = torch::nn::RNN(
      torch::nn::RNNOptions(input_size, hidden_size)
          .num_layers(2)
          .batch_first(true)
          .bidirectional(false));
  auto rnn_input = torch::randn({batch, seq_len, input_size});
  auto rnn_result = rnn->forward(rnn_input);
  PrintSizes("RNN output", std::get<0>(rnn_result));
  PrintSizes("RNN hidden h_n", std::get<1>(rnn_result));

  // An explicit initial hidden state can be passed instead of zeros.
  auto h0 = torch::zeros({2, batch, hidden_size});  // [num_layers, N, H]
  auto rnn_result_h = rnn->forward(rnn_input, h0);
  PrintSizes("RNN output (given h0)", std::get<0>(rnn_result_h));

  // --- LSTM ---
  // Bidirectional doubles the feature dimension of the output. The state is
  // a tuple of hidden state h_n and cell state c_n.
  auto lstm = torch::nn::LSTM(
      torch::nn::LSTMOptions(input_size, hidden_size)
          .num_layers(2)
          .batch_first(true)
          .dropout(0.1)
          .bidirectional(true));
  auto lstm_result = lstm->forward(torch::randn({batch, seq_len, input_size}));
  PrintSizes("LSTM output", std::get<0>(lstm_result));
  PrintSizes("LSTM hidden h_n", std::get<0>(std::get<1>(lstm_result)));
  PrintSizes("LSTM cell c_n", std::get<1>(std::get<1>(lstm_result)));

  // Unidirectional LSTM with explicit (h0, c0) state.
  auto lstm1 = torch::nn::LSTM(
      torch::nn::LSTMOptions(input_size, hidden_size).batch_first(true));
  auto state0 = std::make_tuple(torch::zeros({1, batch, hidden_size}),
                                torch::zeros({1, batch, hidden_size}));
  auto lstm1_result =
      lstm1->forward(torch::randn({batch, seq_len, input_size}), state0);
  PrintSizes("LSTM output (given state)", std::get<0>(lstm1_result));

  // --- GRU ---
  auto gru = torch::nn::GRU(
      torch::nn::GRUOptions(input_size, hidden_size)
          .num_layers(1)
          .batch_first(true));
  auto gru_result = gru->forward(torch::randn({batch, seq_len, input_size}));
  PrintSizes("GRU output", std::get<0>(gru_result));
  PrintSizes("GRU hidden h_n", std::get<1>(gru_result));

  // --- Cell variants: one time step at a time ---
  auto rnn_cell =
      torch::nn::RNNCell(torch::nn::RNNCellOptions(input_size, hidden_size));
  auto lstm_cell =
      torch::nn::LSTMCell(torch::nn::LSTMCellOptions(input_size, hidden_size));
  auto gru_cell =
      torch::nn::GRUCell(torch::nn::GRUCellOptions(input_size, hidden_size));

  auto step_input = torch::randn({batch, input_size});
  auto h = torch::zeros({batch, hidden_size});
  auto c = torch::zeros({batch, hidden_size});
  auto h_gru = torch::zeros({batch, hidden_size});
  for (int64_t t = 0; t < 3; ++t) {
    h = rnn_cell->forward(step_input, h);
    auto lstm_state = lstm_cell->forward(step_input, std::make_tuple(h, c));
    h = std::get<0>(lstm_state);
    c = std::get<1>(lstm_state);
    h_gru = gru_cell->forward(step_input, h_gru);
  }
  PrintSizes("LSTMCell hidden after 3 steps", h);
  PrintSizes("GRUCell hidden after 3 steps", h_gru);

  return 0;
}
