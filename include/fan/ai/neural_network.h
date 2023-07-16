#pragma once

#include _FAN_PATH(types/types.h)
#include _FAN_PATH(types/matrix.h)

struct neural_network_t {
  uint32_t layer_count;
  std::vector<int> sizes_of_layer;
  // weight matrices, bias matrices
  std::vector<fan::runtime_matrix2d<f32_t>> w, b, delta_w, delta_b;

  f64_t learning_rate;

  neural_network_t() {

  }

  // alpha learning rate
  neural_network_t(std::vector<int> sizes_of_layers, f64_t alpha) {
    layer_count = sizes_of_layers.size();
    sizes_of_layer = sizes_of_layers;

    // weight matrices
    w.resize(layer_count - 1);
    // bias matrices
    b.resize(layer_count - 1);

    delta_w.resize(layer_count - 1);
    delta_b.resize(layer_count - 1);

    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      w[i] = fan::runtime_matrix2d<f32_t>(sizes_of_layers[i], sizes_of_layers[i + 1]);
      b[i] = fan::runtime_matrix2d<f32_t>(1, sizes_of_layers[i + 1]);

      delta_w[i] = fan::runtime_matrix2d<f32_t>(sizes_of_layers[i], sizes_of_layers[i + 1]);
      delta_b[i] = fan::runtime_matrix2d<f32_t>(1, sizes_of_layers[i + 1]);

      w[i].randomize();
      b[i].randomize();
    }

    learning_rate = alpha;
  }

  fan::runtime_matrix2d<f32_t> feed_forward(fan::runtime_matrix2d<f32_t> input) {
    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      input = (input * w[i] + b[i]).sigmoid();
    }
    // output
    return input;
  }

  void backpropagation(fan::runtime_matrix2d<f32_t> input, fan::runtime_matrix2d<f32_t> output) {
    std::vector<fan::runtime_matrix2d<f32_t>> layers;

    layers.push_back(input);
    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      input = (input * w[i] + b[i]).sigmoid();
      layers.push_back(input);
    }

    fan::runtime_matrix2d<f32_t> delta = input - output;
    delta = delta.hadamard(layers[layer_count - 1].sigmoid_derivative());

    delta_b[layer_count - 2] += delta;
    delta_w[layer_count - 2] += layers[layer_count - 2].transpose() * delta;

    for (int i = layer_count - 3; i >= 0; i--) {
      delta = delta * w[i + 1].transpose();

      delta = delta.hadamard(layers[i + 1].sigmoid_derivative());

      delta_b[i] += delta;
      delta_w[i] += layers[i].transpose() * delta;
    }
  }

  void train(std::vector<fan::runtime_matrix2d<f32_t>> inputs, std::vector<fan::runtime_matrix2d<f32_t>> outputs) {
    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      delta_w[i].zero();
      delta_b[i].zero();
    }

    for (uint32_t i = 0; i < inputs.size(); ++i) {
      backpropagation(inputs[i], outputs[i]);
    }

    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      // ehk vääri
      for (uint32_t j = 0; j < delta_w[i].rows; ++j) {
        for (uint32_t z = 0; z < delta_w[i].columns; ++z) {
          delta_w[i][j][z] /= (double)inputs.size();
          w[i][j][z] -= learning_rate * delta_w[i][j][z];
        }
      }

      for (uint32_t j = 0; j < delta_b[i].rows; ++i) {
        for (uint32_t z = 0; z < delta_b[i].columns; ++z) {
          delta_b[i][j][z] /= (double)inputs.size();
          b[i][j][z] -= learning_rate * delta_b[i][j][z];
        }
      }
    }
  }
};