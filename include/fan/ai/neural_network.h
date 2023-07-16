#pragma once

#include _FAN_PATH(types/types.h)
#include _FAN_PATH(types/matrix.h)
#include _FAN_PATH(io/file.h)

struct neural_network_t {
  uint32_t layer_count;
  // weight matrices, bias matrices
  std::vector<fan::runtime_matrix2d<f64_t>> w, b, delta_w, delta_b;

  f64_t learning_rate;

  neural_network_t() {

  }

  // alpha learning rate
  neural_network_t(std::vector<int> sizes_of_layers, f64_t alpha) {
    layer_count = sizes_of_layers.size();

    // weight matrices
    w.resize(layer_count - 1);
    // bias matrices
    b.resize(layer_count - 1);

    delta_w.resize(layer_count - 1);
    delta_b.resize(layer_count - 1);

    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      w[i] = fan::runtime_matrix2d<f64_t>(sizes_of_layers[i], sizes_of_layers[i + 1]);
      b[i] = fan::runtime_matrix2d<f64_t>(1, sizes_of_layers[i + 1]);

      delta_w[i] = fan::runtime_matrix2d<f64_t>(sizes_of_layers[i], sizes_of_layers[i + 1]);
      delta_b[i] = fan::runtime_matrix2d<f64_t>(1, sizes_of_layers[i + 1]);

      w[i].randomize();
      b[i].randomize();
    }

    learning_rate = alpha;
  }

  fan::runtime_matrix2d<f64_t> feed_forward(fan::runtime_matrix2d<f64_t> input) {
    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      input = (input * w[i] + b[i]).sigmoid();
    }
    // output
    return input;
  }

  void backpropagation(fan::runtime_matrix2d<f64_t> input, fan::runtime_matrix2d<f64_t> output) {
    std::vector<fan::runtime_matrix2d<f64_t>> layers;

    layers.push_back(input);
    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      input = (input * w[i] + b[i]).sigmoid();
      layers.push_back(input);
    }

    fan::runtime_matrix2d<f64_t> delta = input - output;
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

  void train(std::vector<fan::runtime_matrix2d<f64_t>> inputs, std::vector<fan::runtime_matrix2d<f64_t>> outputs) {
    for (uint32_t i = 0; i < layer_count - 1; ++i) {
      delta_w[i].zero();
      delta_b[i].zero();
    }

    for (uint32_t i = 0; i < inputs.size(); ++i) {
      backpropagation(inputs[i], outputs[i]);
    }

    for (uint32_t i = 0; i < layer_count - 1; ++i) {
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

  void read_from_file(const fan::string& path) {
    fan::string data;
    fan::io::file::read(path, &data);

    uint64_t offset = 0;
    auto string_to_data = [&offset, &data]<typename T>(T& data_vector) {
      uint64_t data_size;
      data_size = *(uint64_t*)&data[offset];
      offset += sizeof(data_size);
      data_vector.clear();
      data_vector.reserve(data_size);
      for (uint64_t imatrix = 0; imatrix < data_size; ++imatrix) {
        uint32_t rows = *(uint32_t*)&data[offset];
        offset += sizeof(rows);

        uint32_t columns = *(uint32_t*)&data[offset];
        offset += sizeof(columns);

        data_vector.push_back({ rows, columns });
        for (uint32_t i = 0; i < rows; ++i) {
          for (uint32_t j = 0; j < columns; ++j) {
            using matrix_value_type_t = typename T::value_type::value_type;
            data_vector[imatrix][i][j] = *(matrix_value_type_t*)&data[offset];
            offset += sizeof(matrix_value_type_t);
          }
        }
      }
    };
    string_to_data(w);
    string_to_data(b);
  }

  void write_to_file(const fan::string& path) {
    auto data_to_string = [](const auto& data_vector) {
      fan::string out;
      uint64_t data_size = data_vector.size();
      out.append((uint8_t*)&data_size, (uint8_t*)&data_size + sizeof(data_size));
      for (auto& matrix : data_vector) {
        uint32_t rows = matrix.rows;
        uint32_t columns = matrix.columns;
        out.append((uint8_t*)&rows, (uint8_t*)&rows + sizeof(rows));
        out.append((uint8_t*)&columns, (uint8_t*)&columns + sizeof(columns));
        for (uint32_t i = 0; i < matrix.rows; ++i) {
          for (uint32_t j = 0; j < matrix.columns; ++j) {
            out.append((uint8_t*)&matrix.data[i][j], (uint8_t*)&matrix.data[i][j] + sizeof(matrix.data[i][j]));
          }
        }
      }
      return out;
    };

    fan::string out;
    out += data_to_string(w);
    out += data_to_string(b);

    fan::io::file::write(path, out);
  }
};