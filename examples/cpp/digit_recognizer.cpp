#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(math/random.h)
#include _FAN_PATH(io/file.h)
#include _FAN_PATH(graphics/webp.h)

double sigmoid(double x) {
  return 1.0 / (1 + exp(-x));
}

double derivative_sigmoid(double x) {
  return x * (1 - x);
}

// initialize random values between 0-1
double init_weights() {
  return fan::random::value_f32(0, 1);
}

void shuffle(int* arr, uint32_t size) {
  if (size == 0) {
    return;
  }

  for (uint32_t i = 0; i < size - 1; ++i) {
    uint32_t j = i + rand() / (RAND_MAX / (size - i) + 1);
    // swap?
    int t = arr[j];
    arr[j] = arr[i];
    arr[i] = t;
  }
}

static constexpr int n_inputs = 256*256;
static constexpr int n_hidden_nodes = 2;
static constexpr int n_outputs = 1;
static constexpr double multiplier = 20;

int main() {
  static constexpr uint32_t number_of_epochs = 1000000;
  static constexpr auto num_of_samples = 10;


  // learning faster or slower
  const double learning_rate = 0.5;

  double hidden_layer[n_hidden_nodes];
  double output_layer[n_outputs];

  double hidden_layer_bias[n_hidden_nodes];
  double output_layer_bias[n_outputs];

  std::vector<std::vector<double>> hidden_weights(n_inputs, std::vector<double>(1, n_hidden_nodes));
  double output_weights[n_hidden_nodes][n_outputs];

  std::vector<fan::vec2d> training_inputs;
  std::vector<double> training_outputs;

  std::vector<int> training_set_order;

  fan::webp::image_info_t im;
  fan::webp::load("images/0.webp", &im);

  std::vector<uint8_t> grayscale;

  for (uint32_t i = 0; i < im.size.x * im.size.y; i += 4) {
    grayscale.push_back(
      0.2126 * ((uint8_t*)im.data)[i + 0] + 0.7152 * ((uint8_t*)im.data)[i + 1] + 0.0722 * ((uint8_t*)im.data)[i + 2]
    );
  }

  for (uint32_t i = 0; i < 10; ++i) {
    training_inputs.push_back(
      fan::vec2d(
        i / multiplier,
        i / multiplier
      )
    );
    training_outputs.push_back(training_inputs[i][0] + training_inputs[i][1]);
    training_set_order.push_back(i);
  }

  for (uint32_t i = 0; i < n_inputs; ++i) {
    for (uint32_t j = 0; j < n_hidden_nodes; ++j) {
      hidden_weights[i][j] = init_weights();
    }
  }

  for (uint32_t i = 0; i < n_hidden_nodes; ++i) {
    hidden_layer_bias[i] = init_weights();
    //                  i  ?
    for (uint32_t j = 0; j < n_outputs; ++j) {
      output_weights[i][j] = init_weights();
    }
  }

  for (uint32_t i = 0; i < n_outputs; ++i) {
    output_layer_bias[i] = init_weights();
  }

  std::string str;

  // training

  uint32_t n_training_sets = training_inputs.size();

  for (uint32_t epoch = 0; epoch < number_of_epochs; ++epoch) {
    bool once = false;
    shuffle(training_set_order.data(), n_training_sets);

    for (uint32_t x = 0; x < n_training_sets; ++x) {
      uint32_t i = training_set_order[x];

      for (uint32_t j = 0; j < n_hidden_nodes; ++j) {
        double activation = hidden_layer_bias[j];

        for (uint32_t k = 0; k < n_inputs; ++k) {
          activation += training_inputs[i][k] * hidden_weights[k][j];
        }
        hidden_layer[j] = sigmoid(activation);
      }

      for (uint32_t j = 0; j < n_outputs; ++j) {
        double activation = output_layer_bias[j];

        for (uint32_t k = 0; k < n_hidden_nodes; ++k) {
          activation += hidden_layer[k] * output_weights[k][j];
        }
        output_layer[j] = sigmoid(activation);
      }

      if ((!once && !(epoch % (number_of_epochs / num_of_samples)) || epoch + 1 == number_of_epochs)) {
        str.append(fan::format("input:{} + {}\noutput:{}\nexpected output:{}\n\n",
          training_inputs[i][0] * multiplier, training_inputs[i][1] * multiplier, output_layer[0] * multiplier, training_outputs[i] * multiplier).data());
        once = true;
      }

      // backpropagation

      double delta_output[n_outputs];

      for (uint32_t j = 0; j < n_outputs; ++j) {
        double error = (training_outputs[i] - output_layer[j]);
        delta_output[j] = error * derivative_sigmoid(output_layer[j]);
      }

      double delta_hidden[n_hidden_nodes];
      for (uint32_t j = 0; j < n_hidden_nodes; ++j) {
        double error = 0;
        for (uint32_t k = 0; k < n_outputs; ++k) {
          error += delta_output[k] * output_weights[j][k];
        }
        delta_hidden[j] = error * derivative_sigmoid(hidden_layer[j]);
      }

      // apply change in output weights
      for (uint32_t j = 0; j < n_outputs; ++j) {
        output_layer_bias[j] += delta_output[j] * learning_rate;
        for (uint32_t k = 0; k < n_hidden_nodes; ++k) {
          output_weights[k][j] += hidden_layer[k] * delta_output[j] * learning_rate;
        }
      }

      // apply change in hidden weights
      for (uint32_t j = 0; j < n_hidden_nodes; ++j) {
        hidden_layer_bias[j] += delta_hidden[j] * learning_rate;
        for (uint32_t k = 0; k < n_inputs; ++k) {
          hidden_weights[k][j] += training_inputs[i][k] * delta_hidden[j] * learning_rate;
        }
      }

    }
  }


  //uint32_t i = 0;

  double a = 7;
  double b = 7;

  a /= multiplier;
  b /= multiplier;


  for (uint32_t j = 0; j < n_hidden_nodes; ++j) {
    double activation = hidden_layer_bias[j];

    activation += a * hidden_weights[0][j];
    activation += b * hidden_weights[1][j];

    hidden_layer[j] = sigmoid(activation);
  }

  for (uint32_t j = 0; j < n_outputs; ++j) {
    double activation = output_layer_bias[j];

    for (uint32_t k = 0; k < n_hidden_nodes; ++k) {
      activation += hidden_layer[k] * output_weights[k][j];
    }
    output_layer[j] = sigmoid(activation);
  }

  fan::print(fan::format("input:{}+{}\noutput:{}\nexpected output:{}\n\n",
    a * multiplier, b * multiplier, output_layer[0] * multiplier, a * multiplier + b * multiplier));
  fan::io::file::write("data", str.data(), std::ios_base::binary);
}