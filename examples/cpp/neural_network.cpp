#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(math/random.h)

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

static constexpr int n_inputs = 2;
static constexpr int n_hidden_nodes = 2;
static constexpr int n_outputs = 2;
static constexpr int n_training_sets = 4;

int main() {
  // learning faster or slower
  const double learning_rate = 0.5;

  double hidden_layer[n_hidden_nodes];
  double output_layer[n_outputs];

  double hidden_layer_bias[n_hidden_nodes];
  double output_layer_bias[n_outputs];

  /*
  2
  O->O
   X
  O->O
  */

  double hidden_weights[n_inputs][n_hidden_nodes];
  double output_weights[n_hidden_nodes][n_outputs];

  double training_inputs[n_training_sets][n_inputs] = {
    {10.0, 5.0},
    {7.0, 3.0},
    {5.0, 3.0},
    {9.0, 2.0}
  };

  double training_outputs[n_training_sets][n_outputs] = {
    {15.0},
    {10.0},
    {8.0},
    {11.0}
  };


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

  int training_set_order[] = {
    0, 1, 2, 3
  };

  uint32_t number_of_epochs = 10000;

  fan::string str;

  // training
  for (uint32_t epoch = 0; epoch < number_of_epochs; ++epoch) {
    shuffle(training_set_order, n_training_sets);

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

      str += fan::format("input:{} {}\noutput:{}\nexpected output:{}\n\n", 
        training_inputs[i][0], training_inputs[i][1], output_layer[0], training_outputs[i][0]);

      // backpropagation

      double delta_output[n_outputs];

      for (uint32_t j = 0; j < n_outputs; ++j) {
        double error = (training_outputs[i][j] - output_layer[j]);
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

  fan::io::file::write("data", str, std::ios_base::binary);
}