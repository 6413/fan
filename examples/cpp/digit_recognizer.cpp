#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(ai/neural_network.h)

#include _FAN_PATH(math/random.h)
#include _FAN_PATH(io/file.h)
#include _FAN_PATH(graphics/webp.h)

static constexpr uint32_t batch_size = 20;

std::vector<fan::runtime_matrix2d<f32_t>> train_input, train_output;

// 784 neurons, there are 28*28 pixels so 784, learning rate 1.0
neural_network_t net({784, 20, 10}, 1.0);

std::vector<int> split(const fan::string& s) {
  int current = 0;
  std::vector<int> ans;

  for (uint32_t i = 0; i < s.size(); ++i) {
    if (s[i] == ',') {
      ans.push_back(current);
      current = 0;
    }
    else {
      current *= 10;
      current += s[i] - '0';
    }
  }
  ans.push_back(current);
  return ans;
}

void parse_training_data() {
  fan::string trash;
  std::ifstream file("train_data/train.csv");

  if (!file.is_open()) {
    fan::throw_error("invalid file path");
  }

  fan::runtime_matrix2d<f32_t> input(1, 784), output(1, 10);
  std::vector<int> v;

  train_input.reserve(42000);
  train_output.reserve(42000);

  file >> trash;
  for (uint32_t i = 0; i < 42000; ++i) {
    file >> trash;

    v = split(trash);

    output.zero();
    output[0][v[0]] = 1.0;

    for (uint32_t j = 1; j < 785; ++j) {
      input[0][j - 1] = v[j] / 255.0; // normalize
    }

    train_input.push_back(input);
    train_output.push_back(output);
  }
  fan::print("training data loaded");
}

void random_shuffle(std::vector<int>& v) {
  auto rng = std::default_random_engine{};
  std::shuffle(std::begin(v), std::end(v), rng);
}

void train() {

  std::vector<int> index;

  std::vector<fan::runtime_matrix2d<>>inputs, outputs;

  fan::runtime_matrix2d<> current_output;

  for (uint32_t i = 0; i < 42000; ++i) {
    index.push_back(i);
  }

  for (uint32_t epoch = 1; epoch <= 10; ++epoch) {
    fan::print(epoch, "starting");

    double error = 0;

    random_shuffle(index);
    for (uint32_t i = 0; i < 42000; i += batch_size) {
      inputs.clear();
      outputs.clear();

      for (uint32_t j = 0; j < batch_size; ++j) {
        inputs.push_back(train_input[index[i + j]]);
        outputs.push_back(train_output[index[i + j]]);
      }
      net.train(inputs, outputs);
    }

    for (uint32_t i = 0; i < 42000; i++) {
      current_output = net.feed_forward(train_input[i]);

      for (uint32_t j = 0; j < 10; j++) {
        error += (current_output[0][j] - train_output[i][0][j]) * (current_output[0][j] - train_output[i][0][j]);
      }
    }

    error /= 10.0;
    error /= 42000.0;

    fan::print("epoch", epoch, "finished");
    fan::print("error rate:", error);
  }
}

void test() {
  std::ifstream file("train_data/test.csv");
  std::ofstream out("train_data/ans.csv");
  fan::string trash;

  fan::runtime_matrix2d current_input(1, 784), current_output;

  int index;

  out << "ImageId,Label" << std::endl;

  file >> trash;
  std::vector<int> v;
  for (uint32_t i = 0; i < 28000; ++i) {
    file >> trash;
    v = split(trash);

    for (uint32_t j = 0; j < 784; ++j) {
      current_input[0][j] = v[j] / 255.0;
    }

    double max_value = -1;
    current_output = net.feed_forward(current_input);

    for (uint32_t j = 0; j < 10; ++j) {
      if (current_output[0][j] > max_value) {
        max_value = current_output[0][j];
        index = j;
      }
    }

    out << i + 1 << "," << index << std::endl;
  }
}

int main() {
  parse_training_data();
  train();

  test();
}