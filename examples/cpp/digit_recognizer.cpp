#include <fan/pch.h>

#include _FAN_PATH(ai/neural_network.h)

#include _FAN_PATH(math/random.h)
#include _FAN_PATH(io/file.h)
#include _FAN_PATH(graphics/webp.h)

static constexpr uint32_t batch_size = 20;

std::vector<fan::runtime_matrix2d<f64_t>> train_input, train_output;

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

  fan::runtime_matrix2d<f64_t> input(1, 784), output(1, 10);
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

  std::vector<fan::runtime_matrix2d<f64_t>>inputs, outputs;

  fan::runtime_matrix2d<f64_t> current_output;

  for (uint32_t i = 0; i < 42000; ++i) {
    index.push_back(i);
  }

  for (uint32_t epoch = 1; epoch <= 1; ++epoch) {
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
  net.write_to_file("train_data/brains");
}

void test() {
  std::ifstream file("train_data/test.csv");
  std::ofstream out("train_data/ans.csv");
  fan::string trash;

  fan::runtime_matrix2d<f64_t> current_input(1, 784), current_output;

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
  out.close();
}

int main() {
  //
  //test();
  //net.
  //parse_training_data();
  //////
  //train();

  //test();

  #if 1
  fan::string test_data = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,100,213,254,245,255,149,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,181,233,102,40,29,102,166,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,57,236,181,35,0,0,0,0,12,207,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,228,187,0,0,0,0,0,0,96,225,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,230,18,0,0,0,0,0,74,242,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,247,60,0,0,0,0,0,67,232,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,211,0,0,0,0,16,127,225,165,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,217,0,15,58,140,189,181,227,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,246,225,235,253,182,61,231,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,143,119,58,1,153,212,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,88,254,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40,244,157,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,212,211,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,237,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,243,156,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,213,213,5,0,0,0,0,0,6,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,163,244,35,0,0,0,0,0,0,139,208,97,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,248,90,0,0,0,0,0,0,0,16,136,172,168,0,0,0,0,0,0,0,0,0,0,0,0,0,5,195,147,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,237,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";


  net.read_from_file("train_data/brains");

 //// test();
  fan::runtime_matrix2d<f64_t> current_input(1, 784);
  //fan::string test_data;
  //fan::io::file::read("train_data/output.raw", &test_data);
  //test_data.insert(test_data.begin(), 0);

  auto v = split(test_data);

  for (uint32_t j = 0; j < 784; j++) {
    current_input[0][j] = (double)(uint8_t)v[j] / 255.0;
    fan::print(current_input[0][j]);
  }

  double max_value = -1;
  auto current_output = net.feed_forward(current_input);

  int index = 0;
  for (uint32_t j = 0; j < 10; ++j) {
    if (current_output[0][j] > max_value) {
      max_value = current_output[0][j];
      index = j;
    }
  }


  fan::print("result", index);
  #elif 0
  net.read_from_file("train_data/brains");

  test();
  fan::runtime_matrix2d<f64_t> current_input(1, 784);
  fan::string test_data;
  fan::io::file::read("train_data/output.raw", &test_data);

  for (uint32_t j = 0; j < 784; j++) {
    current_input[0][j] = (double)(uint8_t)test_data[j] / 255.0;
    fan::print(current_input[0][j]);
  }

  double max_value = -1;
  auto current_output = net.feed_forward(current_input);

  int index = 0;
  for (uint32_t j = 0; j < 10; ++j) {
    if (current_output[0][j] > max_value) {
      max_value = current_output[0][j];
      index = j;
    }
  }


  fan::print(index);

  #endif
}