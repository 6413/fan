#include <fan/utility.h>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>

import fan;
#include <fan/graphics/types.h>

using namespace fan::graphics;

struct neural_network_t {
  static constexpr int input_size = 5;
  static constexpr int hidden_size = 20;
  static constexpr int output_size = 4;

  neural_network_t() {
    for (int i = 0; i < input_size; ++i) {
      for (int h = 0; h < hidden_size; ++h) {
        weights_input_hidden[i][h] = fan::random::value(-1.0f, 1.0f);
      }
    }
    for (int h = 0; h < hidden_size; ++h) {
      for (int o = 0; o < output_size; ++o) {
        weights_hidden_output[h][o] = fan::random::value(-1.0f, 1.0f);
      }
      hidden_biases[h] = fan::random::value(-1.0f, 1.0f);
    }
    for (int o = 0; o < output_size; ++o) {
      output_biases[o] = fan::random::value(-1.0f, 1.0f);
    }
  }

  void forward() {
    for (int h = 0; h < hidden_size; ++h) {
      f32_t sum = hidden_biases[h];
      for (int i = 0; i < input_size; ++i) {
        sum += inputs[i] * weights_input_hidden[i][h];
      }
      hidden_outputs[h] = fan::math::tanh_activation(sum);
    }

    for (int o = 0; o < output_size; ++o) {
      f32_t sum = output_biases[o];
      for (int h = 0; h < hidden_size; ++h) {
        sum += hidden_outputs[h] * weights_hidden_output[h][o];
      }
      outputs[o] = fan::math::tanh_activation(sum);
    }
  }

  void backward_and_update() {
    f32_t output_errors[output_size];
    f32_t output_gradients[output_size];

    total_error = 0.0f;
    for (int o = 0; o < output_size; ++o) {
      output_errors[o] = targets[o] - outputs[o];
      total_error += output_errors[o] * output_errors[o];
      output_gradients[o] = output_errors[o] * fan::math::tanh_derivative(outputs[o]);
    }
    total_error *= 0.5f;

    f32_t hidden_errors[hidden_size];
    f32_t hidden_gradients[hidden_size];

    for (int h = 0; h < hidden_size; ++h) {
      f32_t error = 0.0f;
      for (int o = 0; o < output_size; ++o) {
        error += output_gradients[o] * weights_hidden_output[h][o];
      }
      hidden_errors[h] = error;
      hidden_gradients[h] = error * fan::math::tanh_derivative(hidden_outputs[h]);
    }

    for (int h = 0; h < hidden_size; ++h) {
      for (int o = 0; o < output_size; ++o) {
        weights_hidden_output[h][o] += learning_rate * output_gradients[o] * hidden_outputs[h];
      }
    }

    for (int o = 0; o < output_size; ++o) {
      output_biases[o] += learning_rate * output_gradients[o];
    }

    for (int i = 0; i < input_size; ++i) {
      for (int h = 0; h < hidden_size; ++h) {
        weights_input_hidden[i][h] += learning_rate * hidden_gradients[h] * inputs[i];
      }
    }

    for (int h = 0; h < hidden_size; ++h) {
      hidden_biases[h] += learning_rate * hidden_gradients[h];
    }

    epoch++;
  }

  void train_step() { 
    forward(); 
    backward_and_update();
  }

  f32_t weights_input_hidden[input_size][hidden_size];
  f32_t weights_hidden_output[hidden_size][output_size];
  f32_t hidden_biases[hidden_size];
  f32_t output_biases[output_size];

  f32_t inputs[input_size] = { 0.5f, -0.3f, 0.8f };
  f32_t hidden_outputs[hidden_size];
  f32_t outputs[output_size];
  f32_t targets[output_size] = { 0.3f, -0.7f };

  f32_t learning_rate = 0.1f;
  int epoch = 0;
  f32_t total_error = 0.0f;
};

struct neural_network_visualizer_t {
  static constexpr f32_t input_radius = 40.0f;
  static constexpr f32_t hidden_radius = 35.0f;
  static constexpr f32_t output_radius = 45.0f;
  static constexpr f32_t label_offset = 20.0f;

  neural_network_visualizer_t() { setup_positions(); create_visual_elements(); }

  void setup_positions() {
    fan::vec2 window_size = engine.window.get_size();
    fan::vec2 center(window_size.x * 0.5f, window_size.y * 0.5f);
    
    f32_t input_spacing = 85.0f * scale;
    f32_t hidden_spacing = 75.0f * scale;  
    f32_t output_spacing = 110.0f * scale;
    f32_t layer_distance = window_size.x / 1.2f * scale;
    
    f32_t input_start_y = center.y - (neural_network_t::input_size - 1) * input_spacing / 2.0f;
    for (int i = 0; i < neural_network_t::input_size; ++i) {
      input_positions[i] = fan::vec2(center.x - layer_distance, input_start_y + i * input_spacing);
    }
    
    f32_t hidden_start_y = center.y - (neural_network_t::hidden_size - 1) * hidden_spacing / 2.0f;
    for (int i = 0; i < neural_network_t::hidden_size; ++i) {
      hidden_positions[i] = fan::vec2(center.x, hidden_start_y + i * hidden_spacing);
    }
    
    f32_t output_start_y = center.y - (neural_network_t::output_size - 1) * output_spacing / 2.0f;
    for (int i = 0; i < neural_network_t::output_size; ++i) {
      output_positions[i] = fan::vec2(center.x + layer_distance, output_start_y + i * output_spacing);
    }
  }

  void create_visual_elements() {
    // Clear existing elements
    input_nodes.clear();
    hidden_nodes.clear();
    output_nodes.clear();
    input_to_hidden_connections.clear();
    hidden_to_output_connections.clear();

    for (int i = 0; i < neural_network_t::input_size; ++i) {
      input_nodes.push_back(circle_t{ {
        .position = fan::vec3(input_positions[i], 0),
        .radius = input_radius * scale,
        .color = fan::colors::red
      } });
    }

    for (int i = 0; i < neural_network_t::hidden_size; ++i) {
      hidden_nodes.push_back(circle_t{ {
          .position = fan::vec3(hidden_positions[i], 0),
          .radius = hidden_radius * scale,
          .color = fan::colors::blue
      } });
    }

    for (int i = 0; i < neural_network_t::output_size; ++i) {
      output_nodes.push_back(circle_t{ {
        .position = fan::vec3(output_positions[i], 0),
        .radius = output_radius * scale,
        .color = fan::colors::green
      } });
    }

    for (int i = 0; i < neural_network_t::input_size; ++i) {
      for (int h = 0; h < neural_network_t::hidden_size; ++h) {
        input_to_hidden_connections.push_back(line_t{ {
          .src = fan::vec3(input_positions[i], 0),
          .dst = fan::vec3(hidden_positions[h], 0),
          .color = fan::colors::white,
          .thickness = engine_t::line_t::properties_t().thickness * scale
        } });
      }
    }

    for (int h = 0; h < neural_network_t::hidden_size; ++h) {
      for (int o = 0; o < neural_network_t::output_size; ++o) {
        hidden_to_output_connections.push_back(line_t{ {
          .src = fan::vec3(hidden_positions[h], 0),
          .dst = fan::vec3(output_positions[o], 0),
          .color = fan::colors::white,
          .thickness = engine_t::line_t::properties_t().thickness * scale
        } });
      }
    }
  }

  void set_scale(f32_t new_scale) {
    scale = new_scale;
    setup_positions();
    create_visual_elements();
  }

  fan::color calculate_connection_color(f32_t weight, f32_t activity = 0.0f, f32_t highlight = 0.0f) {
    f32_t abs_weight = std::abs(weight);
    f32_t normalized_weight = std::tanh(abs_weight);
    
    const f32_t min_brightness = 0.4f;
    const f32_t max_brightness = 1.0f;
    f32_t brightness = min_brightness + normalized_weight * (max_brightness - min_brightness);
    
    fan::color conn_color;
    if (weight > 0) {
      conn_color = fan::color(brightness * 0.3f, brightness, brightness * 0.2f, 1.0f);
    } 
    else {
      conn_color = fan::color(brightness, brightness * 0.3f, brightness * 0.1f, 1.0f);
    }
    
    if (auto_train && activity > 0) {
      fan::color activity_highlight = fan::colors::white * (highlight * activity * 0.4f);
      conn_color = conn_color + activity_highlight;
      conn_color.a = std::clamp(0.6f + activity * 0.4f, 0.6f, 1.0f);
    } 
    else {
      conn_color.a = std::clamp(0.5f + normalized_weight * 0.5f, 0.5f, 1.0f);
    }
    
    return conn_color;
  }

  fan::color calculate_node_color(f32_t activation, fan::color base_color, f32_t highlight = 0.0f) {
    fan::color node_color;
    if (activation > 0) {
      node_color = fan::color(0, abs(activation), 0, 1);
    } 
    else {
      node_color = fan::color(abs(activation), 0, 0, 1);
    }

    if (auto_train) {
      f32_t pulse = highlight * 0.3f;
      node_color = node_color * (0.7f + pulse) + base_color * 0.3f;
    } 
    else {
      node_color = node_color * 0.7f + base_color * 0.3f;
    }

    return node_color;
  }

  void update_visualization() {
    static f32_t highlight_timer = 0.0f;
    highlight_timer += engine.delta_time * 3.0f;
    f32_t highlight_intensity = (sin(highlight_timer) + 1.0f) * 0.5f;

    int conn_idx = 0;
    for (int i = 0; i < neural_network_t::input_size; ++i) {
      for (int h = 0; h < neural_network_t::hidden_size; ++h) {
        f32_t weight = network.weights_input_hidden[i][h];
        f32_t activity = std::abs(network.inputs[i] * weight);
        activity = std::clamp(activity * 2.0f, 0.0f, 1.0f);
        
        fan::color conn_color = calculate_connection_color(weight, activity, highlight_intensity);
        input_to_hidden_connections[conn_idx].set_color(conn_color);
        conn_idx++;
      }
    }

    conn_idx = 0;
    for (int h = 0; h < neural_network_t::hidden_size; ++h) {
      for (int o = 0; o < neural_network_t::output_size; ++o) {
        f32_t weight = network.weights_hidden_output[h][o];
        f32_t activity = std::abs(network.hidden_outputs[h] * weight);
        activity = std::clamp(activity * 2.0f, 0.0f, 1.0f);
        
        fan::color conn_color = calculate_connection_color(weight, activity, highlight_intensity);
        hidden_to_output_connections[conn_idx].set_color(conn_color);
        conn_idx++;
      }
    }

    for (int i = 0; i < neural_network_t::hidden_size; ++i) {
      fan::color node_color = calculate_node_color(network.hidden_outputs[i], fan::colors::blue, highlight_intensity);
      hidden_nodes[i].set_color(node_color);
    }

    for (int i = 0; i < neural_network_t::output_size; ++i) {
      fan::color node_color = calculate_node_color(network.outputs[i], fan::colors::green, highlight_intensity);
      output_nodes[i].set_color(node_color);
    }
  }

  void render_gui() {
    if (gui::begin("Neural Network Training")) {
      if (gui::button(auto_train ? "Stop Training" : "Start Training")) {
        auto_train = !auto_train;
      }

      gui::same_line();
      if (gui::button("Single Step")) {
        network.train_step();
      }

      gui::same_line();
      if (gui::button("Reset")) {
        network = neural_network_t();
      }

      gui::slider_float("Speed", &train_interval, 0.01f, 1.0f);
      gui::slider_float("Learning Rate", &network.learning_rate, 0.001f, 1.0f);

      gui::separator();

      gui::text("Epoch: " + std::to_string(network.epoch));
      gui::text("Error: " + std::to_string(network.total_error));
    }
    gui::end();

    render_node_values();
  }

  void render_node_values() {
    fan::vec2 window_size = engine.window.get_size();
    gui::set_next_window_pos(fan::vec2(0, 0));
    gui::set_next_window_size(window_size);
    gui::set_next_window_bg_alpha(0.0f);
    
    f32_t dynamic_label_offset = window_size.y * 0.01f;
    
    gui::push_font(gui::get_font(gui::get_font_size() * scale));

    gui::set_cursor_pos(input_positions[0] + fan::vec2(0, -(input_radius * scale + label_offset * scale + dynamic_label_offset)));
    gui::text_centered("Input", fan::colors::red);

    for (int i = 0; i < neural_network_t::input_size; ++i) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << network.inputs[i];
      fan::vec2 text_size = gui::calc_text_size(ss.str());
      fan::vec2 text_pos = input_positions[i] - text_size / 2.0f;
      gui::set_cursor_pos(text_pos);
      gui::text(ss.str(), fan::colors::white);
    }

    gui::set_cursor_pos(hidden_positions[0] + fan::vec2(0, -(hidden_radius * scale + label_offset * scale + dynamic_label_offset)));
    gui::text_centered("Hidden", fan::colors::aqua);

    for (int i = 0; i < neural_network_t::hidden_size; ++i) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << network.hidden_outputs[i];
      fan::vec2 text_size = gui::calc_text_size(ss.str());
      fan::vec2 text_pos = hidden_positions[i] - text_size / 2.0f;
      gui::set_cursor_pos(text_pos);
      gui::text(ss.str(), fan::colors::white);
    }

    gui::set_cursor_pos(output_positions[0] + fan::vec2(0, -(output_radius * scale + label_offset * scale + dynamic_label_offset)));
    gui::text_centered("Output", fan::colors::green);

    for (int i = 0; i < neural_network_t::output_size; ++i) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << network.outputs[i];
      fan::vec2 text_size = gui::calc_text_size(ss.str());
      fan::vec2 text_pos = output_positions[i] - text_size / 2.0f;
      gui::set_cursor_pos(text_pos);
      gui::text(ss.str(), fan::colors::white);

      f32_t target_offset = window_size.x * 0.06f;
      gui::set_cursor_pos(output_positions[i] + fan::vec2(target_offset, 0));
      std::stringstream target_ss;
      target_ss << std::fixed << std::setprecision(2) << network.targets[i];
      gui::text_centered(target_ss.str(), fan::colors::yellow);
    }

    gui::pop_font();
  }

  void update() {
    if (auto_train) {
      train_timer += engine.delta_time;
      if (train_timer >= train_interval) {
        network.train_step();
        train_timer = 0.0f;
      }
    }

    update_visualization();
    render_gui();
  }

  void run() {
    network.forward();
    fan_window_loop{ update(); };
  }

  engine_t engine{ {.renderer = engine_t::renderer_t::opengl} };
  neural_network_t network;

  std::vector<circle_t> input_nodes;
  std::vector<circle_t> hidden_nodes;
  std::vector<circle_t> output_nodes;
  std::vector<line_t> input_to_hidden_connections;
  std::vector<line_t> hidden_to_output_connections;

  fan::vec2 input_positions[neural_network_t::input_size];
  fan::vec2 hidden_positions[neural_network_t::hidden_size];
  fan::vec2 output_positions[neural_network_t::output_size];

  bool auto_train = false;
  f32_t train_timer = 0.0f;
  f32_t train_interval = 0.1f;

  f32_t scale = 1.0f;
};

int main() {
  neural_network_visualizer_t visualizer;
  visualizer.set_scale(0.5f);
  visualizer.run();
  return 0;
}