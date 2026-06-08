// client sends rectangle info to server and server draws the rectangle it received
#include <string>
#include <ranges>
#include <string_view>
#include <coroutine>

import fan;

fan::graphics::engine_t engine;
fan::graphics::shape_t server_shape, client_shape = fan::graphics::rectangle_t{ {
    .position = fan::vec3(400, 400, 0),
    .size = 50,////
    .color = fan::colors::red
} };
fan::graphics::render_view_t server_render_view;

fan::event::task_t tcp_server_test() {
  using namespace fan;
  co_await network::tcp_server_listen({ .port = 7777 }, [](const fan::network::tcp_t& client) -> event::task_t {
    std::string json_data;
    network::message_t data;
    while (data = co_await client.read()) {
      json_data += std::string_view(data.buffer);
      if (!data.done || json_data.empty()) {
        continue;
      }
      try {
        server_shape = json_data;
        server_shape.set_render_view(server_render_view);
      }
      catch (const std::exception& e) {
        fan::print_warning("Failed to deserialize rectangle: " + std::string(e.what()));
      }
      json_data.clear();
    }
  });
}

fan::event::task_t tcp_client_test() {
  using namespace fan;

  while (1) {
    try {
      fan::network::tcp_t client;
      co_await client.connect("127.0.0.1", 7777);

      ssize_t error = network::error_code::ok;
      while (error == network::error_code::ok) {
        error = co_await client.write(client_shape);
        static constexpr f32_t fps = 144.f;
        co_await fan::co_sleep(1000.0 / fps);
      }
    }
    catch (std::exception& e) {
      fan::print_warning(std::string("Client error:") + e.what());
    }
    co_await fan::co_sleep(1000); // retry connection every 1s
  }
}

void split_screen(fan::graphics::line_t& splitter) {
  auto size = engine.window.get_size();
  f32_t mid_x = size.x / 2;

  splitter.set_line({mid_x, 0}, {mid_x, size.y});

  server_render_view.set({0, mid_x}, {0, size.y}, {mid_x, 0}, {mid_x, size.y}, fan::window::get_size());
  engine.orthographic_render_view.set({0, mid_x}, {0, size.y}, {0, 0}, {mid_x, size.y}, fan::window::get_size());
  fan::graphics::gui::text("Local view");
  fan::graphics::gui::text_at("Peer view", fan::vec2(10.f + mid_x, 0));
}

void move_shape_based_on_input() {
  auto pos = client_shape.get_position();
  f32_t speed = 500 * engine.delta_time;
  pos += engine.get_input_vector() * speed;
  client_shape.set_position(pos);
}

int main() {
  server_render_view.create();
  try {
    auto tcp_server = tcp_server_test();
    auto tcp_client = tcp_client_test();

    fan::graphics::line_t screen_splitter;

    engine.loop([&] {
      split_screen(screen_splitter);
      move_shape_based_on_input();
    });
  }
  catch (const std::exception& e) {
    fan::print(std::string("Exception:") + e.what());
  }
  server_render_view.remove();
  return 0;
}