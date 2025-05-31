#include <string>
#include <ranges>
#include <fan/time/timer.h>
#include <string_view>

import fan;

fan::graphics::engine_t engine;
fan::graphics::rectangle_t r{ {
    .position = fan::vec3(400, 400, 0),
    .size = 50,
    .color = fan::colors::red
} };

fan::event::task_t tcp_server_test() {
  co_await fan::network::tcp_listen({ .port = 7777 }, [](auto&& client) -> fan::event::task_t {
    std::string json_data;

    auto reader = client.read();
    while (fan::network::message_t* data = co_await reader) {
      json_data.insert(json_data.end(), data->buffer.begin(), data->buffer.end());
      if (!data->done) {
        continue;
      }
      if (json_data.empty()) {
        continue;
      }
      try {
        fan::json result = fan::json::parse(json_data);
        r = result;
        // set other properties
      }
      catch (const std::exception& e) {
        fan::print("\n\n", data, json_data.size(), json_data);
        fan::print_warning("Failed to deserialize rectangle: " + std::string(e.what()));
      }
      json_data.clear();
    }
  });
}

fan::event::task_t tcp_client_test() {
  try {
    fan::network::tcp_t client;
    co_await client.connect("127.0.0.1", 7777);

    fan::vec3 last_sent_position = r.get_position();

    while (1) {
      fan::vec3 current_pos = r.get_position();

      fan::json j = r;
      std::string json_data = j.dump();
      co_await client.write(json_data);
      last_sent_position = current_pos;
      co_await fan::co_sleep(1000.0 / 64.f);
    }
  }
  catch (std::exception& e) {
    fan::print_warning(std::string("Client error:") + e.what());
  }
}

int main() {
  try {
#define SERVER 1
#if SERVER == 1
    auto tcp_server = tcp_server_test();
#else
    auto tcp_client = tcp_client_test();
#endif
    engine.loop([&] {
      fan::graphics::gui::text(fan::random::string(10));

      auto pos = r.get_position();
      const float speed = 500 * engine.delta_time;

      pos += fan::vec3(
        speed * (fan::window::is_key_down(fan::key_d) - fan::window::is_key_down(fan::key_a)),
        speed * (fan::window::is_key_down(fan::key_s) - fan::window::is_key_down(fan::key_w)),
        0
      );

#if SERVER == 0
      r.set_position(pos);
#endif
    });
  }
  catch (const std::exception& e) {
    fan::print(std::string("Exception:") + e.what());
  }
  return 0;
}