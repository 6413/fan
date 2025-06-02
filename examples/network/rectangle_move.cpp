#include <string>
#include <ranges>
#include <fan/time/timer.h>
#include <string_view>

import fan;

#define SERVER 0

fan::graphics::engine_t engine;
fan::graphics::rectangle_t r{ {
    .position = fan::vec3(400, 400, 0),
    .size = 50,////
    .color = fan::colors::red
} };

fan::event::task_t tcp_server_test() {
  using namespace fan;
  co_await network::tcp_listen({ .port = 7777 }, [](const fan::network::tcp_t& client) -> event::task_t {
    std::string json_data;
    network::message_t data;
    while (data = co_await client.read()) {
      json_data += data.buffer;
      if (!data.done || json_data.empty()) {
        continue;
      }
      try {
        r = json_data;
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
      fan::network::tcp_t server;
      co_await server.connect("127.0.0.1", 7777);

      ssize_t error = network::error_code::ok;
      while (error == network::error_code::ok) {
        error = co_await server.write(r);
        co_await fan::co_sleep(1000.0 / 64.f);
      }
    }
    catch (std::exception& e) {
      fan::print_warning(std::string("Client error:") + e.what());
    }
    co_await fan::co_sleep(1000); // retry connection every 1s
  }
}

int main() {
  try {
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