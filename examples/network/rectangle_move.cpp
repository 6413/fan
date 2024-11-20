#include <fan/pch.h>
#include <fan/network/network.h>

fan::graphics::engine_t engine;
fan::graphics::rectangle_t r{ {
    .position = fan::vec3(400, 400, 0),
    .size = 50,
    .color = fan::colors::red
} };

fan::ev::task_t tcp_server_test() {
  try {
    co_await fan::network::tcp_listen({.port = 8080}, [](auto&& client) -> fan::ev::task_t {
      fan::json_stream_parser_t parser;
      auto reader = client.read();
      while (auto data = co_await reader) {
        auto results = parser.process(data);
        for (const auto& result : results) {
          fan::vec3 p = result.value["position"];
          r.set_position(p);
        }
      }
      fan::print("");
    });
  }
  catch (std::exception& e) {
    fan::print_warning(std::string("server error:") + e.what());
  }
}

fan::ev::task_t tcp_client_test() {
  try {
    fan::network::tcp_t client;
    co_await client.connect("127.0.0.1", 8080);
    fan::json j;
    while (1) {
      j["position"] = r.get_position();
      co_await client.write(j.dump());
      co_await fan::co_sleep(10);
    }
  }
  catch (std::exception& e) {
    fan::print_warning(std::string("client error:") + e.what());
  }
}

int main() {
  try {
    auto tcp_server = tcp_server_test();
    //auto tcp_client = tcp_client_test();

    engine.input_action.add(fan::key_a, "a");
    engine.input_action.add(fan::key_d, "d");
    engine.input_action.add(fan::key_w, "w");
    engine.input_action.add(fan::key_s, "s");

    engine.loop([&] {
      fan::graphics::text(fan::random::string(10));
        auto pos = r.get_position();
        const float speed = 500 * engine.delta_time;
  
        pos += fan::vec3(
          speed * (engine.input_action.is_active("d", 2) - engine.input_action.is_active("a", 2)),
          speed * (engine.input_action.is_active("s", 2) - engine.input_action.is_active("w", 2)),
          0
        );
  
        r.set_position(pos);
    });
  }
  catch (const std::exception& e) {
    fan::print(std::string("exception:") + e.what());
  }

  return 0;
}