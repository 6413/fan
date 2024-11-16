#include <fan/pch.h>

#include <fan/network/sock.h>

struct pile_t {
  bool should_close = false;
  loco_t loco;
  task_t<void> run() {
    co_await start_render_loop();
  }
  task_t<void> start_render_loop() {
    fan::graphics::rectangle_t r{ {
        .position = fan::vec3(400, 400, 0),
        .size = 200,
        .color = fan::colors::blue
    } };
    f32_t angle = 0;
    while (should_close == false) {

      r.set_angle(fan::vec3(0, 0, angle));
      angle += loco.delta_time;
      render_frame();
      co_await resume_other_tasks();
    }
    should_close = true;
  }

  void render_frame() {
    if (loco.process_loop([] {})) {
      should_close = true;
    }
  }
};

struct client_stuff_t {

  fan::network::network_client_t client;

  task_t<void> run() {
    try {
      fan::print("Attempting to connect to the server...");
      if (!co_await client.connect("127.0.0.1", 12346)) {
        fan::throw_error("Failed to connect to server");
        __unreachable();
      }

      while (true) {
        auto sleep_time = fan::random::value_i64(300, 2000);
        fan::print("Sending 'ping' to the server...");
        auto response = co_await client.echo("ping");

        if (response.empty()) {
          fan::throw_error("Failed to receive response from server");
          __unreachable();
        }

        fan::print("render thread id:",
                   std::hash<std::thread::id>{}(std::this_thread::get_id()),
                   "received:", response, "sleeping for", sleep_time, "ms");

        co_await co_sleep_for(std::chrono::milliseconds(sleep_time));
      }
    } catch (const std::exception& e) {
      fan::print("Exception caught in client_stuff_t::run: ", e.what());
    } catch (...) {
      fan::print("Unknown exception caught in client_stuff_t::run");
    }
  }
};

int main() {
  pile_t pile;
  auto render_task = pile.run();
  client_stuff_t pinger;
  auto ping_task = pinger.run();

  while (!render_task.handle.done()) {
    get_event_loop().run_one();
  }

  return 0;
}