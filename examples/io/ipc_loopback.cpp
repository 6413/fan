#include <coroutine>
#include <string_view>
#include <string>

import fan;

fan::event::task_t run() {
  auto path = fan::process::ipc_default_path("fan.ipc");

  fan::process::ipc_server_t server;
  server.listen(path, [](std::string_view msg) {
    fan::print("server got:", msg);
  });

  co_await fan::co_sleep(10);

  fan::process::ipc_client_t client;
  client.connect(path, [](std::string_view msg) {
    fan::print("client got:", msg);
  });

  while (!client.is_connected()) {
    co_await fan::co_sleep(1);
  }

  client.send("hello from client");
  co_await fan::co_sleep(10);
  server.send("hello from server");
  co_await fan::co_sleep(10);
  fan::event::loop_stop();
}

int main() {
  fan::event::task_t task = run();
  fan::event::loop();
}