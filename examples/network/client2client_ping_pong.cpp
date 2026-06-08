#include <string>
#include <coroutine>
#include <vector>

import fan.print;
import fan.network;


fan::event::task_t tcp_server_test() {
  using namespace fan;
  co_await network::tcp_listen({ .port = 7777 }, [](const fan::network::tcp_t& client) -> event::task_t {
    std::string json_data;
    network::message_t data;
    fan::print("client connected", client.socket->socket);
    while (data = co_await client.read()) {
      json_data.insert(json_data.end(), data.buffer.begin(), data.buffer.end());
      if (!data.done || json_data.empty()) {
        continue;
      }
      fan::print("server received", client.nr.NRI, json_data);
      client.broadcast([&] (const fan::network::tcp_t& clientx) {
        clientx.write(json_data == "ping" ? "pong" : "ping");
      });
      json_data.clear();
    }
  });
}

int client = 0;
fan::event::task_t tcp_client_test() {
  if (client == 0) {
    client++;
    co_await fan::co_sleep(1400);
  }
  using namespace fan;
  while (1) {
    try {
      int ping = 1;
      fan::network::tcp_t server;
      co_await server.connect("127.0.0.1", 7777);
      fan::print("connected");

      ssize_t error = network::error_code::ok;
      while (error == network::error_code::ok) {
        server.write(ping ? "ping" : "pong");
        ping = (ping + 1) & 1;
        std::string json_data;
        network::message_t data;
        while (data = co_await server.read()) {
          json_data.insert(json_data.end(), data.buffer.begin(), data.buffer.end());
          if (!data.done || json_data.empty()) {
            continue;
          }
          fan::print("client received", server.nr.NRI, json_data);
          break;
        }
        co_await fan::co_sleep(1000);
      }
    }
    catch (std::exception& e) {
      fan::print_warning(std::string("Client error:") + e.what());
    }
    co_await fan::co_sleep(1000); // retry connection every 1s
  }
}


int main() {
  auto task = tcp_server_test();
  auto task2 = tcp_client_test();
  auto task3 = tcp_client_test();
  fan::event::loop();
}