#include <coroutine>

import fan;

#define debug_server 1

fan::event::task_t task_server(uint16_t port) {
  co_await fan::network::tcp_server_listen({ .port = port }, [](const fan::network::tcp_t& client) -> fan::event::task_t {
    while (1) {
      std::string json_data;
      fan::network::message_t data;

      while (data = co_await client.read()) {
        std::vector<fan::network::tcp_t*> clients;
        client.broadcast([&clients, &client](fan::network::tcp_t& clientx) {
          if (clientx.nr != client.nr) {
            clients.push_back(&clientx);
          }
        });

        for (auto* client_ptr : clients) {
          try {
            co_await client_ptr->write(data.buffer);
          }
          catch (const fan::exception_t& e) {
            fan::network::get_client_handler().remove_client(client_ptr->nr);
          }
        }
      }
    }
    });
}

int main(int argc, char** argv) {
  uint16_t port = 7777;
  if (argc != 2) {
    fan::print("usage:*.exe port, overriding port with default (7777)");
  }
  else {
    port = std::stoul(argv[1]);
  }
  auto server_task = task_server(port);
  fan::event::loop();
}