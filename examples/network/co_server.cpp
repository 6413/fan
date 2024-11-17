#include <fan/types/print.h>
#include <fan/network/sock.h>

int main() {
  fan::network::server_t server;
  auto server_task = server.run("127.0.0.1", 12345);
  get_event_loop().schedule([&] { 
    server_task.handle.resume(); 
  });

  server.on_connect = [&](fan::network::client_t& socket) {
    fan::print("client connected");
    socket.loop = [&] {
      int msg_size = 0;
      fan::print("starting client loop");
      {
        auto v = fan::network::read_data(socket.sock, reinterpret_cast<char*>(&msg_size), sizeof(msg_size));
        v.handle.resume();
      }

      {
        socket.buffer.resize(msg_size);
        auto v = fan::network::read_data(socket.sock, socket.buffer.data(), socket.buffer.size());
        v.handle.resume();
      }

      std::string received(socket.buffer.data(), msg_size);
      std::string response;

      if (received == "ping") {
        response = "pong";
        fan::print("received ping, sending", response);
      }
      else {
        response = "unknown command";
        fan::print("Received unknown command:", received);
      }

      int response_size = response.size();
      {
        auto task = fan::network::write_data(socket.sock, reinterpret_cast<const char*>(&response_size), sizeof(response_size));
        task.handle.resume();
      }

      {
        auto task = fan::network::write_data(socket.sock, response.c_str(), response.size());
        task.handle.resume();
      }
    };
  };

  while (!server_task.handle.done()) {
    get_event_loop().run_one();
  }

  return 0;
}