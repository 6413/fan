#include <coroutine>
import std;
import fan;

fan::event::task_t server() {
  co_await fan::network::tcp_server_listen({.ip = "127.0.0.1", .port=8080},
      [](auto& client) -> fan::event::task_t {
      std::string body = fan::io::file::read("xmake.lua");
      std::vector<std::string> paths;
      fan::io::iterate_files_recursive("examples/", [&](const auto& path, const auto& rel) {
        if (!path.string().contains(".cpp") || !path.string().contains("h")) return; 
        paths.emplace_back(path);
      });
      if (paths.size())
      body = fan::io::file::read(paths[fan::random::value(0, int(paths.size()-1))]);
      co_await client.write_raw(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "\r\n" +
        body
      );

      });
}

int main() {
 auto s = server();
 fan::event::loop();
}
