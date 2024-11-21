#include <fan/pch.h>
#include <fan/network/network.h>

fan::ev::task_t tcp_test() {
  try {
    fan::network::getaddrinfo_t info("www.google.com", "80");
    int result = co_await info;
    if (result != 0) {
      fan::print("failed to resolve address");
      co_return;
    }

    fan::network::tcp_t tcp;
    result = co_await tcp.connect(info);
    if (result != 0) {
      fan::print("failed to connect", result);
      co_return;
    }
    std::string httpget = R"(GET / HTTP/1.0
Host: www.google.com
Cache-Control: max-age=0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8)";
    httpget += "\n\n";
    co_await tcp.write(httpget);
    auto reader = tcp.read();
    while (auto data = co_await reader) {
      fan::print(data);
    }
  }
  catch (std::exception& e) {
    fan::print_warning(std::string("tcp_test error:") + e.what());
  }
}

fan::ev::task_t tcp_server_test() {
  try {
    fan::print("TCPSEVER THREAD", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::vector<fan::network::tcp_t> clients;
    fan::network::tcp_t tcp;

    co_await tcp.listen({.port=8080}, [&](auto&& client) -> fan::ev::task_t{
      while (auto msg = co_await client.read()) {
        fan::print_no_endline("server received buffer:", msg);
      }
      clients.push_back(std::move(client));
    });
  }
  catch (std::exception& e) {
    fan::print_warning(std::string("server error:") + e.what());
  }
}

fan::ev::task_t tcp_client_test() {
  try {
    fan::print("TCPCLIENT THREAD", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    fan::network::tcp_t client;
    co_await client.connect("127.0.0.1", 8080);
    std::string msg = "hello world";

    fan::print("sending:" + msg);
    co_await client.write(msg);
    while (auto msg = co_await client.read()) {
      fan::print_no_endline("client received buffer:", msg);
    }
  }
  catch (std::exception& e) {
    fan::print_warning(std::string("client error:") + e.what());
  }
}

int main() {
  try {
    fan::print("MAIN THREAD", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    fan::graphics::engine_t engine;

    auto tcp_task = tcp_test();
   // auto tcp_server = tcp_server_test();
   // auto tcp_client = tcp_client_test();

    engine.loop([&] {
      fan::graphics::text(fan::random::string(10));
    });
  }
  catch (const std::exception& e) {
    fan::print(std::string("exception:") + e.what());
  }

  return 0;
}
