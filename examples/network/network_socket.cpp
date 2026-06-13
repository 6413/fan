import std;
import fan;

namespace examples {

  fan::event::task_t local_http_server() {
    std::cout << "server starting\n";

    co_await fan::network::tcp_server_listen({.ip = "127.0.0.1", .port = 7777}, [](fan::network::tcp_t& client) -> fan::event::task_t {
      std::cout << "server accepted\n";

      auto request = co_await client.read_raw();
      (void)request;

      std::cout << "server got request\n";

      std::string body =
        "hello from fan server\n"
        "this tests socket extension callbacks\n";

      co_await client.write_raw(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "\r\n" +
        body
      );

      std::cout << "server sent response\n";
    });
  }

  fan::event::task_t start_client(fan::network::socket_t& socket) {
    co_await fan::co_sleep(250);
    std::cout << "client opening peer\n";
    socket.open_peer("127.0.0.1", 7777);
  }

  struct delay_connect_ext_t : fan::network::extension_impl_t<std::monostate> {
    delay_connect_ext_t() {
      on_connect = [](void*, fan::network::peer_slot_t& peer) {
        std::cout << "delay begin\n";

        auto* peer_ptr = &peer;

        peer.block();

        fan::event::add_awaitable([peer_ptr]() -> fan::event::task_t {
          co_await fan::co_sleep(1000);

          std::cout << "delay done\n";
          std::cout << "resuming connect chain\n";

          peer_ptr->resume_connect();
        }());
      };
    }
  };

  struct uppercase_data_t {
    fan::network::read_cb_list_t::nr_t read_nr;
    fan::bytes_t buffer;
  };

  struct uppercase_ext_t : fan::network::extension_impl_t<uppercase_data_t> {
    uppercase_ext_t() {
      on_connect = [](void* raw, fan::network::peer_slot_t& peer) {
        std::cout << "uppercase connected\n";

        auto& data = *static_cast<uppercase_data_t*>(raw);

        data.read_nr = peer.add_read_cb(raw, [](void* raw, fan::network::peer_slot_t& peer, fan::network::read_data_t rd) -> bool {
          auto& data = *static_cast<uppercase_data_t*>(raw);

          data.buffer.assign(rd.data, rd.data + rd.size);

          for (auto& c : data.buffer) {
            if (c >= 'a' && c <= 'z') {
              c -= 'a' - 'A';
            }
          }

          peer.dispatch_read(data.read_nr.next(peer.read_cbs), {data.buffer.data(), data.buffer.size()});
          return false;
        });
      };
    }
  };

  struct http_get_ext_t : fan::network::extension_impl_t<std::monostate> {
    http_get_ext_t(std::string host, std::string path)
      : host(std::move(host)), path(std::move(path)) {
      on_connect = [this](void*, fan::network::peer_slot_t& peer) {
        std::cout << "http connected\n";

        peer.add_read_cb(nullptr, [](void*, fan::network::peer_slot_t&, fan::network::read_data_t rd) -> bool {
          std::cout << rd << std::flush;
          fan::event::loop_stop();
          return false;
        });

        peer.write(fan::as_bytes(
          "GET " + this->path + " HTTP/1.1\r\n"
          "Host: " + this->host + "\r\n"
          "Connection: close\r\n"
          "\r\n"
        ));

        std::cout << "http request sent\n";
      };

      on_disconnect = [](void*, fan::network::peer_slot_t&) {
        std::cout << "disconnected\n";
        fan::event::loop_stop();
      };
    }

    std::string host;
    std::string path;
  };

}

int main() {
  static fan::network::socket_t socket;

  socket.add_extension<examples::delay_connect_ext_t>();
  socket.add_extension<examples::uppercase_ext_t>();
  socket.add_extension<examples::http_get_ext_t>("127.0.0.1", "/");

  fan::event::add_awaitable(examples::local_http_server());
  fan::event::add_awaitable(examples::start_client(socket));

  fan::event::loop();
}