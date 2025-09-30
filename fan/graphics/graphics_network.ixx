module;

#include <fan/utility.h>

#include <functional>
#include <string>
#include <string_view>

export module fan.graphics.network;

export import fan.utility;
export import fan.graphics;
export import fan.network;
export import fan.event;

export namespace fan::graphics::network {
  template<typename send_type_t, typename receive_type_t>
  struct game_client_t {
  public:
    fan::network::tcp_t client;
    bool connected = false;
    std::function<send_type_t()> get_data;
    std::function<void(const std::vector<receive_type_t>&, const std::string& payload)> on_receive;
    fan::event::task_t tasks[3]; // connection, send, receive
    f32_t update_fps = 60.f;

  private:
    fan::event::task_t connect_loop(const std::string& ip, uint16_t port) {
      while (true) {
        try {
          if (!connected) {
            co_await client.connect(ip, port);
            connected = true;
          }
          co_await fan::co_sleep(connected ? 100 : 1000);
        }
        catch (const fan::exception_t& e) {
          connected = false;
          break;
        }
      }
    }
    fan::event::task_t send_loop() {
      while (true) {
        if (connected && get_data) {
          try {
            co_await client.write(get_data());
          }
          catch (const fan::exception_t& e) {
            connected = false;
            break;
          }
        }

        co_await fan::co_sleep(1000.0 / update_fps);
      }
    }
    fan::event::task_t receive_loop() {
      while (true) {
        if (!connected) { 
          co_await fan::co_sleep(10); 
          continue; 
        }

        try {
          std::string buffer;
          fan::network::message_t data;

          while ((data = co_await client.read()) && data) {
            buffer += std::string_view(data.buffer);
            if (data.done && on_receive) {
              std::vector<receive_type_t> items(1);
              fan::graphics::shape_deserialize_t it;
              while (it.iterate(fan::json::parse(buffer), &items.back())) {
                items.resize(items.size() + 1);
              }
              on_receive(items, buffer);
              buffer.clear();
              break;
            }
          }
        }
        catch (const fan::exception_t& e) {
          connected = false;
        }
      }
    }

  public:

    game_client_t(const std::string& ip, uint16_t port,
      std::function<send_type_t()> sender,
      std::function<void(const std::vector<receive_type_t>&, const std::string& payload)> receiver)
      : get_data(sender), on_receive(receiver) {

      tasks[0] = connect_loop(ip, port);
      tasks[1] = send_loop();
      tasks[2] = receive_loop();
    }

    bool is_connected() const { return connected; }

    void set_update_fps(f32_t fps) {
      update_fps = fps;
    }
  };
}