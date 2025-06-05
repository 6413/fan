#include <fan/types/types.h>
#include <fan/types/dme.h>
#include <unordered_map>
#include <exception>
#include <coroutine>
#include <string>
#include <format>
#include <array>
#include <functional>
#include <mutex>
#include <vector>

import fan;
import fan.fmt;
import fan.graphics.video.screen_codec;
using namespace fan::graphics;

struct ecps_backend_t {

#include "prot.h"

  ecps_backend_t() {
    __dme_get(Protocol_S2C, KeepAlive) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      fan::print("tcp keep alive came");
      backend.tcp_keep_alive.reset();
      co_return;
      };
    __dme_get(Protocol_S2C, Response_Login) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Response_Login_t>();
      fan::print_format(R"({{
  [SERVER] Response_login
  SessionID: {}
  AccountID: {}
}})", msg->SessionID.i, msg->AccountID.i);
      };
    __dme_get(Protocol_S2C, CreateChannel_OK) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::CreateChannel_OK_t>();
      fan::print_format(R"({{
[SERVER] CreateChannel_OK
ID: {}
ChannelID: {}
}})", base.ID, msg->ChannelID.i);

      auto it = backend.pending_requests.find(base.ID);
      if (it != backend.pending_requests.end()) {
        it->second.channel_id = msg->ChannelID;
        it->second.completed = true;
        if (it->second.continuation) {
          it->second.continuation.resume();
        }
      }
    };
    __dme_get(Protocol_S2C, JoinChannel_OK) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_OK_t>();
      fan::print_format(R"({{
  [SERVER] JoinChannel_OK
  ID: {}
  ChannelID: {}
}})", base.ID, msg->ChannelID.i);
    };
    __dme_get(Protocol_S2C, JoinChannel_Error) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_Error_t>();
      fan::print_format(R"({{
  [SERVER] JoinChannel_Error
  ID: {}
  ChannelID: {}
}})", base.ID, Protocol::JoinChannel_Error_Reason_String[(uint8_t)msg->Reason]);
    };
  }
  struct channel_create_awaiter {
    ecps_backend_t& backend;
    uint32_t request_id;

    channel_create_awaiter(ecps_backend_t& b, uint32_t id) : backend(b), request_id(id) {}

    bool await_ready() const noexcept {
      auto it = backend.pending_requests.find(request_id);
      return it != backend.pending_requests.end() && it->second.completed;
    }
    void await_suspend(std::coroutine_handle<> h) noexcept {
      auto it = backend.pending_requests.find(request_id);
      if (it != backend.pending_requests.end()) {
        it->second.continuation = h;
      }
    }
    uint16_t await_resume() {
      auto it = backend.pending_requests.find(request_id);
      if (it != backend.pending_requests.end()) {
        uint16_t channel_id = it->second.channel_id;
        backend.pending_requests.erase(it);
        return channel_id;
      }
      throw std::runtime_error("Channel creation request not found");
    }
  };
  fan::event::task_t tcp_read() {
    while (1) {
      auto msg = co_await tcp_client.read<tcp::ProtocolBasePacket_t>();
      fan::print_format(R"({{
  ID: {}
  Command: {}
}})", msg->ID, msg->Command);
      co_await(*Protocol_S2C.NA(msg->Command))(*this, msg.data);
    }
  }
  fan::event::task_value_resume_t<uint32_t> tcp_write(int command, void* data = 0, uint32_t len = 0) {
    static int id = 0;
    tcp::ProtocolBasePacket_t bp;
    bp.ID = id++;
    bp.Command = command;
    co_await tcp_client.write_raw(&bp, sizeof(bp));
    if (data == nullptr) {
      uint8_t random = 0;
      co_await tcp_client.write_raw(&random, sizeof(random));
    }
    else {
      co_await tcp_client.write_raw(data, len);
    }
    co_return bp.ID;
  }

  fan::event::task_t connect(const std::string& ip, uint16_t port) {
    this->ip = ip;
    this->port = port;
    try {
      co_await tcp_client.connect(ip, port);
    }
    catch (...) { co_return; }
    udp_keep_alive.set_server(
      fan::network::buffer_t{
        (char*)&keep_alive_payload,
        (char*)&keep_alive_payload + sizeof(keep_alive_payload)
      },
      { ip, port }
    );
    task_udp_listen = udp_client.listen(
      fan::network::listen_address_t{ ip, port },
      [this, ip, port](const fan::network::udp_t& udp, const fan::network::udp_datagram_t& datagram) -> fan::event::task_t {
        fan::print("udp keep alive came");
        udp::BasePacket_t bp = *(udp::BasePacket_t*)datagram.data.data();
        udp_keep_alive.reset();
        co_return;
      }
    );
    tcp_keep_alive.reset();
    udp_keep_alive.reset();
    task_tcp_read = tcp_read();
  }
  fan::event::task_t login() {
    co_await tcp_write(Protocol_C2S_t::Request_Login);
  }
  fan::event::task_value_resume_t<Protocol_ChannelID_t> channel_create() {
    uint32_t request_id = co_await tcp_write(Protocol_C2S_t::CreateChannel);
    pending_requests[request_id] = pending_request_t{
      .request_id = request_id,
      .continuation = {},
      .channel_id = 0,
      .completed = false
    };
    co_return co_await channel_create_awaiter(*this, request_id);
  }
  // add check if channel join is ok
  fan::event::task_t channel_join(Protocol_ChannelID_t channel_id) {
    co_await tcp_write(Protocol_C2S_t::JoinChannel, &channel_id, sizeof(channel_id));
    channel_info_t ci;
    ci.channel_id = channel_id;
    channel_info.emplace_back(ci);
  }

  fan::event::task_t task_tcp_read;

  std::string ip;
  uint16_t port;
  fan::network::tcp_t tcp_client;
  fan::network::udp_t udp_client;
  fan::event::task_t task_udp_listen;

  tcp::ProtocolBasePacket_t keep_alive_payload{ .ID = 0, .Command = (Protocol_CI_t)Protocol_C2S_t::KeepAlive };
  fan::network::tcp_keep_alive_t tcp_keep_alive{
    tcp_client,
    fan::network::buffer_t{
      (char*)&keep_alive_payload,
      (char*)&keep_alive_payload + sizeof(keep_alive_payload)
    }
  };
  fan::network::udp_keep_alive_t udp_keep_alive{ udp_client };

  struct pending_request_t {
    uint32_t request_id;
    std::coroutine_handle<> continuation;
    Protocol_ChannelID_t channel_id = 0;
    bool completed = false;
  };
  std::unordered_map<uint32_t, pending_request_t> pending_requests;

  struct channel_info_t {
    Protocol_ChannelID_t channel_id = 0;
  };
  std::vector<channel_info_t> channel_info;
}ecps_backend;

struct screen_codec_t : fan::graphics::screen_codec_t {
  std::mutex mutex;
}screen_codec;

struct render_thread_t {
  engine_t engine;
  #define engine OFFSETLESS(This, render_thread_t, ecps_gui)->engine
  #include "gui.h"
  ecps_gui_t ecps_gui;
  std::mutex mutex;

  fan::graphics::universal_image_renderer_t screen_frame{{
    .position = fan::vec3(gloco->window.get_size() / 2, 0),
    .size = gloco->window.get_size() / 2,
  }};

  void render() {
    render_thread->engine.process_loop([&]{ ecps_gui.render(); });
  }
}*render_thread=0;

int main() {
  auto thread_id = fan::event::thread_create([] {
    render_thread = new render_thread_t;
    while (!render_thread->engine.should_close()) {
      render_thread->render();
    }
    delete render_thread;
  });

  auto task_process_backend_queue = []() -> fan::event::task_t {
    while (true) {
      if (render_thread != nullptr) {
        std::vector<std::function<fan::event::task_t()>> local_tasks;
        {
          std::lock_guard<std::mutex> lock(render_thread->mutex);
          local_tasks = std::move(render_thread->ecps_gui.task_queue);
          render_thread->ecps_gui.task_queue.clear();
        }
        for (const auto& f : local_tasks) {
          co_await f();
        }
      }
      co_await fan::co_sleep(1);
    }
  }();

  fan::event::loop();
}