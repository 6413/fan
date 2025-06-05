#include <fan/types/types.h>
#include <fan/time/timer.h>
#include <fan/event/types.h>
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

#include <cuda.h>

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

struct screen_encode_t : fan::graphics::screen_encode_t {
  std::vector<uint8_t> buffer_copy;
  std::mutex mutex;
  std::atomic<uint64_t> decoder_timestamp{0};
  std::atomic<uint64_t> frame_counter{0};
}screen_encode;

struct render_thread_t {
  engine_t engine;
  #define engine OFFSETLESS(This, render_thread_t, ecps_gui)->engine
  #include "gui.h"
  ecps_gui_t ecps_gui;
  std::mutex mutex;

  // static fan::graphics::image_t image = engine.image_create(gui::get_color(gui::col_window_bg));
  fan::graphics::universal_image_renderer_t screen_frame{{
    .position = fan::vec3(gloco->window.get_size() / 2, 0),
    .size = gloco->window.get_size() / 2,
  }};

  void render() {
    render_thread->engine.process_loop([&]{ ecps_gui.render(); });
  }

  struct screen_decode_t : fan::graphics::screen_decode_t {

  }screen_decode;
}*render_thread=0;

std::atomic<bool> can_encode = true;
uint64_t timestamp = 0;

int main() {
  std::promise<void> render_thread_promise;
  std::future<void> render_thread_future = render_thread_promise.get_future();

  uint64_t encode_start = 0;

  auto thread_id = fan::event::thread_create([&render_thread_promise, &encode_start] {
    render_thread = (render_thread_t*)malloc(sizeof(render_thread_t));
    std::construct_at(render_thread);

    render_thread_promise.set_value(); // signal

    while (!render_thread->engine.should_close()) {
      render_thread->mutex.lock();
      if (encode_start != 0) {
        if (render_thread->screen_decode.FrameProcessStartTime == 0) {
          render_thread->screen_decode.init(encode_start);
        }
        //screen_encode.mutex.lock();
        if (screen_encode.buffer_copy.size() > 0) {
          render_thread->screen_decode.decode(
            screen_encode.buffer_copy.data(),
            screen_encode.buffer_copy.size(),
            render_thread->screen_frame
          );

          //uint64_t current_time = fan::time::clock::now();
          //uint64_t latency_ms = (current_time - timestamp) / 1000000;
          //  
          //if (latency_ms > 20) {
          //  fan::print_format("Latency: {}ms", latency_ms);
          //}

          screen_encode.buffer_copy.clear();
          can_encode.store(true);
        }
        //screen_encode.mutex.unlock();
        render_thread->screen_decode.sleep_thread(screen_encode.settings.InputFrameRate);
      }
      render_thread->mutex.unlock();

      render_thread->render();
    }
    delete render_thread;
  });

  render_thread_future.wait(); // wait for render_thread init

  // task queue - render thread -> main thread, process ecps_backend calls
  fan::event::task_idle([]() -> fan::event::task_t {
    if (render_thread == nullptr) {
      co_return;
    }
    std::vector<std::function<fan::event::task_t()>> local_tasks;
    {
      std::lock_guard<std::mutex> lock(render_thread->mutex);
      local_tasks = std::move(render_thread->ecps_gui.task_queue);
      render_thread->ecps_gui.task_queue.clear();
    }
    for (const auto& f : local_tasks) {
      co_await f();
    }
  });


  // codec task
  fan::event::task_idle([&encode_start]() -> fan::event::task_t {
    if (render_thread == nullptr) {
      co_return;
    }

    if (can_encode.load() == false) {
      co_return;
    }

    std::lock_guard<std::mutex> lock1(render_thread->mutex);
    if (render_thread->ecps_gui.is_streaming) {
      if (encode_start == 0) {
        encode_start = fan::time::clock::now();
        screen_encode.init(encode_start);
      }
      if (!screen_encode.screen_read()) {
        co_return;
      }
      timestamp = fan::time::clock::now();
      if (!screen_encode.encode_write(render_thread->screen_decode.FrameProcessStartTime)) {
        co_return;
      }
      screen_encode.encode_read();

      screen_encode.buffer_copy.resize(screen_encode.amount);
      std::memcpy(screen_encode.buffer_copy.data(), screen_encode.data, screen_encode.amount);
      screen_encode.frame_counter.fetch_add(1);
      static uint64_t last_stats = 0;
      uint64_t now = fan::time::clock::now();

      if (now - last_stats > 5000000000ULL) {
        fan::print_format("Frames encoded: {}, latest size: {} bytes",
          screen_encode.frame_counter.load(), screen_encode.amount);
        last_stats = now;
      }
      can_encode.store(!screen_encode.amount);
    }
  });

  fan::event::loop();
}