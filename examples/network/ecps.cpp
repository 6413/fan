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

#include "backend.h"

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

void ecps_backend_t::view_t::WriteFramePacket() {
#if set_VerboseProtocol_HoldStreamTimes == 1
  uint64_t t = T_nowi();
  WriteInformation("[CLIENT] [DEBUG] VerboseTime WriteFramePacket0 %llu\r\n", t - this->_VerboseTime);
  this->_VerboseTime = t;
#endif
  uint32_t FramePacketSize = (uint32_t)(this->m_Possible - 1) * 0x400 + this->m_ModuloSize;


  render_thread->mutex.lock();

  std::vector<uint8_t> data(FramePacketSize);
  __builtin_memcpy(data.data(), ecps_backend.view.m_data.data(), data.size());
  frame_packets.emplace_back(std::move(data));

  //if (Decode->FramePacketList.Usage() == 1) {
  //  this->ThreadCommon->StartDecoder();
  //}

#if set_VerboseProtocol_HoldStreamTimes == 1
  fp->_VerboseTime = T_nowi();
  WriteInformation("[CLIENT] [DEBUG] VerboseTime WriteFramePacket1 %llu %lx\r\n", fp->_VerboseTime - this->_VerboseTime, fpnr);
#endif

  render_thread->mutex.unlock();
}

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
        if (screen_encode.buffer_copy.size() > 0) {
          render_thread->screen_decode.decode(
            screen_encode.buffer_copy.data(),
            screen_encode.buffer_copy.size(),
            render_thread->screen_frame
          );

          screen_encode.buffer_copy.clear();
          can_encode.store(true);
        }
        render_thread->screen_decode.sleep_thread(screen_encode.settings.InputFrameRate);
      }

      if (ecps_backend.view.frame_packets.size()) {
        render_thread->screen_decode.decode(
          ecps_backend.view.frame_packets.front().data.data(),
          ecps_backend.view.frame_packets.front().data.size(),
          render_thread->screen_frame
        );
        ecps_backend.view.frame_packets.pop_front();
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

    render_thread->mutex.lock();
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
      while (screen_encode.encode_read() > 0) {
        std::size_t prev_size = screen_encode.buffer_copy.size();
        screen_encode.buffer_copy.resize(prev_size + screen_encode.amount);
        std::memcpy(screen_encode.buffer_copy.data() + prev_size, screen_encode.data, screen_encode.amount);
      }
      ;//
      ////

      {
        uint16_t Possible = (screen_encode.buffer_copy.size() / 0x400) + !!(screen_encode.buffer_copy.size() % 0x400);
        uint16_t Current = 0;
        for (; Current < Possible;Current++) {
          uintptr_t DataSize = screen_encode.buffer_copy.size() - Current * 0x400;
          if (DataSize > 0x400) {
            DataSize = 0x400;
          }
          render_thread->mutex.unlock();
          co_await ecps_backend.write_stream(Current, Possible, 0, &screen_encode.buffer_copy[Current * 0x400], DataSize);
          render_thread->mutex.lock();
        }
        ++ecps_backend.share.frame_index;
      }

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
    render_thread->mutex.unlock();
  });

  fan::event::loop();
}