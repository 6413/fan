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
#include <condition_variable>
#include <atomic>
#include <chrono>

#include <cuda.h>

import fan;
import fan.fmt;
import fan.graphics.video.screen_codec;
using namespace fan::graphics;

#include "backend.h"

struct screen_encode_t : fan::graphics::screen_encode_t {
  std::vector<uint8_t> buffer_copy;
  std::condition_variable encode_cv;
  std::atomic<uint64_t> decoder_timestamp{0};
  std::atomic<uint64_t> frame_counter{0};
  std::atomic<bool> has_encoded_data{false};
}screen_encode;

struct render_thread_t {
  engine_t engine;
  #define engine OFFSETLESS(This, render_thread_t, ecps_gui)->engine
  #include "gui.h"
  ecps_gui_t ecps_gui;
  
  std::mutex frame_mutex;
  std::mutex task_mutex;
  
  std::condition_variable frame_cv;
  std::condition_variable task_cv;
  
  // Work flags
  std::atomic<bool> has_frame_work{false};
  std::atomic<bool> has_task_work{false};
  std::atomic<bool> should_stop{false};

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

fan::event::task_t ecps_backend_t::default_s2c_cb(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
  co_await backend.tcp_client.read(backend.Protocol_S2C.NA(base.Command)->m_DSS); // advance tcp data
//  fan::print("unhandled callback");
  co_return;
}

void ecps_backend_t::view_t::WriteFramePacket() {
#if set_VerboseProtocol_HoldStreamTimes == 1
  uint64_t t = T_nowi();
  WriteInformation("[CLIENT] [DEBUG] VerboseTime WriteFramePacket0 %llu\r\n", t - this->_VerboseTime);
  this->_VerboseTime = t;
#endif
  if (m_Possible == 0) {
    return;
  }
  uint32_t FramePacketSize = (uint32_t)(this->m_Possible - 1) * 0x400 + this->m_ModuloSize;

  std::vector<uint8_t> data(FramePacketSize);
  __builtin_memcpy(data.data(), ecps_backend.view.m_data.data(), data.size());

  {
    std::lock_guard<std::mutex> lock(render_thread->frame_mutex);
    frame_packets.emplace_back(std::move(data));
    render_thread->has_frame_work.store(true);
  }
  
  render_thread->frame_cv.notify_one();

#if set_VerboseProtocol_HoldStreamTimes == 1
  fp->_VerboseTime = T_nowi();
  WriteInformation("[CLIENT] [DEBUG] VerboseTime WriteFramePacket1 %llu %lx\r\n", fp->_VerboseTime - this->_VerboseTime, fpnr);
#endif
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

    while (!render_thread->engine.should_close() && !render_thread->should_stop.load()) {
      bool did_work = false;
      
      {
        std::unique_lock<std::mutex> decode_lock(render_thread->screen_decode.mutex);
        
        if (encode_start != 0) {
          if (render_thread->screen_decode.FrameProcessStartTime == 0) {
            render_thread->screen_decode.init(encode_start);
          }
          
          if (screen_encode.encode_cv.wait_for(decode_lock, std::chrono::milliseconds(1), 
              [&]{ return screen_encode.has_encoded_data.load() || render_thread->should_stop.load(); })) {
            
            if (screen_encode.has_encoded_data.load()) {
              std::lock_guard<std::mutex> encode_lock(screen_encode.mutex);
              if (screen_encode.buffer_copy.size() > 0) {
                render_thread->screen_decode.decode(
                  screen_encode.buffer_copy.data(),
                  screen_encode.buffer_copy.size(),
                  render_thread->screen_frame
                );

                screen_encode.buffer_copy.clear();
                screen_encode.has_encoded_data.store(false);
                can_encode.store(true);
                did_work = true;
              }
            }
          }
          
          if (did_work) {
            render_thread->screen_decode.sleep_thread(screen_encode.settings.InputFrameRate);
          }
        }
      }

      {
        std::unique_lock<std::mutex> decode_lock(render_thread->screen_decode.mutex);
        std::unique_lock<std::mutex> frame_lock(render_thread->frame_mutex);
        
        if (render_thread->frame_cv.wait_for(frame_lock, std::chrono::milliseconds(1),
            [&]{ return render_thread->has_frame_work.load() || render_thread->should_stop.load(); })) {
          
          if (render_thread->has_frame_work.load() && ecps_backend.view.frame_packets.size()) {
            constexpr size_t max_batch_size = 5;
            size_t processed = 0;
            
            while (ecps_backend.view.frame_packets.size() && processed < max_batch_size) {
              render_thread->screen_decode.decode(
                ecps_backend.view.frame_packets.front().data.data(),
                ecps_backend.view.frame_packets.front().data.size(),
                render_thread->screen_frame
              );
              ecps_backend.view.frame_packets.pop_front();
              processed++;
              did_work = true;
            }
            
            if (ecps_backend.view.frame_packets.empty()) {
              render_thread->has_frame_work.store(false);
            }
          }
        }
      }

      render_thread->render();
      
      if (!did_work) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    
    delete render_thread;
  });

  render_thread_future.wait(); // wait for render_thread init

  // task queue - render thread -> main thread, process ecps_backend calls
  fan::event::task_idle([]() -> fan::event::task_t {
    if (render_thread == nullptr || render_thread->should_stop.load()) {
      co_return;
    }
    
    std::vector<std::function<fan::event::task_t()>> local_tasks;
    
    {
      std::unique_lock<std::mutex> task_lock(render_thread->task_mutex);
      
      if (!render_thread->ecps_gui.task_queue.empty()) {
        local_tasks = std::move(render_thread->ecps_gui.task_queue);
        render_thread->ecps_gui.task_queue.clear();
        render_thread->has_task_work.store(false);
      }
    }
    
    for (const auto& f : local_tasks) {
      co_await f();
    }
    
    if (local_tasks.empty()) {
      co_await fan::co_sleep(1);
    }
  });

  fan::time::clock clock;
  clock.start(5000000);
  
  fan::event::task_idle([&clock, &encode_start]() -> fan::event::task_t {
    if (render_thread == nullptr || render_thread->should_stop.load()) {
      co_return;
    }

    if (!can_encode.load()) {
      co_await fan::co_sleep(1);
      co_return;
    }

    if (render_thread->ecps_gui.is_streaming && clock.finished()) {
      clock.restart();
      
      if (encode_start == 0) {
        encode_start = fan::time::clock::now();
        screen_encode.init(encode_start);
      }
      
      if (!screen_encode.screen_read()) {
        fan::print("screen_read failed");
        co_return;
      }
      
      timestamp = fan::time::clock::now();
      if (!screen_encode.encode_write(timestamp)) {
        co_return;
      }
      
      uint64_t encoded_count = 0;
      std::vector<uint8_t> local_buffer;
      
      {
        std::lock_guard<std::mutex> encode_lock(screen_encode.mutex);
        
        while (screen_encode.encode_read() > 0) {
          encoded_count += screen_encode.amount;
          std::size_t prev_size = screen_encode.buffer_copy.size();
          screen_encode.buffer_copy.resize(prev_size + screen_encode.amount);
          std::memcpy(screen_encode.buffer_copy.data() + prev_size, screen_encode.data, screen_encode.amount);
        }
        
        if (encoded_count == 0) {
          co_return;
        }
        
        local_buffer = screen_encode.buffer_copy;
        screen_encode.has_encoded_data.store(true);
      }
      
      screen_encode.encode_cv.notify_one();
      
      {
        uint16_t Possible = (local_buffer.size() / 0x400) + !!(local_buffer.size() % 0x400);
        uint16_t Current = 0;
        
        for (; Current < Possible; Current++) {
          uintptr_t DataSize = local_buffer.size() - Current * 0x400;
          if (DataSize > 0x400) {
            DataSize = 0x400;
          }
          co_await ecps_backend.write_stream(Current, Possible, 0, &local_buffer[Current * 0x400], DataSize);
        }
        ++ecps_backend.share.frame_index;
      }

      screen_encode.frame_counter.fetch_add(1);
      
      static uint64_t last_stats = 0;
      uint64_t now = fan::time::clock::now();

      if (now - last_stats > 5000000000ULL) {
        fan::print_format("Frames encoded: {}, latest size: {} bytes",
          screen_encode.frame_counter.load(), encoded_count);
        last_stats = now;
      }
      
      can_encode.store(encoded_count == 0);
    }
    else {
      co_await fan::co_sleep(5);
    }
  });

  fan::event::loop();
  
  if (render_thread) {
    render_thread->should_stop.store(true);
    render_thread->frame_cv.notify_all();
    render_thread->task_cv.notify_all();
    screen_encode.encode_cv.notify_all();
  }
}