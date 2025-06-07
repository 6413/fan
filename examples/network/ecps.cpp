#include <fan/types/types.h>
#include <fan/time/timer.h>
#include <fan/event/types.h>
#include <fan/types/dme.h>
#include <cstring>
#include <unordered_map>
#include <exception>
#include <coroutine>
#include <string>
#include <array>
#include <functional>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <future>
#include <queue>

#if __has_include("cuda.h")
  #include <cuda.h>
#endif

import fan;
import fan.fmt;
import fan.graphics.video.screen_codec;
using namespace fan::graphics;

std::mutex render_mutex;
std::mutex task_mutex;
std::mutex frame_list_mutex;

struct screen_decode_t;
::screen_decode_t* screen_decode = 0;

struct screen_encode_t;
::screen_encode_t* screen_encode = 0;

struct render_thread_t;
render_thread_t* render_thread = 0;

#include "backend.h"

struct screen_encode_t : fan::graphics::screen_encode_t {
  std::condition_variable encode_cv;
  std::atomic<uint64_t> decoder_timestamp{0};
  std::atomic<uint64_t> frame_counter{0};
  std::atomic<bool> has_encoded_data{false};
};

struct screen_decode_t : fan::graphics::screen_decode_t {

};

ecps_backend_t::ecps_backend_t() {
  __dme_get(Protocol_S2C, KeepAlive) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    fan::print("tcp keep alive came");
    backend.tcp_keep_alive.reset();
    co_return;
  };
  __dme_get(Protocol_S2C, InformInvalidIdentify) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::InformInvalidIdentify_t>();
    if (msg->ClientIdentify != identify_secret) {
      co_return;
    }
    identify_secret = msg->ServerIdentify;
    fan::print("inform invalid identify came");
    co_return;
  };
    
  __dme_get(Protocol_S2C, Response_Login) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Response_Login_t>();
    fan::print_format(R"({{
[SERVER] Response_login
SessionID: {}
AccountID: {}
}})", msg->SessionID.i, msg->AccountID.i);
    backend.session_id = msg->SessionID;
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
  __dme_get(Protocol_S2C, JoinChannel_OK) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_OK_t>();
    fan::print_format(R"({{
[SERVER] JoinChannel_OK
ID: {}
ChannelID: {}
}})", base.ID, msg->ChannelID.i);
    channel_info.front().session_id = msg->ChannelSessionID;
  };
  __dme_get(Protocol_S2C, JoinChannel_Error) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_Error_t>();
    fan::print_format(R"({{
[SERVER] JoinChannel_Error
ID: {}
ChannelID: {}
}})", base.ID, Protocol::JoinChannel_Error_Reason_String[(uint8_t)msg->Reason]);
  };

  __dme_get(Protocol_S2C, Channel_ScreenShare_View_InformationToViewSetFlag) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_View_InformationToViewSetFlag_t>();
    if (msg->Flag & ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
    }
  };
}

void ecps_backend_t::share_t::CalculateNetworkFlowBucket() {
  uintptr_t MaxBufferSize = (sizeof(ScreenShare_StreamHeader_Head_t) + 0x400) * 8;
  m_NetworkFlow.BucketSize = screen_encode->settings.RateControl.TBR.bps / 8;
  if (m_NetworkFlow.BucketSize < MaxBufferSize) {
    fan::print_format("[CLIENT] [WARNING] {} {}:{} BucketSize rounded up", __FUNCTION__, __FILE__, __LINE__);
    m_NetworkFlow.BucketSize = MaxBufferSize;
  }
  if (m_NetworkFlow.BucketSize / 8 > 0x7fff) {
    fan::print_format("[CLIENT] [WARNING] {} {}:{} BucketSize({}) is too big", __FUNCTION__, __FILE__, __LINE__, m_NetworkFlow.BucketSize);
  }
}

struct render_thread_t {

  render_thread_t() {
    engine.set_target_fps(0, false);
  }

  engine_t engine;
  #define engine OFFSETLESS(This, render_thread_t, ecps_gui)->engine
  #include "gui.h"
  ecps_gui_t ecps_gui;

  std::condition_variable frame_cv;
  std::condition_variable task_cv;
  
  std::atomic<bool> has_task_work{false};
  std::atomic<bool> should_stop{false};

  fan::graphics::universal_image_renderer_t screen_frame{{
    .position = fan::vec3(gloco->window.get_size() / 2, 0),
    .size = gloco->window.get_size() / 2,
  }};
  fan::graphics::sprite_t screen_frame_hider{{
    .position = fan::vec3(gloco->window.get_size() / 2, 1),
    .size = gloco->window.get_size() / 2,
  }};

  void render(auto l) {
    engine.process_loop([this, l] { ecps_gui.render(); l();  });
  }

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix FrameList
#define BLL_set_Language 1
#define BLL_set_Usage 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeDataType fan::graphics::screen_decode_t::decode_data_t
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>
  FrameList_t FrameList; // for decoding

};

fan::event::task_t ecps_backend_t::default_s2c_cb(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
  co_await backend.tcp_client.read(backend.Protocol_S2C.NA(base.Command)->m_DSS); // advance tcp data
//  fan::print("unhandled callback");
  co_return;
}

struct frame_timing_t {
  std::chrono::steady_clock::time_point encode_time;
  std::chrono::steady_clock::time_point decode_time;
  uint64_t frame_id;
  std::vector<uint8_t> data;
  enum source_type_t { LOCAL, NETWORK } source;
};

std::queue<frame_timing_t> local_decode_queue;
std::queue<frame_timing_t> network_decode_queue;
std::mutex local_decode_mutex;
std::mutex network_decode_mutex;
std::condition_variable decode_queue_cv;
const size_t MAX_DECODE_QUEUE_SIZE = 10;

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

  static std::vector<uint8_t> data;
  data.resize(FramePacketSize);
  __builtin_memcpy(data.data(), ecps_backend.view.m_data.data(), data.size());

  {
    std::lock_guard<std::mutex> lock(render_mutex);
    while (network_decode_queue.size() >= MAX_DECODE_QUEUE_SIZE) {
      network_decode_queue.pop();
    }
    
    frame_timing_t frame_data;
    frame_data.encode_time = std::chrono::steady_clock::now();
    frame_data.frame_id = frame_index++;
    frame_data.data = std::move(data);
    frame_data.source = frame_timing_t::NETWORK;
    
    network_decode_queue.push(std::move(frame_data));
  }
  
  render_thread->frame_cv.notify_one();

#if set_VerboseProtocol_HoldStreamTimes == 1
  fp->_VerboseTime = T_nowi();
  WriteInformation("[CLIENT] [DEBUG] VerboseTime WriteFramePacket1 %llu %lx\r\n", fp->_VerboseTime - this->_VerboseTime, fpnr);
#endif
}

int main() {
  std::promise<void> render_thread_promise;
  std::future<void> render_thread_future = render_thread_promise.get_future();

  uint64_t encode_start = 0;

  auto encode_thread_id = fan::event::thread_create([] {
    screen_encode = (::screen_encode_t*)malloc(sizeof(::screen_encode_t));
    std::construct_at(screen_encode);

    while (screen_decode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    uint64_t frame_id = 0;

    while (1) {
      if (render_thread->ecps_gui.is_streaming) {
        auto encode_start = std::chrono::steady_clock::now();

        if (!screen_encode->screen_read()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }

        if (!screen_encode->encode_write()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }

        uint64_t encoded_count = 0;
        std::vector<uint8_t> encoded_buffer;

        {
          std::lock_guard<std::mutex> encode_lock(screen_encode->mutex);

          while (screen_encode->encode_read() > 0) {
            encoded_count += screen_encode->amount;
            encoded_buffer.resize(encoded_count);
            std::memcpy(encoded_buffer.data() + (encoded_buffer.size() - screen_encode->amount),
              screen_encode->data, screen_encode->amount);
          }

          if (encoded_count == 0) {
            continue;
          }

          // add to network queue
          {
            std::lock_guard<std::mutex> frame_list_lock(ecps_backend.share.frame_list_mutex);
            auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.NewNodeLast();
            auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];
            f->vec = encoded_buffer;
            f->SentOffset = 0;
          }

          // add to local queue
          {
            std::lock_guard<std::mutex> decode_lock(local_decode_mutex);

            while (local_decode_queue.size() >= MAX_DECODE_QUEUE_SIZE) {
              local_decode_queue.pop();
            }

            frame_timing_t frame_data;
            frame_data.encode_time = encode_start;
            frame_data.frame_id = frame_id++;
            frame_data.data = encoded_buffer;
            frame_data.source = frame_timing_t::LOCAL;

            local_decode_queue.push(std::move(frame_data));
          }

          screen_encode->has_encoded_data.store(true);
        }

        decode_queue_cv.notify_one();
        screen_encode->encode_cv.notify_one();
        screen_encode->frame_counter.fetch_add(1);
        screen_encode->sleep_thread();
      }
      else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  });



  auto decode_thread_id = fan::event::thread_create([] {
    screen_decode = (::screen_decode_t*)malloc(sizeof(::screen_decode_t));
    std::construct_at(screen_decode);

    while (screen_encode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    while (1) {
      bool processed_frame = false;
      auto now = std::chrono::steady_clock::now();

      // process network frames
      {
        std::lock_guard<std::mutex> network_lock(network_decode_mutex);
        if (!network_decode_queue.empty()) {
          frame_timing_t frame_data = std::move(network_decode_queue.front());
          network_decode_queue.pop();

          frame_data.decode_time = now;

          // Network frames are more tolerant of delay (up to 200ms)
          auto processing_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_data.decode_time - frame_data.encode_time).count();

          if (processing_delay <= 200) {
            fan::graphics::screen_decode_t::decode_data_t decode_data = screen_decode->decode(
              frame_data.data.data(),
              frame_data.data.size(),
              render_thread->screen_frame
            );
            // or if cuvid
            if (!decode_data.data[0].empty() || decode_data.type == 0) {
              std::lock_guard<std::mutex> frame_lock(render_mutex);
              auto flnr = render_thread->FrameList.NewNodeLast();
              auto f = &render_thread->FrameList[flnr];
              *f = std::move(decode_data);
              processed_frame = true;
            }
          }
          else {
            fan::print_format("Skipping old network frame {}, delay: {}ms", frame_data.frame_id, processing_delay);
          }
        }
      }

      // process only local frames if no network frame was processed, they like to race
      if (!processed_frame) {
        std::lock_guard<std::mutex> local_lock(local_decode_mutex);
        if (!local_decode_queue.empty()) {
          frame_timing_t frame_data = std::move(local_decode_queue.front());
          local_decode_queue.pop();

          frame_data.decode_time = now;

          // Local frames should be more recent (up to 100ms)
          auto processing_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_data.decode_time - frame_data.encode_time).count();

          if (processing_delay <= 100) {
            fan::graphics::screen_decode_t::decode_data_t decode_data = screen_decode->decode(
              frame_data.data.data(),
              frame_data.data.size(),
              render_thread->screen_frame
            );

            if (!decode_data.data[0].empty() || decode_data.type == 0) {
              std::lock_guard<std::mutex> frame_lock(render_mutex);
              auto flnr = render_thread->FrameList.NewNodeLast();
              auto f = &render_thread->FrameList[flnr];
              *f = std::move(decode_data);
              processed_frame = true;
            }
          }
          else {
            fan::print_format("Skipping old local frame {}, delay: {}ms", frame_data.frame_id, processing_delay);
          }
        }
      }

      if (!processed_frame) {
        std::unique_lock<std::mutex> wait_lock(local_decode_mutex);
        decode_queue_cv.wait_for(wait_lock, std::chrono::milliseconds(16), [&] {
          return !local_decode_queue.empty() || !network_decode_queue.empty() || render_thread->should_stop.load();
        });
      }
    }
  });

  auto render_thread_id = fan::event::thread_create([&render_thread_promise, &encode_start] {
    render_thread = (render_thread_t*)malloc(sizeof(render_thread_t));
    std::construct_at(render_thread);

    render_thread_promise.set_value(); // signal
    
    while (!render_thread->engine.should_close() && !render_thread->should_stop.load()) {

      // process decoded data
      {
        std::lock_guard<std::mutex> render_lock(render_mutex);
        std::lock_guard<std::mutex> decode_lock(screen_decode->mutex);
        auto flnr = render_thread->FrameList.GetNodeFirst();
        if (flnr == render_thread->FrameList.dst) {
          goto g_feed_frames_skip;
        }
        auto& node = render_thread->FrameList[flnr];
        if (node.type == 0) {// cuvid
          screen_decode->decoder_change_cb(); // only required here because cuvid needs to call it in opengl thread
          screen_decode->decode_cuvid(render_thread->screen_frame);
        }
        else if (node.type == 1) {
          f32_t sx = (f32_t)node.image_size.x / node.stride[0].x;
          std::array<void*, 4> raw_ptrs;
          for (size_t i = 0; i < 4; ++i) {
            raw_ptrs[i] = static_cast<void*>(node.data[i].data());
          }

          render_thread->screen_frame.set_tc_size(fan::vec2(sx, 1));
          render_thread->screen_frame.reload(node.pixel_format, raw_ptrs.data(), fan::vec2ui(node.stride[0].x, node.image_size.y));
        }
        render_thread->FrameList.unlrec(flnr);
      }
g_feed_frames_skip:

      bool did_work = false;
      // render lock?
      render_thread->render([]{
        gui::begin("debug");
        gui::drag_int("bucket size", (int*)&ecps_backend.share.m_NetworkFlow.BucketSize, 1000, 0, std::numeric_limits<int>::max() - 1, "%d", gui::slider_flags_always_clamp);
        gui::end();
      });
      
      if (!did_work) {
        std::this_thread::yield();
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
      std::unique_lock<std::mutex> task_lock(render_mutex);
      
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


  while (screen_encode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    continue;
  }

  auto network_task_id = fan::event::task_timer(5, []() -> fan::event::task_value_resume_t<bool> {
    if (ecps_backend.channel_info.empty()) {
      co_return 0;
    }

    //std::lock_guard<std::mutex> frame_lock(render_thread->render_mutex);
    std::lock_guard<std::mutex> frame_list_lock(ecps_backend.share.frame_list_mutex);

    uint64_t ctime = fan::event::now();
    uint64_t DeltaTime = ctime - ecps_backend.share.m_NetworkFlow.TimerLastCallAt;
    ecps_backend.share.m_NetworkFlow.TimerLastCallAt = ctime;

    ecps_backend.share.m_NetworkFlow.Bucket +=
      (f32_t)DeltaTime / 1000000000 * screen_encode->settings.RateControl.TBR.bps * 2;
    if (ecps_backend.share.m_NetworkFlow.Bucket > ecps_backend.share.m_NetworkFlow.BucketSize) {
      ecps_backend.share.m_NetworkFlow.Bucket = ecps_backend.share.m_NetworkFlow.BucketSize;
    }

    auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
    if (flnr == ecps_backend.share.m_NetworkFlow.FrameList.dst) {
      /* no frame waiting for us */
      co_return 0;
    }

    f32_t Delay = (f32_t)ecps_backend.share.m_NetworkFlow.FrameList.Usage() / screen_encode->settings.InputFrameRate;
    if (Delay >= 0.4) {
      //WriteInformation("[CLIENT] [WARNING] %s %s:%u Delay is above ~0.4s %lu %f\r\n", __FUNCTION__, __FILE__, __LINE__, ecps_backend.share.m_NetworkFlow.FrameList.Usage(), screen_encode->ThreadCommon->EncoderSetting.EncoderSetting.Setting.InputFrameRate);
      if (ecps_backend.share.m_NetworkFlow.Bucket == ecps_backend.share.m_NetworkFlow.BucketSize) {
        //WriteInformation("  Bucket == BucketSize, maybe decrease Interval and increase BucketSize?\r\n");
      }
    }

    auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];

    uint8_t Flag = 0; /* used to get flags from encode but looks useless */
    uint16_t Possible = (f->vec.size() / 0x400) + !!(f->vec.size() % 0x400);
    uint16_t Current = 0;
    std::vector<uint8_t> frame_data = f->vec;
    uint16_t sent_offset = f->SentOffset; // Start from last position
    auto sent = sent_offset * 0x400; // Calculate already sent bytes

    for (; sent_offset < Possible; sent_offset++) {
      uintptr_t DataSize = f->vec.size() - sent_offset * 0x400;
      if (DataSize > 0x400) {
        DataSize = 0x400;
      }

      // Add bucket check
      if (ecps_backend.share.m_NetworkFlow.Bucket < DataSize) {
        // fan::print_format("Bucket empty: need {}, have {}", DataSize, ecps_backend.share.m_NetworkFlow.Bucket);
        break;
      }

      bool ret = co_await ecps_backend.write_stream(sent_offset, Possible, Flag, &f->vec[sent_offset * 0x400], DataSize);
      sent += DataSize;

      if (ret != false) {
        // fan::print_format("write_stream failed at chunk {}/{}", sent_offset, Possible);
        break;
      }

      // Deduct from bucket
      ecps_backend.share.m_NetworkFlow.Bucket -= DataSize;
    }

    //fan::print_format("sent {} bytes, chunk {}/{}", sent, sent_offset, Possible);

    f->SentOffset = sent_offset;

    if (sent_offset >= Possible) {
      //fan::print("Frame completely sent, removing from queue");
      f->vec.clear();
      ecps_backend.share.m_NetworkFlow.FrameList.unlrec(flnr);
      ++ecps_backend.share.frame_index;
    }
    else {
      // fan::print_format("Frame partially sent {}/{}, will continue next iteration", sent_offset, Possible);
    }
    co_return 0;
  });

  fan::event::loop();
  
  if (render_thread) {
    render_thread->should_stop.store(true);
    render_thread->frame_cv.notify_all();
    render_thread->task_cv.notify_all();
    screen_encode->encode_cv.notify_all();
  }
}