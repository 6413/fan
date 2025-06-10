////
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
#include <memory>
//
#if __has_include("cuda.h")
#include <cuda.h>
#endif

import fan;
import fan.fmt;
import fan.graphics.video.screen_codec;
using namespace fan::graphics;

#include <fan/graphics/types.h>

std::timed_mutex render_mutex;
std::timed_mutex task_mutex;

struct screen_decode_t;
::screen_decode_t* screen_decode = 0;

struct screen_encode_t;
::screen_encode_t* screen_encode = 0;

struct render_thread_t;
render_thread_t* render_thread = 0;

#include "backend.h"

class FrameMemoryPool {
private:
  static constexpr size_t INITIAL_POOL_SIZE = 64;
  static constexpr size_t MAX_POOL_SIZE = 256;
  static constexpr size_t FRAME_SIZE = 0x400400;

  struct FrameBuffer {
    alignas(64) std::vector<uint8_t> data;
    std::atomic<bool> in_use{ false };
    std::chrono::steady_clock::time_point last_used;

    FrameBuffer() {
      data.reserve(FRAME_SIZE);
    }
  };

  std::vector<std::unique_ptr<FrameBuffer>> buffers;
  std::atomic<size_t> pool_size{ INITIAL_POOL_SIZE };
  mutable std::mutex pool_mutex;
  std::atomic<size_t> allocation_failures{ 0 };

public:
  FrameMemoryPool() {
    buffers.reserve(MAX_POOL_SIZE);
    for (size_t i = 0; i < INITIAL_POOL_SIZE; ++i) {
      buffers.emplace_back(std::make_unique<FrameBuffer>());
    }
  }

  std::shared_ptr<std::vector<uint8_t>> acquire() {
    std::unique_lock<std::mutex> lock(pool_mutex);

    for (auto& buffer : buffers) {
      bool expected = false;
      if (buffer->in_use.compare_exchange_strong(expected, true)) {
        buffer->data.clear();
        buffer->last_used = std::chrono::steady_clock::now();

        return std::shared_ptr<std::vector<uint8_t>>(
          &buffer->data,
          [buffer_ptr = buffer.get()](std::vector<uint8_t>*) {
            buffer_ptr->in_use.store(false);
          }
        );
      }
    }

    // Expand pool if needed and under limit
    if (buffers.size() < MAX_POOL_SIZE) {
      auto new_buffer = std::make_unique<FrameBuffer>();
      new_buffer->in_use.store(true);
      new_buffer->last_used = std::chrono::steady_clock::now();
      auto* data_ptr = &new_buffer->data;
      buffers.push_back(std::move(new_buffer));

      return std::shared_ptr<std::vector<uint8_t>>(
        data_ptr,
        [buffer_ptr = buffers.back().get()](std::vector<uint8_t>*) {
          buffer_ptr->in_use.store(false);
        }
      );
    }

    allocation_failures.fetch_add(1);
    if (allocation_failures % 100 == 1) {
      fan::print("Warning: Frame pool exhausted, creating temporary buffer");
    }
    return std::make_shared<std::vector<uint8_t>>();
  }
} frame_pool;

struct screen_encode_t : fan::graphics::screen_encode_t {
  std::condition_variable encode_cv;
  std::atomic<uint64_t> decoder_timestamp{ 0 };
  std::atomic<uint64_t> frame_counter{ 0 };
  std::atomic<bool> has_encoded_data{ false };
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

    // ADDED: Request IDR frame immediately on join
    if (screen_encode) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
    }
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

  __dme_get(Protocol_S2C, ChannelList) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
  auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::ChannelList_t>();
//  
//  fan::print_format(R"({{
//[SERVER] ChannelList
//ChannelCount: {}
//}})", msg->ChannelCount);

  backend.available_channels.clear();
  backend.available_channels.reserve(msg->ChannelCount);
  
  // Read the channel info array
  for (uint16_t i = 0; i < msg->ChannelCount; ++i) {
    auto channel_info = co_await backend.tcp_client.read<Protocol_S2C_t::ChannelInfo_t>();
    
    ecps_backend_t::channel_list_info_t info;
    info.channel_id = channel_info->ChannelID;
    info.type = channel_info->Type;
    info.user_count = channel_info->UserCount;
    info.name = std::string(channel_info->Name, strnlen(channel_info->Name, 63));
    info.is_password_protected = (channel_info->IsPasswordProtected != 0);
    info.host_session_id = channel_info->HostSessionID;
    
    backend.available_channels.push_back(info);
    //
    //fan::print_format(R"(  Channel {}: "{}" ({} users, host: {}))", 
    //  info.channel_id.i, info.name, info.user_count, info.host_session_id.i);
  }
  
  backend.channel_list_received = true;
};

__dme_get(Protocol_S2C, ChannelSessionList) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
  auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::ChannelSessionList_t>();
  
//  fan::print_format(R"({{
//[SERVER] ChannelSessionList
//ChannelID: {}
//SessionCount: {}
//}})", msg->ChannelID.i, msg->SessionCount);

  Protocol_ChannelID_t channel_id = msg->ChannelID;
  
  // Clear existing sessions for this channel
  backend.channel_sessions[channel_id.i].clear();
  backend.channel_sessions[channel_id.i].reserve(msg->SessionCount);
  
  // Read the session info array
  for (uint16_t i = 0; i < msg->SessionCount; ++i) {
    auto session_info = co_await backend.tcp_client.read<Protocol_S2C_t::SessionInfo_t>();
    
    ecps_backend_t::session_info_t info;
    info.session_id = session_info->SessionID;
    info.channel_session_id = session_info->ChannelSessionID;
    info.account_id = session_info->AccountID;
    info.username = std::string((const char*)session_info->Username, strnlen((const char*)session_info->Username, 31));
    info.is_host = (session_info->IsHost != 0);
    info.joined_at = session_info->JoinedAt;
    
    backend.channel_sessions[channel_id.i].push_back(info);
    
    //fan::print_format(R"(  User {}: "{}" (session: {}, account: {}, {}))", 
    //  i + 1, info.username, info.session_id.i, info.account_id.i, 
    //  info.is_host ? "HOST" : "viewer");
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

  // ADDED: Start with a larger initial bucket for faster startup
  m_NetworkFlow.Bucket = m_NetworkFlow.BucketSize * 0.5; // Start at 50% capacity

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

  std::atomic<bool> has_task_work{ false };
  std::atomic<bool> should_stop{ false };

  fan::graphics::universal_image_renderer_t screen_frame{ {
    .position = fan::vec3(gloco->window.get_size() / 2, 0),
    .size = gloco->window.get_size() / 2,
  } };
  fan::graphics::sprite_t screen_frame_hider{ {
    .position = fan::vec3(gloco->window.get_size() / 2, 1),
    .size = gloco->window.get_size() / 2,
  } };

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
  std::shared_ptr<std::vector<uint8_t>> data;
  enum source_type_t { LOCAL, NETWORK } source;

  frame_timing_t() = default;
  frame_timing_t(frame_timing_t&&) = default;
  frame_timing_t& operator=(frame_timing_t&&) = default;
  frame_timing_t(const frame_timing_t&) = default;
  frame_timing_t& operator=(const frame_timing_t&) = default;
};

std::queue<frame_timing_t> local_decode_queue;
std::queue<frame_timing_t> network_decode_queue;
std::timed_mutex local_decode_mutex;
std::timed_mutex network_decode_mutex;
std::condition_variable decode_queue_cv;
const size_t MAX_DECODE_QUEUE_SIZE = 8;

void ecps_backend_t::view_t::WriteFramePacket() {
  if (m_Possible == 0) {
    return;
  }

  uint32_t FramePacketSize = (uint32_t)(this->m_Possible - 1) * 0x400 + this->m_ModuloSize;
  {
    std::unique_lock<std::timed_mutex> lock(network_decode_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
      this->m_stats.Frame_Drop++;
      return;
    }

    if (network_decode_queue.size() >= MAX_DECODE_QUEUE_SIZE - 1) {
      while (!network_decode_queue.empty()) {
        network_decode_queue.pop();
      }
    }

    auto data_copy = frame_pool.acquire();
    data_copy->resize(FramePacketSize);
    std::memcpy(data_copy->data(), ecps_backend.view.m_data.data(), FramePacketSize);

    frame_timing_t frame_data;
    frame_data.encode_time = std::chrono::steady_clock::now();
    frame_data.frame_id = frame_index++;
    frame_data.data = data_copy;
    frame_data.source = frame_timing_t::NETWORK;

    network_decode_queue.push(std::move(frame_data));
  }

  render_thread->frame_cv.notify_one();
}

int main() {
  std::promise<void> render_thread_promise;
  std::future<void> render_thread_future = render_thread_promise.get_future();

  uint64_t encode_start = 0;

  auto encode_thread_id = fan::event::thread_create([] -> fan::event::task_t {
    screen_encode = (::screen_encode_t*)malloc(sizeof(::screen_encode_t));
    std::construct_at(screen_encode);

    while (screen_decode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    uint64_t frame_id = 0;
    bool first_frame = true;

    while (1) {
      if (render_thread->ecps_gui.is_streaming) {
        if (first_frame) {
          screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
          first_frame = false;
        }

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
        auto encoded_buffer = frame_pool.acquire();

        {
          std::lock_guard<std::timed_mutex> encode_lock(screen_encode->mutex);

          while (screen_encode->encode_read() > 0) {
            encoded_count += screen_encode->amount;
            encoded_buffer->resize(encoded_count);
            std::memcpy(encoded_buffer->data() + (encoded_buffer->size() - screen_encode->amount),
              screen_encode->data, screen_encode->amount);
          }

          if (encoded_count == 0) {
            continue;
          }

          {
            std::unique_lock<std::timed_mutex> frame_list_lock(ecps_backend.share.frame_list_mutex, std::try_to_lock);
            if (frame_list_lock.owns_lock()) {
              bool is_idr = (screen_encode->encode_write_flags & fan::graphics::codec_update_e::reset_IDR) != 0;
              if (is_idr) {
                while (ecps_backend.share.m_NetworkFlow.FrameList.Usage() > 0) {
                  auto old = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
                  ecps_backend.share.m_NetworkFlow.FrameList.unlrec(old);
                }
              }

              auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.NewNodeLast();
              auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];
              f->vec = *encoded_buffer;
              f->SentOffset = 0;
            }
          }

          {
            std::unique_lock<std::timed_mutex> decode_lock(local_decode_mutex, std::defer_lock);
            if (decode_lock.try_lock_for(std::chrono::milliseconds(3))) {
              while (local_decode_queue.size() >= MAX_DECODE_QUEUE_SIZE) {
                local_decode_queue.pop();
              }

              frame_timing_t frame_data;
              frame_data.encode_time = encode_start;
              frame_data.frame_id = frame_id++;
              frame_data.data = encoded_buffer;
              frame_data.source = frame_timing_t::LOCAL;

              local_decode_queue.push(frame_data);
              decode_queue_cv.notify_one();
            }
          }

          screen_encode->has_encoded_data.store(true);
        }

        decode_queue_cv.notify_one();
        screen_encode->encode_cv.notify_one();
        screen_encode->frame_counter.fetch_add(1);
        screen_encode->sleep_thread();
      }
      else {
        first_frame = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    });

  auto idr_refresh_task = fan::event::task_timer(5000, []() -> fan::event::task_value_resume_t<bool> {
    if (render_thread && render_thread->ecps_gui.is_streaming && screen_encode) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
    }
    co_return 0;
  });


  auto decode_thread_id = fan::event::thread_create([] {
    screen_decode = (::screen_decode_t*)malloc(sizeof(::screen_decode_t));
    std::construct_at(screen_decode);

    while (screen_encode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto last_network_frame_time = std::chrono::steady_clock::now();
    const auto MAX_FRAME_AGE = std::chrono::milliseconds(100);

    while (1) {
      bool processed_frame = false;
      auto now = std::chrono::steady_clock::now();

      {
        std::unique_lock<std::timed_mutex> network_lock(network_decode_mutex, std::try_to_lock);
        if (network_lock.owns_lock() && !network_decode_queue.empty()) {
          // Process the most recent frame, skip old ones
          frame_timing_t most_recent_frame;
          bool found_recent = false;

          while (!network_decode_queue.empty()) {
            auto frame = network_decode_queue.front();
            network_decode_queue.pop();

            auto age = now - frame.encode_time;
            if (age < MAX_FRAME_AGE) {
              most_recent_frame = std::move(frame);
              found_recent = true;
            }
          }

          if (found_recent) {
            network_lock.unlock();

            most_recent_frame.decode_time = now;
            last_network_frame_time = now;

            if (most_recent_frame.data && !most_recent_frame.data->empty()) {
              fan::graphics::screen_decode_t::decode_data_t decode_data = screen_decode->decode(
                most_recent_frame.data->data(),
                most_recent_frame.data->size(),
                render_thread->screen_frame
              );

              if (!decode_data.data[0].empty() || decode_data.type == 0) {
                std::lock_guard<std::timed_mutex> frame_lock(render_mutex);
                auto flnr = render_thread->FrameList.NewNodeLast();
                auto f = &render_thread->FrameList[flnr];
                *f = std::move(decode_data);
                processed_frame = true;
              }
            }
          }
        }
      }

      auto time_since_network = now - last_network_frame_time;
      if (!processed_frame && time_since_network > std::chrono::milliseconds(500)) {
        std::unique_lock<std::timed_mutex> local_lock(local_decode_mutex, std::try_to_lock);
        if (local_lock.owns_lock() && !local_decode_queue.empty()) {
          // Get most recent local frame
          frame_timing_t frame_data;
          while (!local_decode_queue.empty()) {
            frame_data = local_decode_queue.front();
            local_decode_queue.pop();
            if (local_decode_queue.empty()) break;
          }
          local_lock.unlock();

          frame_data.decode_time = now;

          if (frame_data.data && !frame_data.data->empty()) {
            fan::graphics::screen_decode_t::decode_data_t decode_data = screen_decode->decode(
              frame_data.data->data(),
              frame_data.data->size(),
              render_thread->screen_frame
            );

            if (!decode_data.data[0].empty() || decode_data.type == 0) {
              std::lock_guard<std::timed_mutex> frame_lock(render_mutex);
              auto flnr = render_thread->FrameList.NewNodeLast();
              auto f = &render_thread->FrameList[flnr];
              *f = std::move(decode_data);
              processed_frame = true;
            }
          }
        }
      }

      if (!processed_frame) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
      else {
        std::this_thread::yield();
      }
    }
    });

  auto render_thread_id = fan::event::thread_create([&render_thread_promise, &encode_start] {
    render_thread = (render_thread_t*)malloc(sizeof(render_thread_t));
    std::construct_at(render_thread);

    render_thread_promise.set_value(); // signal

    while (!render_thread->engine.should_close() && !render_thread->should_stop.load()) {

      {
        std::lock_guard<std::timed_mutex> render_lock(render_mutex);
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
          fan::graphics::image_data_t& image_data = render_thread->engine.image_get_data(render_thread->screen_frame.get_image());
          render_thread->screen_frame.reload(node.pixel_format, raw_ptrs.data(), fan::vec2ui(node.stride[0].x, node.image_size.y), image_data.image_settings.min_filter);
        }
        render_thread->FrameList.unlrec(flnr);
      }
    g_feed_frames_skip:

      bool did_work = false;
      render_thread->render([] {

        static const char* sampler_names[] = {
          "Nearest",
          "Linear",
          "Nearest Mipmap Nearest",
          "Linear Mipmap Nearest",
          "Nearest Mipmap Linear",
          "Linear Mipmap Linear"
        };

        static fan::graphics::image_filter sampler_filters[] = {
            fan::graphics::image_filter::nearest,
            fan::graphics::image_filter::linear,
            fan::graphics::image_filter::nearest_mipmap_nearest,
            fan::graphics::image_filter::linear_mipmap_nearest,
            fan::graphics::image_filter::nearest_mipmap_linear,
            fan::graphics::image_filter::linear_mipmap_linear
        };

        static int current_sampler = 0;

        std::vector<fan::graphics::image_t> img_list;
        img_list.emplace_back(render_thread->screen_frame.get_image());
        auto images = render_thread->screen_frame.get_images();
        img_list.insert(img_list.end(), images.begin(), images.end());
        static bool p_open_debug = false;
        if (fan::window::is_key_down(fan::key_shift) &&
          fan::window::is_key_pressed(fan::key_5)) {
          p_open_debug = !p_open_debug;
        }
        if (p_open_debug) {
          gui::begin("debug");
          static fan::event::task_t task;
          if (gui::button("get list")) {
             render_thread->ecps_gui.backend_queue([=]() -> fan::event::task_t {
              try {
                co_await ecps_backend.request_channel_list();
              }
              catch (...) {
                fan::print("Failed to request channel list");
              }
            });
          }
          if (gui::combo("Sampler", &current_sampler, sampler_names, std::size(sampler_names))) {
            auto selected_filter = sampler_filters[current_sampler];
            for (auto& i : img_list) {
              if (!render_thread->engine.is_image_valid(i)) {
                continue;
              }
              fan::graphics::image_data_t& image_data = render_thread->engine.image_list[i];
              image_data.image_settings.min_filter = selected_filter;
              if (selected_filter == fan::graphics::image_filter::linear ||
                selected_filter == fan::graphics::image_filter::nearest) {
                image_data.image_settings.mag_filter = selected_filter;
              }
              else {
                image_data.image_settings.mag_filter = fan::graphics::image_filter::linear;
              }
              render_thread->engine.image_set_settings(i, image_data.image_settings);
            }
          }
          gui::drag_int("bucket size", (int*)&ecps_backend.share.m_NetworkFlow.BucketSize, 1000, 0, std::numeric_limits<int>::max() - 1, "%d", gui::slider_flags_always_clamp);
          gui::end();
        }
      });

      if (!did_work) {
        std::this_thread::yield();
      }
    }

    delete render_thread;
    });

  render_thread_future.wait(); // wait for render_thread init

  fan::event::task_idle([]() -> fan::event::task_t {
    if (render_thread == nullptr || render_thread->should_stop.load()) {
      co_return;
    }

    constexpr size_t MAX_BATCH_SIZE = 4; // Reduced batch size for stability
    std::vector<std::function<fan::event::task_t()>> local_tasks;

    {
      std::unique_lock<std::timed_mutex> task_lock(render_mutex, std::defer_lock);

      if (task_lock.try_lock_for(std::chrono::milliseconds(5)) && !render_thread->ecps_gui.task_queue.empty()) {
        size_t batch_size = std::min(render_thread->ecps_gui.task_queue.size(), MAX_BATCH_SIZE);
        local_tasks.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
          local_tasks.emplace_back(std::move(render_thread->ecps_gui.task_queue[i]));
        }

        render_thread->ecps_gui.task_queue.erase(
          render_thread->ecps_gui.task_queue.begin(),
          render_thread->ecps_gui.task_queue.begin() + batch_size
        );

        if (render_thread->ecps_gui.task_queue.empty()) {
          render_thread->has_task_work.store(false);
        }
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

  auto network_task_id = fan::event::task_timer(2, []() -> fan::event::task_value_resume_t<bool> { // Reduced from 5ms to 2ms
    if (ecps_backend.channel_info.empty()) {
      co_return 0;
    }

    std::unique_lock<std::timed_mutex> frame_list_lock(ecps_backend.share.frame_list_mutex, std::try_to_lock);
    if (!frame_list_lock.owns_lock()) {
      co_return 0; // Skip if can't get lock immediately
    }

    uint64_t ctime = fan::event::now();
    uint64_t DeltaTime = ctime - ecps_backend.share.m_NetworkFlow.TimerLastCallAt;
    ecps_backend.share.m_NetworkFlow.TimerLastCallAt = ctime;

    ecps_backend.share.m_NetworkFlow.Bucket +=
      (f32_t)DeltaTime / 1000000000 * screen_encode->settings.RateControl.TBR.bps * 3;

    if (ecps_backend.share.m_NetworkFlow.Bucket > ecps_backend.share.m_NetworkFlow.BucketSize) {
      ecps_backend.share.m_NetworkFlow.Bucket = ecps_backend.share.m_NetworkFlow.BucketSize;
    }

    auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
    if (flnr == ecps_backend.share.m_NetworkFlow.FrameList.dst) {
      co_return 0;
    }

    size_t queue_depth = ecps_backend.share.m_NetworkFlow.FrameList.Usage();

    if (queue_depth > 3) {
      while (ecps_backend.share.m_NetworkFlow.FrameList.Usage() > 2) {
        auto old_frame = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
        ecps_backend.share.m_NetworkFlow.FrameList.unlrec(old_frame);
      }
      flnr = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
      if (flnr == ecps_backend.share.m_NetworkFlow.FrameList.dst) {
        co_return 0;
      }
    }

    auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];

    uint8_t Flag = 0;
    uint16_t Possible = (f->vec.size() / 0x400) + !!(f->vec.size() % 0x400);
    uint16_t sent_offset = f->SentOffset;

    constexpr size_t MAX_CHUNKS_PER_ITERATION = 8;
    size_t chunks_sent = 0;

    for (; sent_offset < Possible && chunks_sent < MAX_CHUNKS_PER_ITERATION; sent_offset++, chunks_sent++) {
      uintptr_t DataSize = f->vec.size() - sent_offset * 0x400;
      if (DataSize > 0x400) {
        DataSize = 0x400;
      }

      if (ecps_backend.share.m_NetworkFlow.Bucket < DataSize * 8) {
        break;
      }

      bool ret = co_await ecps_backend.write_stream(sent_offset, Possible, Flag, &f->vec[sent_offset * 0x400], DataSize);

      if (ret != false) {
        break;
      }

      ecps_backend.share.m_NetworkFlow.Bucket -= DataSize * 8;
    }

    f->SentOffset = sent_offset;

    if (sent_offset >= Possible) {
      f->vec.clear();
      ecps_backend.share.m_NetworkFlow.FrameList.unlrec(flnr);
      ++ecps_backend.share.frame_index;
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