
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

struct screen_decode_t : fan::graphics::screen_decode_t {};
::screen_decode_t* screen_decode = 0;

struct screen_encode_t : fan::graphics::screen_encode_t {
  std::condition_variable encode_cv;
  std::atomic<uint64_t> decoder_timestamp{ 0 };
  std::atomic<uint64_t> frame_counter{ 0 };
  std::atomic<bool> has_encoded_data{ false };
};
::screen_encode_t* screen_encode = 0;

struct render_thread_t;
render_thread_t* render_thread = 0;

#define ecps_debug_prints 0

#include "backend.h"

// Dynamic config based on framerate
struct dynamic_config_t {
  static uint32_t get_target_framerate() {
    if (screen_encode && screen_encode->settings.InputFrameRate > 0) {
      return screen_encode->settings.InputFrameRate;
    }
    return 60; // Default fallback
  }
  
  static uint32_t get_adaptive_bitrate() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 12000000;      // 12Mbps for 120-144fps
    else if (fps >= 90) return 9000000;   // 9Mbps for 90-119fps  
    else if (fps >= 60) return 6000000;   // 6Mbps for 60-89fps
    else return 4000000;                  // 4Mbps for <60fps
  }
  
  static uint32_t get_adaptive_bucket_size() {
    uint32_t fps = get_target_framerate();
    uint32_t base_bitrate = get_adaptive_bitrate();
    if (fps >= 120) return base_bitrate * 3;     // 3-second burst for high FPS
    else if (fps >= 60) return base_bitrate * 2; // 2-second burst for medium FPS
    else return base_bitrate;                     // 1-second burst for low FPS
  }
  
  static size_t get_adaptive_pool_size() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 256;      // Large pool for 120-144fps
    else if (fps >= 90) return 192;  // Medium-large for 90-119fps
    else if (fps >= 60) return 128;  // Medium for 60-89fps
    else return 64;                  // Small for <60fps
  }
  
  static size_t get_adaptive_queue_size() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 8;        // Larger buffer for high FPS
    else if (fps >= 90) return 6;    // Medium buffer
    else if (fps >= 60) return 4;    // Standard buffer
    else return 3;                   // Small buffer for low FPS
  }
  
  static uint32_t get_adaptive_frame_age_ms() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 150;      // Drop frames faster at high FPS
    else if (fps >= 90) return 200;  // Medium drop time
    else if (fps >= 60) return 300;  // Standard drop time
    else return 500;                 // Longer retention for low FPS
  }
  
  static uint32_t get_adaptive_idr_interval_ms() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 1500;     // More frequent IDR for high FPS
    else if (fps >= 90) return 2000; // Medium frequency
    else if (fps >= 60) return 2500; // Standard frequency
    else return 3000;                // Less frequent for low FPS
  }
  
  static uint32_t get_adaptive_motion_poll_ms() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 30;       // Very fast polling for high FPS
    else if (fps >= 90) return 40;   // Fast polling
    else if (fps >= 60) return 50;   // Medium polling
    else return 80;                  // Slower polling for low FPS
  }
  
  static uint32_t get_adaptive_sleep_us() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 50;       // Minimal sleep for high FPS
    else if (fps >= 90) return 75;   // Short sleep
    else if (fps >= 60) return 100;  // Medium sleep
    else return 200;                 // Longer sleep for low FPS
  }
  
  static size_t get_adaptive_chunk_count() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 40;       // More chunks for high FPS
    else if (fps >= 90) return 30;   // Medium chunks
    else if (fps >= 60) return 25;   // Standard chunks
    else return 20;                  // Fewer chunks for low FPS
  }
  
  static float get_adaptive_bucket_multiplier() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 8.0f;     // Very fast refill for high FPS
    else if (fps >= 90) return 6.5f; // Fast refill
    else if (fps >= 60) return 5.0f; // Medium refill
    else return 3.5f;                // Slower refill for low FPS
  }
};

class FrameMemoryPool {
private:
  static constexpr size_t INITIAL_POOL_SIZE = 64; // Will be adjusted dynamically
  static constexpr size_t FRAME_SIZE = 0x400400;

  struct FrameBuffer {
    alignas(64) std::vector<uint8_t> data;
    std::atomic<bool> in_use{ false };
    std::chrono::steady_clock::time_point last_used;
    FrameBuffer() { data.reserve(FRAME_SIZE); }
  };

  std::vector<std::unique_ptr<FrameBuffer>> buffers;
  std::atomic<size_t> pool_size{ INITIAL_POOL_SIZE };
  mutable std::mutex pool_mutex;
  std::atomic<size_t> allocation_failures{ 0 };

public:
  FrameMemoryPool() {
    size_t adaptive_size = dynamic_config_t::get_adaptive_pool_size();
    buffers.reserve(adaptive_size);
    for (size_t i = 0; i < INITIAL_POOL_SIZE; ++i) {
      buffers.emplace_back(std::make_unique<FrameBuffer>());
    }
  }

  std::shared_ptr<std::vector<uint8_t>> acquire() {
    auto now = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(pool_mutex);
    
    size_t max_pool_size = dynamic_config_t::get_adaptive_pool_size();

    // Try to find an available buffer
    for (auto& buffer : buffers) {
      bool expected = false;
      if (buffer->in_use.compare_exchange_strong(expected, true)) {
        buffer->data.clear();
        buffer->last_used = now;
        return std::shared_ptr<std::vector<uint8_t>>(
          &buffer->data,
          [buffer_ptr = buffer.get()](std::vector<uint8_t>*) {
            buffer_ptr->in_use.store(false);
          }
        );
      }
    }

    // Cleanup old buffers and try to create new one
    auto cutoff_time = now - std::chrono::seconds(5);
    for (auto it = buffers.begin(); it != buffers.end();) {
      if (!(*it)->in_use.load() && (*it)->last_used < cutoff_time && buffers.size() > INITIAL_POOL_SIZE) {
        it = buffers.erase(it);
      } else {
        ++it;
      }
    }

    if (buffers.size() < max_pool_size) {
      auto new_buffer = std::make_unique<FrameBuffer>();
      new_buffer->in_use.store(true);
      new_buffer->last_used = now;
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
    return std::make_shared<std::vector<uint8_t>>();
  }
} frame_pool;

void ecps_backend_t::share_t::CalculateNetworkFlowBucket() {
  uintptr_t MaxBufferSize = (sizeof(ScreenShare_StreamHeader_Head_t) + 0x400) * 8;
  
  // Dynamic bucket size based on framerate
  m_NetworkFlow.BucketSize = dynamic_config_t::get_adaptive_bucket_size();
  m_NetworkFlow.Bucket = m_NetworkFlow.BucketSize;
  
#if ecps_debug_prints >= 1
  uint32_t fps = dynamic_config_t::get_target_framerate();
  fan::print_throttled_format("Adaptive bucket: {}fps -> {} bits ({} MB)", 
    fps, m_NetworkFlow.BucketSize, m_NetworkFlow.BucketSize / 8 / 1024 / 1024);
#endif
}

struct render_thread_t {
  render_thread_t() { engine.set_target_fps(0, false); }

  engine_t engine;
#define engine OFFSETLESS(This, render_thread_t, ecps_gui)->engine
#include "gui.h"
  ecps_gui_t ecps_gui;

  std::condition_variable frame_cv;
  std::condition_variable task_cv;
  std::atomic<bool> has_task_work{ false };
  std::atomic<bool> should_stop{ false };

  fan::graphics::image_t screen_image = engine.image_create();
  fan::graphics::sprite_t local_frame{ {
    .position = fan::vec3(fan::vec2(0), 1),
    .size = gloco->window.get_size() / 2,
  } };
  fan::graphics::universal_image_renderer_t network_frame{ {
    .position = fan::vec3(fan::vec2(0), 1),
    .size = gloco->window.get_size() / 2,
  } };

  void render(auto l) {
    engine.process_loop([this, l] { ecps_gui.render(); l(); });
  }

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix FrameList
#define BLL_set_Language 1
#define BLL_set_Usage 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeDataType fan::graphics::screen_decode_t::decode_data_t
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>
  FrameList_t FrameList;
};

ecps_backend_t::ecps_backend_t() {
  __dme_get(Protocol_S2C, KeepAlive) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    backend.tcp_keep_alive.reset();
    co_return;
  };

  __dme_get(Protocol_S2C, InformInvalidIdentify) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::InformInvalidIdentify_t>();
    if (msg->ClientIdentify != identify_secret) {
      co_return;
    }
    identify_secret = msg->ServerIdentify;
    co_await backend.udp_write(0, ProtocolUDP::C2S_t::KeepAlive, {}, 0, 0); // set dst addr
    co_return;
  };

  __dme_get(Protocol_S2C, Response_Login) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Response_Login_t>();
    backend.session_id = msg->SessionID;
    co_await backend.udp_write(0, ProtocolUDP::C2S_t::KeepAlive, {}, 0, 0); // identify 
  };

  __dme_get(Protocol_S2C, CreateChannel_OK) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::CreateChannel_OK_t>();
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
    channel_info.front().session_id = msg->ChannelSessionID;
    if (screen_encode) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
    }
  };

  __dme_get(Protocol_S2C, JoinChannel_Error) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_Error_t>();
  };

  __dme_get(Protocol_S2C, Channel_ScreenShare_ViewToShare) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_ViewToShare_t>();
    if (render_thread->ecps_gui.is_streaming == false) {
      co_return;
    }
    if (msg->Flag & ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
    }
  };

  __dme_get(Protocol_S2C, ChannelList) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::ChannelList_t>();
    backend.available_channels.clear();
    backend.available_channels.reserve(msg->ChannelCount);

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
    }
    backend.channel_list_received = true;
  };

  __dme_get(Protocol_S2C, ChannelSessionList) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::ChannelSessionList_t>();
    Protocol_ChannelID_t channel_id = msg->ChannelID;
    backend.channel_sessions[channel_id.i].clear();
    backend.channel_sessions[channel_id.i].reserve(msg->SessionCount);

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
    }
  };
}

fan::event::task_t ecps_backend_t::default_s2c_cb(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
  co_await backend.tcp_client.read(backend.Protocol_S2C.NA(base.Command)->m_DSS);
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

// Dynamic queue size based on framerate
size_t get_max_decode_queue_size() {
  return dynamic_config_t::get_adaptive_queue_size();
}

void ecps_backend_t::view_t::WriteFramePacket() {
  if (m_Possible == 0) return;

  uint32_t FramePacketSize = (uint32_t)(this->m_Possible - 1) * 0x400 + this->m_ModuloSize;
  if (FramePacketSize == 0 || FramePacketSize > 0x500000 || this->m_data.size() < FramePacketSize) {
    this->m_stats.Frame_Drop++;
    return;
  }

#if ecps_debug_prints >= 2
  printf("NETWORK: Frame received at %lld ms\n",
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
#endif

  std::vector<uint8_t> frame_data(this->m_data.begin(), this->m_data.begin() + FramePacketSize);

  // Detect important frames (keyframes/SPS/PPS)
  bool has_sps_pps = false;
  for (size_t i = 0; i < std::min(frame_data.size() - 4, size_t(32)); ++i) {
    if (frame_data[i] == 0x00 && frame_data[i + 1] == 0x00 && frame_data[i + 2] == 0x00 && frame_data[i + 3] == 0x01) {
      uint8_t nal_type = frame_data[i + 4] & 0x1F;
      if (nal_type == 7 || nal_type == 8) {
        has_sps_pps = true;
        break;
      }
    }
  }

  bool is_important = has_sps_pps || (FramePacketSize > 30000);

  // Enhanced startup detection
  static auto stream_start_time = std::chrono::steady_clock::now();
  static bool was_streaming = false;
  bool currently_streaming = render_thread && render_thread->ecps_gui.is_streaming;

  if (currently_streaming && !was_streaming) {
    stream_start_time = std::chrono::steady_clock::now();
  }
  was_streaming = currently_streaming;

  auto now = std::chrono::steady_clock::now();
  auto time_since_stream_start = std::chrono::duration_cast<std::chrono::seconds>(now - stream_start_time);
  bool is_startup_phase = (time_since_stream_start.count() < 10);

  {
    std::unique_lock<std::timed_mutex> lock(network_decode_mutex, std::defer_lock);
    auto lock_timeout = is_startup_phase ? std::chrono::milliseconds(100) : 
                       std::chrono::milliseconds(is_important ? 50 : 10);

    if (!lock.try_lock_for(lock_timeout)) {
      if (!is_startup_phase) {
        this->m_stats.Frame_Drop++;
      }
      return;
    }

    size_t max_queue_size = get_max_decode_queue_size();
    while (network_decode_queue.size() >= max_queue_size) {
      if (is_important || is_startup_phase) {
        network_decode_queue.pop();
      } else {
        this->m_stats.Frame_Drop++;
        return;
      }
    }

    auto data_copy = frame_pool.acquire();
    data_copy->resize(FramePacketSize);
    std::memcpy(data_copy->data(), frame_data.data(), FramePacketSize);

    frame_timing_t frame_data_timing;
    frame_data_timing.encode_time = std::chrono::steady_clock::now();
    frame_data_timing.frame_id = frame_index++;
    frame_data_timing.data = data_copy;
    frame_data_timing.source = frame_timing_t::NETWORK;

    network_decode_queue.push(std::move(frame_data_timing));
  }

  render_thread->frame_cv.notify_one();
}

void configure_encoder_for_artifact_reduction() {
  if (screen_encode) {
    // Dynamic bitrate based on framerate
    uint32_t adaptive_bitrate = dynamic_config_t::get_adaptive_bitrate();
    screen_encode->settings.RateControl.VBR.bps = std::max(screen_encode->settings.RateControl.VBR.bps, adaptive_bitrate);
    
    // Optimize for detected framerate
    uint32_t target_fps = dynamic_config_t::get_target_framerate();
    screen_encode->settings.InputFrameRate = target_fps;
    
    screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
#if ecps_debug_prints >= 1
    fan::print_throttled_format("Encoder configured adaptively: {}fps, {} bps", 
      target_fps, adaptive_bitrate);
#endif
  }
}

bool is_local_stream() {
  return render_thread && render_thread->ecps_gui.is_streaming;
}

static auto request_idr_smart = [](const std::string& reason = "") {
  if (is_local_stream()) {
    if (screen_encode) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
    }
  } else {
    try {
      render_thread->ecps_gui.backend_queue([=]() -> fan::event::task_t {
        try {
          ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
          rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
          rest.ChannelID = ecps_backend.channel_info.front().channel_id;
          co_await ecps_backend.tcp_write(
            ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
            &rest,
            sizeof(rest)
          );
        } catch (...) {}
      });
    } catch (...) {}
  }
};

int main() {
  std::promise<void> render_thread_promise;
  std::future<void> render_thread_future = render_thread_promise.get_future();

  // Encode thread
  auto encode_thread_id = fan::event::thread_create([] {
    screen_encode = (::screen_encode_t*)malloc(sizeof(::screen_encode_t));
    std::construct_at(screen_encode);
    configure_encoder_for_artifact_reduction();

  #if ecps_debug_prints >= 1
    fan::print_throttled("Encode thread started, waiting for dependencies...");
  #endif

    while (screen_decode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

  #if ecps_debug_prints >= 1
    fan::print_throttled("Encode thread dependencies ready, starting main loop");
  #endif

    uint64_t frame_id = 0;
    bool first_frame = true;
    auto last_idr_time = std::chrono::steady_clock::now();

    while (1) {
      if (render_thread->ecps_gui.is_streaming) {
  #if ecps_debug_prints >= 2
        static bool streaming_logged = false;
        if (!streaming_logged) {
          fan::print_throttled("Streaming is active, starting encoding");
          streaming_logged = true;
        }
  #endif

        if (first_frame) {
          screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
          first_frame = false;
          last_idr_time = std::chrono::steady_clock::now();
  #if ecps_debug_prints >= 1
          fan::print_throttled("Encode thread: First frame setup complete");
  #endif
        }

        auto now = std::chrono::steady_clock::now();
        auto time_since_idr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_idr_time);

        // Dynamic IDR timing based on framerate
        uint32_t adaptive_idr_interval = dynamic_config_t::get_adaptive_idr_interval_ms();
        if (time_since_idr > std::chrono::milliseconds(adaptive_idr_interval)) {
          screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
          last_idr_time = now;
  #if ecps_debug_prints >= 2
          printf("ENCODE: Frame encoded at %lld ms\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count());

          fan::print_throttled_format("Encode thread: Forced IDR ({}ms interval for {}fps)", 
            adaptive_idr_interval, dynamic_config_t::get_target_framerate());
  #endif
        }

        if (!screen_encode->screen_read()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }

  #if ecps_debug_prints >= 2
        static uint64_t screen_read_count = 0;
        if (++screen_read_count % 60 == 1) {
          fan::print_throttled_format("Encode thread: screen_read successful (frame {})", screen_read_count);
        }
  #endif

        // Local stream
        if (render_thread->ecps_gui.show_own_stream && screen_encode->screen_buffer) {
          screen_decode->graphics_queue_callback([=]() {
            try {
              uint32_t width = screen_encode->mdscr.Geometry.Resolution.x;
              uint32_t height = screen_encode->mdscr.Geometry.Resolution.y;
            
              if (width > 0 && height > 0 && screen_encode->screen_buffer) {
                fan::graphics::image_load_properties_t props;
                props.format = fan::graphics::image_format::b8g8r8a8_unorm;
                props.visual_output = fan::graphics::image_filter::linear;

                fan::image::info_t image_info;
                image_info.data = screen_encode->screen_buffer;
                image_info.size = fan::vec2ui(width, height);
              
                render_thread->engine.image_reload(render_thread->screen_image, image_info, props);
                render_thread->local_frame.set_image(render_thread->screen_image);
  #if ecps_debug_prints >= 2
                static uint64_t local_update_count = 0;
                if (++local_update_count % 30 == 1) {
                  fan::print_throttled_format("LOCAL: Direct sprite updated {}x{} (frame {})", width, height, local_update_count);
                }
  #endif
              }
            } catch (const std::exception& e) {
  #if ecps_debug_prints >= 1
              fan::print_throttled_format("LOCAL: Direct sprite update failed: {}", e.what());
  #endif
            }
          });
        }

        if (!screen_encode->encode_write()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }

  #if ecps_debug_prints >= 2
        static uint64_t encode_write_count = 0;
        if (++encode_write_count % 30 == 1) {
          fan::print_throttled_format("Encode thread: encode_write successful (frame {})", encode_write_count);
        }
  #endif

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

          if (encoded_count == 0) continue;

  #if ecps_debug_prints >= 2
          static uint64_t encoded_frame_count = 0;
          if (++encoded_frame_count % 30 == 1) {
            fan::print_throttled_format("Encode thread: Encoded {} bytes (frame {})", encoded_count, encoded_frame_count);
          }
  #endif

          // Network transmission
          {
            std::unique_lock<std::timed_mutex> frame_list_lock(ecps_backend.share.frame_list_mutex, std::try_to_lock);
            if (frame_list_lock.owns_lock()) {
              bool is_idr = (screen_encode->encode_write_flags & fan::graphics::codec_update_e::reset_IDR) != 0;
              if (is_idr) {
                while (ecps_backend.share.m_NetworkFlow.FrameList.Usage() > 0) {
                  auto old = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
                  ecps_backend.share.m_NetworkFlow.FrameList.unlrec(old);
                }
  #if ecps_debug_prints >= 2
                fan::print_throttled("Encode thread: Cleared frame list for IDR");
  #endif
              }

              auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.NewNodeLast();
              auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];
              f->vec = *encoded_buffer;
              f->SentOffset = 0;
  #if ecps_debug_prints >= 2
              static uint64_t queued_frame_count = 0;
              if (++queued_frame_count % 30 == 1) {
                fan::print_throttled_format("Encode thread: Queued frame for network transmission ({})", queued_frame_count);
              }
  #endif
            } else {
  #if ecps_debug_prints >= 2
              fan::print_throttled("Encode thread: Could not acquire frame list lock");
  #endif
            }
          }

          screen_encode->has_encoded_data.store(true);
        }

        decode_queue_cv.notify_one();
        screen_encode->encode_cv.notify_one();
        screen_encode->frame_counter.fetch_add(1);
        screen_encode->sleep_thread();
      } else {
  #if ecps_debug_prints >= 2
        static bool not_streaming_logged = false;
        if (!not_streaming_logged) {
          fan::print_throttled("Encode thread: Waiting for streaming to start...");
          not_streaming_logged = true;
        }
  #endif
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  });

  // Decode thread
  auto decode_thread_id = fan::event::thread_create([] {
    screen_decode = (::screen_decode_t*)malloc(sizeof(::screen_decode_t));
    std::construct_at(screen_decode);

#if ecps_debug_prints >= 1
    fan::print_throttled("Decode thread started, waiting for dependencies...");
#endif

    while (screen_encode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

#if ecps_debug_prints >= 1
    fan::print_throttled("Decode thread dependencies ready - NETWORK ONLY (local uses direct sprite)");
#endif

    auto last_network_frame_time = std::chrono::steady_clock::now();
    auto startup_time = std::chrono::steady_clock::now();
    auto last_successful_decode = std::chrono::steady_clock::now(); // Start fresh
    auto MAX_FRAME_AGE = std::chrono::milliseconds(dynamic_config_t::get_adaptive_frame_age_ms());
    uint64_t consecutive_decode_failures = 0;
    uint64_t successful_decodes = 0;
    bool decoder_needs_reset = false;
    bool has_processed_any_frame = false; // Track if we've ever processed frames

    while (1) {
      bool processed_frame = false;
      auto now = std::chrono::steady_clock::now();
      auto time_since_success = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_successful_decode);

      // Only check for reset if we've actually tried to process frames and have failures
      // Don't reset just because we haven't received any frames yet
      if (has_processed_any_frame && 
          (consecutive_decode_failures > 30 || 
           (time_since_success > std::chrono::milliseconds(10000) && successful_decodes > 0))) {
        decoder_needs_reset = true;
#if ecps_debug_prints >= 1
        fan::print_throttled_format("Decode thread: Decoder reset needed (failures: {}, time since success: {}ms, processed frames: {})", 
          consecutive_decode_failures, time_since_success.count(), has_processed_any_frame);
#endif
      }

      // Process network streams only
      bool should_process_network = !render_thread->ecps_gui.show_own_stream || 
                                   !render_thread->ecps_gui.is_streaming;

      if (should_process_network) {
        std::unique_lock<std::timed_mutex> network_lock(network_decode_mutex, std::try_to_lock);
        if (network_lock.owns_lock() && !network_decode_queue.empty()) {
#if ecps_debug_prints >= 2
          printf("DECODE: Frame decoded at %lld ms\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count());

          static uint64_t queue_check_count = 0;
          if (++queue_check_count % 100 == 1) {
            fan::print_throttled_format("Decode thread: Processing network queue (size: {})", network_decode_queue.size());
          }
#endif

          frame_timing_t frame_to_decode;
          bool found_valid_frame = false;

          while (!network_decode_queue.empty()) {
            auto frame = network_decode_queue.front();
            network_decode_queue.pop();
            auto age = now - frame.encode_time;
            if (age < MAX_FRAME_AGE) {
              frame_to_decode = std::move(frame);
              found_valid_frame = true;
              break;
            }
          }

          network_lock.unlock();

          if (found_valid_frame) {
            has_processed_any_frame = true; // Mark that we're now processing frames
            frame_to_decode.decode_time = now;
            last_network_frame_time = now;

#if ecps_debug_prints >= 2
            static uint64_t decode_attempt_count = 0;
            if (++decode_attempt_count % 30 == 1) {
              fan::print_throttled_format("NETWORK: Attempting decode (frame {}, size: {})", 
                decode_attempt_count, frame_to_decode.data ? frame_to_decode.data->size() : 0);
            }
#endif

            if (frame_to_decode.data && !frame_to_decode.data->empty()) {
              if (decoder_needs_reset) {
                try {
#if ecps_debug_prints >= 1
                  fan::print_throttled("Decode thread: Resetting decoder state for NETWORK stream...");
#endif
                  screen_decode->close_decoder();
                  std::this_thread::sleep_for(std::chrono::milliseconds(50));
                  screen_decode->open_decoder();

                  std::lock_guard<std::timed_mutex> frame_lock(render_mutex);
                  while (render_thread->FrameList.Usage() > 0) {
                    auto old = render_thread->FrameList.GetNodeFirst();
                    render_thread->FrameList.unlrec(old);
                  }

                  decoder_needs_reset = false;
                  consecutive_decode_failures = 0;
                  last_successful_decode = now; // Reset timer after decoder reset
                } catch (...) {
#if ecps_debug_prints >= 1
                  fan::print_throttled("Decode thread: Decoder reset failed");
#endif
                }
              }

              try {
                fan::graphics::screen_decode_t::decode_data_t decode_data = screen_decode->decode(
                  frame_to_decode.data->data(),
                  frame_to_decode.data->size(),
                  render_thread->network_frame
                );

                bool decode_successful = false;
                if (decode_data.type == 0) {
                  decode_successful = true;
                } else if (decode_data.type == 1) {
                  bool has_y_data = !decode_data.data[0].empty();
                  bool has_u_data = !decode_data.data[1].empty();
                  bool has_v_data = !decode_data.data[2].empty();
                  bool valid_dimensions = decode_data.image_size.x > 0 && decode_data.image_size.y > 0;
                  bool valid_stride = decode_data.stride[0].x >= decode_data.image_size.x;
                  decode_successful = has_y_data && has_u_data && has_v_data && valid_dimensions && valid_stride;
                }

                if (decode_successful) {
                  std::lock_guard<std::timed_mutex> frame_lock(render_mutex);
                  while (render_thread->FrameList.Usage() > 0) {
                    auto old = render_thread->FrameList.GetNodeFirst();
                    render_thread->FrameList.unlrec(old);
                  }

                  auto flnr = render_thread->FrameList.NewNodeLast();
                  auto f = &render_thread->FrameList[flnr];
                  *f = std::move(decode_data);

                  processed_frame = true;
                  consecutive_decode_failures = 0;
                  successful_decodes++;
                  last_successful_decode = now;

#if ecps_debug_prints >= 2
                  static uint64_t successful_decode_count = 0;
                  if (++successful_decode_count % 30 == 1) {
                    fan::print_throttled_format("NETWORK: Frame decoded successfully (type: {}, total: {})", 
                      decode_data.type, successful_decode_count);
                  }
#endif
                } else {
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("NETWORK: Decode validation failed (type: {})", decode_data.type);
#endif
                  consecutive_decode_failures++;
                }
              } catch (const std::exception& e) {
                consecutive_decode_failures++;
#if ecps_debug_prints >= 1
                fan::print_throttled_format("NETWORK: Decode exception: {}", e.what());
#endif
              }
            }
          }
        }
      }

      if (!processed_frame) {
        if (!has_processed_any_frame) {
          // Haven't started processing frames yet, sleep longer
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } else {
          // Dynamic sleep based on framerate
          uint32_t adaptive_sleep = dynamic_config_t::get_adaptive_sleep_us();
          std::this_thread::sleep_for(std::chrono::microseconds(adaptive_sleep));
        }
      }
    }
  });

  // Based on framerate
  auto frequent_idr_task = fan::event::task_timer(dynamic_config_t::get_adaptive_idr_interval_ms(), []() -> fan::event::task_value_resume_t<bool> {
    static auto startup_time = std::chrono::steady_clock::now();
    static bool startup_phase = true;

    auto now = std::chrono::steady_clock::now();
    auto time_since_startup = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);
    uint32_t fps = dynamic_config_t::get_target_framerate();
    uint32_t startup_duration = fps >= 120 ? 4 : (fps >= 60 ? 6 : 8); // Shorter startup for high FPS

    if (render_thread && render_thread->ecps_gui.is_streaming && screen_encode) {
      if (startup_phase && time_since_startup.count() < startup_duration) {
        static auto last_startup_idr = startup_time;
        uint32_t startup_idr_interval = dynamic_config_t::get_adaptive_idr_interval_ms() / 2; // Half interval during startup
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_startup_idr).count() > startup_idr_interval) {
          screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
          last_startup_idr = now;
        }
      } else {
        startup_phase = false;
        screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
      }
    }

    if (!render_thread || !render_thread->ecps_gui.is_streaming) {
      startup_time = now;
      startup_phase = true;
    }

    co_return 0;
  });

  auto motion_idr_task = fan::event::task_timer(dynamic_config_t::get_adaptive_motion_poll_ms(), []() -> fan::event::task_value_resume_t<bool> {
    static uint64_t last_frame_count = 0;
    static int motion_frames = 0;
    static auto startup_time = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    auto time_since_startup = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);
    uint32_t fps = dynamic_config_t::get_target_framerate();
    uint32_t startup_duration = fps >= 120 ? 4 : (fps >= 60 ? 6 : 8);

    if (render_thread && render_thread->ecps_gui.is_streaming && screen_encode) {
      uint64_t current_frame_count = screen_encode->frame_counter.load();
      uint32_t poll_ms = dynamic_config_t::get_adaptive_motion_poll_ms();
      uint64_t frames_in_poll = current_frame_count - last_frame_count;

      if (time_since_startup.count() > startup_duration) {
        // Calculate expected frames for motion detection (poll_ms * fps / 1000 * 1.5 for motion threshold)
        uint32_t motion_threshold = (poll_ms * fps / 1000) * 3 / 2;
        if (frames_in_poll > motion_threshold) {
          motion_frames++;
          uint32_t motion_trigger = fps >= 120 ? 8 : (fps >= 60 ? 6 : 4); // More frames needed for higher FPS
          if (motion_frames >= motion_trigger) {
            screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
            motion_frames = 0;
          }
        } else {
          motion_frames = 0;
        }
      }
      last_frame_count = current_frame_count;
    } else {
      startup_time = now;
    }

    co_return 0;
  });

  // Render thread
  auto render_thread_id = fan::event::thread_create([&render_thread_promise] {
    render_thread = (render_thread_t*)malloc(sizeof(render_thread_t));
    std::construct_at(render_thread);
    render_thread_promise.set_value();

    while (!render_thread->engine.should_close() && !render_thread->should_stop.load()) {
      {
        std::lock_guard<std::timed_mutex> render_lock(render_mutex);      
        {
          std::lock_guard<std::mutex> lock(screen_decode->mutex);
          for (auto& i : screen_decode->graphics_queue) {
            i();
          }
          screen_decode->graphics_queue.clear();
        }

        if (render_thread->ecps_gui.show_own_stream == false) {
          render_thread->local_frame.set_image(render_thread->engine.default_texture);
          auto flnr = render_thread->FrameList.GetNodeFirst();
          if (flnr != render_thread->FrameList.dst) {
            auto& node = render_thread->FrameList[flnr];
            bool frame_valid = false;
#if ecps_debug_prints >= 2
            printf("RENDER: Frame displayed at %lld ms\n",
              std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
#endif

            if (node.type == 0) { // CUVID
              try {
                screen_decode->decode_cuvid(render_thread->network_frame);
                frame_valid = true;
#if ecps_debug_prints >= 2
                static uint64_t cuvid_render_count = 0;
                if (++cuvid_render_count % 30 == 1) {
                  fan::print_throttled_format("RENDER: CUVID frame processed for network stream ({})", cuvid_render_count);
                }
#endif
              } catch (const std::exception& e) {
#if ecps_debug_prints >= 1
                fan::print_throttled_format("RENDER: CUVID decode failed: {}", e.what());
#endif
              }
            } else if (node.type == 1) { // YUV
              bool y_valid = !node.data[0].empty();
              bool u_valid = !node.data[1].empty();
              bool v_valid = !node.data[2].empty();
              bool dims_valid = node.image_size.x > 0 && node.image_size.y > 0 &&
                node.image_size.x <= 7680 && node.image_size.y <= 4320;
              bool stride_valid = node.stride[0].x >= node.image_size.x && node.stride[0].x > 0;

              if (y_valid && u_valid && v_valid && dims_valid && stride_valid) {
                f32_t sx = (f32_t)node.image_size.x / node.stride[0].x;
                if (sx > 0 && sx <= 1.0f) {
                  std::array<void*, 4> raw_ptrs;
                  for (size_t i = 0; i < 4; ++i) {
                    raw_ptrs[i] = static_cast<void*>(node.data[i].data());
                  }

                  if (raw_ptrs[0] && raw_ptrs[1] && raw_ptrs[2]) {
                    try {
                      render_thread->network_frame.set_tc_size(fan::vec2(sx, 1));
                      render_thread->network_frame.reload(
                        node.pixel_format,
                        raw_ptrs.data(),
                        fan::vec2ui(node.stride[0].x, node.image_size.y)
                      );
                      frame_valid = true;
#if ecps_debug_prints >= 2
                      static uint64_t yuv_render_count = 0;
                      if (++yuv_render_count % 30 == 1) {
                        fan::print_throttled_format("RENDER: YUV network frame processed {}x{}, stride={}, sx={} ({})",
                          node.image_size.x, node.image_size.y, node.stride[0].x, sx, yuv_render_count);
                      }
#endif
                    } catch (const std::exception& e) {
#if ecps_debug_prints >= 1
                      fan::print_throttled_format("RENDER: YUV reload failed for network: {}", e.what());
#endif
                    }
                  }
                } else {
#if ecps_debug_prints >= 1
                  fan::print_throttled_format("RENDER: Invalid sx ratio for network: {}", sx);
#endif
                }
              } else {
#if ecps_debug_prints >= 1
                fan::print_throttled_format("RENDER: YUV validation failed for network: Y={}, U={}, V={}, dims={}x{}, stride={}",
                  y_valid, u_valid, v_valid, node.image_size.x, node.image_size.y, node.stride[0].x);
#endif
              }
            }

            render_thread->FrameList.unlrec(flnr);

            if (!frame_valid) {
              static auto last_idr_request = std::chrono::steady_clock::now();
              auto now = std::chrono::steady_clock::now();
              if (std::chrono::duration_cast<std::chrono::seconds>(now - last_idr_request).count() > 3) {
                render_thread->ecps_gui.backend_queue([=]() -> fan::event::task_t {
                  try {
                    if (ecps_backend.channel_info.size()) {
                      ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
                      rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
                      rest.ChannelID = ecps_backend.channel_info.front().channel_id;
                      co_await ecps_backend.tcp_write(
                        ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
                        &rest,
                        sizeof(rest)
                      );
#if ecps_debug_prints >= 1
                      fan::print_throttled("RENDER: Requesting network IDR due to invalid frame");
#endif
                    }
                  } catch (...) {}
                });
                last_idr_request = now;
              }
            }
          }
        }
      }

      render_thread->render([] {
        static const char* sampler_names[] = {
          "Nearest", "Linear", "Nearest Mipmap Nearest",
          "Linear Mipmap Nearest", "Nearest Mipmap Linear", "Linear Mipmap Linear"
        };

        static fan::graphics::image_filter sampler_filters[] = {
          fan::graphics::image_filter::nearest,
          fan::graphics::image_filter::linear,
          fan::graphics::image_filter::nearest_mipmap_nearest,
          fan::graphics::image_filter::linear_mipmap_nearest,
          fan::graphics::image_filter::nearest_mipmap_linear,
          fan::graphics::image_filter::linear_mipmap_linear
        };

        static int current_sampler = fan::graphics::image_filter::linear;

        std::vector<fan::graphics::image_t> img_list;
        img_list.emplace_back(render_thread->local_frame.get_image());
        img_list.emplace_back(render_thread->network_frame.get_image());
        auto images = render_thread->network_frame.get_images();
        img_list.insert(img_list.end(), images.begin(), images.end());
        
        static bool p_open_debug = false;
        if (fan::window::is_key_down(fan::key_shift) && fan::window::is_key_pressed(fan::key_5)) {
          p_open_debug = !p_open_debug;
        }
        
        if (p_open_debug) {
          gui::begin("debug");
          
          if (gui::button("get list")) {
            render_thread->ecps_gui.backend_queue([=]() -> fan::event::task_t {
              try {
                co_await ecps_backend.request_channel_list();
              } catch (...) {}
            });
          }

          gui::checkbox("show own stream", &render_thread->ecps_gui.show_own_stream);
          
          if (gui::combo("Sampler", &current_sampler, sampler_names, std::size(sampler_names))) {
            auto selected_filter = sampler_filters[current_sampler];
            for (auto& i : img_list) {
              if (!render_thread->engine.is_image_valid(i)) continue;
              fan::graphics::image_data_t& image_data = render_thread->engine.image_list[i];
              image_data.image_settings.min_filter = selected_filter;
              if (selected_filter == fan::graphics::image_filter::linear ||
                  selected_filter == fan::graphics::image_filter::nearest) {
                image_data.image_settings.mag_filter = selected_filter;
              } else {
                image_data.image_settings.mag_filter = fan::graphics::image_filter::linear;
              }
              render_thread->engine.image_set_settings(i, image_data.image_settings);
            }
          }
          gui::end();
        }
      });

      std::this_thread::yield();
    }

    delete render_thread;
  });

  render_thread_future.wait();

  fan::event::task_idle([]() -> fan::event::task_t {
    if (render_thread == nullptr || render_thread->should_stop.load()) {
      co_return;
    }

    constexpr size_t MAX_BATCH_SIZE = 4;
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

  auto enhanced_network_task = fan::event::task_timer(1, []() -> fan::event::task_value_resume_t<bool> {
    if (ecps_backend.channel_info.empty()) {
      co_return 0;
    }

    std::unique_lock<std::timed_mutex> frame_list_lock(ecps_backend.share.frame_list_mutex, std::try_to_lock);
    if (!frame_list_lock.owns_lock()) {
      co_return 0;
    }

    uint64_t ctime = fan::event::now();
    uint64_t DeltaTime = ctime - ecps_backend.share.m_NetworkFlow.TimerLastCallAt;
    ecps_backend.share.m_NetworkFlow.TimerLastCallAt = ctime;

    // Dynamic bucket refill based on framerate
    float bucket_multiplier = dynamic_config_t::get_adaptive_bucket_multiplier();
    ecps_backend.share.m_NetworkFlow.Bucket +=
      (f32_t)DeltaTime / 1000000000 * screen_encode->settings.RateControl.VBR.bps * bucket_multiplier;

    if (ecps_backend.share.m_NetworkFlow.Bucket > ecps_backend.share.m_NetworkFlow.BucketSize) {
      ecps_backend.share.m_NetworkFlow.Bucket = ecps_backend.share.m_NetworkFlow.BucketSize;
    }

    auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
    if (flnr == ecps_backend.share.m_NetworkFlow.FrameList.dst) {
      co_return 0;
    }

    auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];
    bool is_likely_keyframe = f->vec.size() > 30000;
    uint8_t Flag = 0;
    uint16_t Possible = (f->vec.size() / 0x400) + !!(f->vec.size() % 0x400);
    uint16_t sent_offset = f->SentOffset;

    // Dynamic chunk count based on framerate
    size_t max_chunks = dynamic_config_t::get_adaptive_chunk_count();
    size_t chunks_to_send = is_likely_keyframe ? max_chunks : (max_chunks * 2 / 3); // 2/3 for regular frames
    size_t chunks_sent = 0;

    for (; sent_offset < Possible && chunks_sent < chunks_to_send; sent_offset++, chunks_sent++) {
      uintptr_t DataSize = f->vec.size() - sent_offset * 0x400;
      if (DataSize > 0x400) {
        DataSize = 0x400;
      }

      if (!is_likely_keyframe && ecps_backend.share.m_NetworkFlow.Bucket < DataSize * 8) {
        break;
      }

      bool ret = co_await ecps_backend.write_stream(sent_offset, Possible, Flag, &f->vec[sent_offset * 0x400], DataSize);
      if (ret != false) {
        break;
      }

      if (ecps_backend.share.m_NetworkFlow.Bucket >= DataSize * 8) {
        ecps_backend.share.m_NetworkFlow.Bucket -= DataSize * 8;
      } else if (is_likely_keyframe) {
        ecps_backend.share.m_NetworkFlow.Bucket -= DataSize * 8;
      }
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