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

#define ecps_debug_prints 0

#include "backend.h"

class FrameMemoryPool {
private:
  static constexpr size_t INITIAL_POOL_SIZE = 32;
  static constexpr size_t MAX_POOL_SIZE = 128;
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
    auto now = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(pool_mutex);

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

    auto cutoff_time = now - std::chrono::seconds(5);
    for (auto it = buffers.begin(); it != buffers.end();) {
      if (!(*it)->in_use.load() && (*it)->last_used < cutoff_time && buffers.size() > INITIAL_POOL_SIZE) {
        it = buffers.erase(it);
      }
      else {
        ++it;
      }
    }

    if (buffers.size() < MAX_POOL_SIZE) {
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
    if (allocation_failures % 50 == 1) { // Reduced logging frequency
#if ecps_debug_prints >= 2
      fan::print_throttled("Warning: Frame pool exhausted, creating temporary buffer");
#endif
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

void ecps_backend_t::share_t::CalculateNetworkFlowBucket() {
  uintptr_t MaxBufferSize = (sizeof(ScreenShare_StreamHeader_Head_t) + 0x400) * 8;

  // MUCH larger bucket for keyframes - they can be 100KB+
  m_NetworkFlow.BucketSize = screen_encode->settings.RateControl.VBR.bps; // Full 1-second burst capability

  // Minimum bucket size for large keyframes
  if (m_NetworkFlow.BucketSize < 10000000) { // 10 Mbps minimum
    m_NetworkFlow.BucketSize = 10000000;
  }

  // Start with full bucket to handle initial keyframes
  m_NetworkFlow.Bucket = m_NetworkFlow.BucketSize;

#if ecps_debug_prints >= 2
  fan::print_throttled_format("Enhanced bucket: size={} bits ({} MB)",
    m_NetworkFlow.BucketSize, m_NetworkFlow.BucketSize / 8 / 1024 / 1024);
#endif
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


ecps_backend_t::ecps_backend_t() {
  __dme_get(Protocol_S2C, KeepAlive) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
#if ecps_debug_prints >= 1
    fan::print_throttled("tcp keep alive came");
#endif
    backend.tcp_keep_alive.reset();
    co_return;
    };
  __dme_get(Protocol_S2C, InformInvalidIdentify) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::InformInvalidIdentify_t>();
    if (msg->ClientIdentify != identify_secret) {
      co_return;
    }
    identify_secret = msg->ServerIdentify;
#if ecps_debug_prints >= 1
    fan::print_throttled("inform invalid identify came");
#endif
    co_return;
    };

  __dme_get(Protocol_S2C, Response_Login) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Response_Login_t>();
#if ecps_debug_prints >= 1
    fan::print_throttled_format(R"({{
[SERVER] Response_login
SessionID: {}
AccountID: {}
}})", msg->SessionID.i, msg->AccountID.i);
#endif
    backend.session_id = msg->SessionID;
    };
  __dme_get(Protocol_S2C, CreateChannel_OK) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::CreateChannel_OK_t>();
#if ecps_debug_prints >= 1
    fan::print_throttled_format(R"({{
[SERVER] CreateChannel_OK
ID: {}
ChannelID: {}
}})", base.ID, msg->ChannelID.i);
#endif
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
#if ecps_debug_prints >= 1
    fan::print_throttled_format(R"({{
[SERVER] JoinChannel_OK
ID: {}
ChannelID: {}
}})", base.ID, msg->ChannelID.i);
#endif
    channel_info.front().session_id = msg->ChannelSessionID;

    // ADDED: Request IDR frame immediately on join
    if (screen_encode) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
    }
    };
  __dme_get(Protocol_S2C, JoinChannel_Error) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_Error_t>();
#if ecps_debug_prints >= 1
    fan::print_throttled_format(R"({{
[SERVER] JoinChannel_Error
ID: {}
ChannelID: {}
}})", base.ID, Protocol::JoinChannel_Error_Reason_String[(uint8_t)msg->Reason]);
#endif
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
    //  
    //  fan::print_throttled_format(R"({{
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
      //fan::print_throttled_format(R"(  Channel {}: "{}" ({} users, host: {}))", 
      //  info.channel_id.i, info.name, info.user_count, info.host_session_id.i);
    }

    backend.channel_list_received = true;
    };

  __dme_get(Protocol_S2C, ChannelSessionList) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::ChannelSessionList_t>();

    //  fan::print_throttled_format(R"({{
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

      //fan::print_throttled_format(R"(  User {}: "{}" (session: {}, account: {}, {}))", 
      //  i + 1, info.username, info.session_id.i, info.account_id.i, 
      //  info.is_host ? "HOST" : "viewer");
    }
    };
}

fan::event::task_t ecps_backend_t::default_s2c_cb(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
  co_await backend.tcp_client.read(backend.Protocol_S2C.NA(base.Command)->m_DSS); // advance tcp data
  //  fan::print_throttled("unhandled callback");
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
const size_t MAX_DECODE_QUEUE_SIZE = 3;

#if ecps_debug_prints >= 2
bool validate_video_packet(const std::vector<uint8_t>& data, size_t expected_size) {
  if (data.empty() || data.size() < 8) {
    return false;
  }

  // More lenient size checking for variable bitrate content
  if (data.size() < expected_size / 20 || data.size() > expected_size * 5) {
    fan::print_throttled_format("Frame size out of bounds: {} (expected ~{})", data.size(), expected_size);
    return false;
  }

  // Check for valid H.264 NAL unit headers
  bool has_valid_nal = false;
  for (size_t i = 0; i < std::min(data.size() - 4, size_t(32)); ++i) {
    if (data[i] == 0x00 && data[i + 1] == 0x00 &&
      (data[i + 2] == 0x01 || (data[i + 2] == 0x00 && data[i + 3] == 0x01))) {
      has_valid_nal = true;
      break;
    }
  }

  if (!has_valid_nal) {
    fan::print_throttled("No valid NAL unit found in frame");
    return false;
  }

  return true;
}

void debug_packet_structure(const std::vector<uint8_t>& data, size_t expected_size) {
  if (data.size() >= 16) {
    fan::print_throttled_format("Packet debug: size={}, expected={}, first_bytes=[{:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}]",
      data.size(), expected_size,
      data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
  }
}

bool validate_network_packet(const std::vector<uint8_t>& data, size_t expected_size) {
  if (data.empty() || data.size() != expected_size) {
    return false;
  }

  if (data.size() < 16) {
    return false;
  }

  return true;
}
#endif


void ecps_backend_t::view_t::WriteFramePacket() {
  if (m_Possible == 0) {
    return;
  }

  uint32_t FramePacketSize = (uint32_t)(this->m_Possible - 1) * 0x400 + this->m_ModuloSize;

  if (FramePacketSize == 0 || FramePacketSize > 0x500000 || this->m_data.size() < FramePacketSize) {
    this->m_stats.Frame_Drop++;
    return;
  }

  std::vector<uint8_t> frame_data(this->m_data.begin(), this->m_data.begin() + FramePacketSize);

  bool has_sps_pps = false;

  // Detect SPS/PPS (always important regardless of debug level)
  for (size_t i = 0; i < std::min(frame_data.size() - 4, size_t(32)); ++i) {
    if (frame_data[i] == 0x00 && frame_data[i + 1] == 0x00 && frame_data[i + 2] == 0x00 && frame_data[i + 3] == 0x01) {
      uint8_t nal_type = frame_data[i + 4] & 0x1F;
      if (nal_type == 7 || nal_type == 8) {
        has_sps_pps = true;
        break;
      }
    }
  }

  // Prioritize important frames
  bool is_important = has_sps_pps || (FramePacketSize > 30000);

  // STARTUP DETECTION
  static auto stream_start_time = std::chrono::steady_clock::now();
  static bool was_streaming = false;
  bool currently_streaming = render_thread && render_thread->ecps_gui.is_streaming;

  if (currently_streaming && !was_streaming) {
    stream_start_time = std::chrono::steady_clock::now(); // Reset on stream start
  }
  was_streaming = currently_streaming;

  auto now = std::chrono::steady_clock::now();
  auto time_since_stream_start = std::chrono::duration_cast<std::chrono::seconds>(now - stream_start_time);
  bool is_startup_phase = (time_since_stream_start.count() < 10);

  {
    std::unique_lock<std::timed_mutex> lock(network_decode_mutex, std::defer_lock);

    // RELAXED locking during startup
    auto lock_timeout = is_startup_phase ?
      std::chrono::milliseconds(100) :
      std::chrono::milliseconds(is_important ? 50 : 10);

    if (!lock.try_lock_for(lock_timeout)) {
      // Don't drop frames during startup phase
      if (!is_startup_phase) {
        this->m_stats.Frame_Drop++;
      }
      return;
    }

    // RELAXED queue management during startup
    size_t max_queue_size = is_startup_phase ? 5 : MAX_DECODE_QUEUE_SIZE;

    while (network_decode_queue.size() >= max_queue_size) {
      if (is_important || is_startup_phase) {
        network_decode_queue.pop(); // Make room for important frames
      }
      else {
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
    // Ensure proper encoder settings
    screen_encode->settings.RateControl.VBR.bps = std::max(screen_encode->settings.RateControl.VBR.bps, 3000000u); // Minimum 3Mbps

    // Force more frequent keyframes
    screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
#if ecps_debug_prints >= 2
    fan::print_throttled_format("Encoder configured with bitrate: {} bps", screen_encode->settings.RateControl.VBR.bps);
#endif
  }
}



// Helper function to determine if we're viewing local or network stream
bool is_local_stream() {
  // You can determine this based on your application logic
  // For example, check if we're the host/streamer vs viewer
  return render_thread && render_thread->ecps_gui.is_streaming; // or however you determine this
}



int main() {
  std::promise<void> render_thread_promise;
  std::future<void> render_thread_future = render_thread_promise.get_future();

  uint64_t encode_start = 0;

  auto encode_thread_id = fan::event::thread_create([] {
    screen_encode = (::screen_encode_t*)malloc(sizeof(::screen_encode_t));
    std::construct_at(screen_encode);

    configure_encoder_for_artifact_reduction();

    while (screen_decode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    uint64_t frame_id = 0;
    bool first_frame = true;
    auto last_idr_time = std::chrono::steady_clock::now();

    while (1) {
      if (render_thread->ecps_gui.is_streaming) {
        if (first_frame) {
          screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
          first_frame = false;
          last_idr_time = std::chrono::steady_clock::now();
        }

        auto now = std::chrono::steady_clock::now();
        auto time_since_idr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_idr_time);

        // MUCH less aggressive: Force IDR only every 5 seconds
        if (time_since_idr > std::chrono::milliseconds(5000)) {
          screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
          last_idr_time = now;
#if ecps_debug_prints >= 2
          fan::print_throttled("Encode thread forced IDR: 5-second interval");
#endif
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

          // Network transmission (unchanged)
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

          // Local decode (unchanged)
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


  // Alternative: More explicit version with separate functions
  static auto request_local_idr = [](const std::string& reason = "") {
    if (screen_encode) {
      screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
#if ecps_debug_prints >= 2
      fan::print_throttled_format("Local IDR requested: {}", reason);
#endif
    }
    };

  static auto request_network_idr = [](const std::string& reason = "") -> fan::event::task_t {
    try {
      ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
      rest.ChannelID = ecps_backend.channel_info.front().channel_id;
      rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
      co_await ecps_backend.tcp_write(
        ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
        &rest,
        sizeof(rest)
      );
#if ecps_debug_prints >= 2
      fan::print_throttled_format("Network IDR requested: {}", reason);
#endif
    }
    catch (...) {
#if ecps_debug_prints >= 2
      fan::print_throttled_format("Failed to send network IDR: {}", reason);
#endif
    }
    };



  static auto request_idr_smart = [](const std::string& reason = "") {
    if (is_local_stream()) {
      // LOCAL: Direct encoder flag manipulation (fast and direct)
      if (screen_encode) {
        screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
#if ecps_debug_prints >= 2
        fan::print_throttled_format("Requested local IDR: {}", reason);
#endif
      }
    }
    else {
      // NETWORK: Use protocol to request IDR from remote encoder
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
#if ecps_debug_prints >= 2
            fan::print_throttled_format("Sent network IDR request: {}", reason);
#endif
          }
          catch (...) {}
          });
      }
      catch (...) {
#if ecps_debug_prints >= 2
        fan::print_throttled_format("Failed to send network IDR request: {}", reason);
#endif
      }
    }
    };

  // ENHANCED decode thread with proper decoder state management:
 // Updated decode thread with proper error code handling

  auto decode_thread_id = fan::event::thread_create([] {
    screen_decode = (::screen_decode_t*)malloc(sizeof(::screen_decode_t));
    std::construct_at(screen_decode);

    // No special setup needed - decoder changes now execute immediately

    while (screen_encode == nullptr || render_thread == nullptr || render_thread->should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto last_network_frame_time = std::chrono::steady_clock::now();
    auto startup_time = std::chrono::steady_clock::now();
    auto last_successful_decode = std::chrono::steady_clock::now();
    auto last_decoder_change = std::chrono::steady_clock::now();
    const auto MAX_FRAME_AGE = std::chrono::milliseconds(500);
    uint64_t last_decoded_frame_id = 0;
    uint64_t consecutive_decode_failures = 0;
    uint64_t successful_decodes = 0;
    bool decoder_initialized = false;
    bool decoder_needs_reset = false;
    uint8_t last_decode_type = 255;

    // Specific error tracking for different error types
    uint64_t consecutive_not_readable = 0;
    uint64_t consecutive_read_failures = 0;
    uint64_t consecutive_stride_errors = 0;
    uint64_t consecutive_format_errors = 0;
    uint64_t consecutive_decoder_closed = 0;

    // DECODER CHANGE STATE TRACKING
    bool decoder_recently_changed = false;
    uint64_t frames_since_decoder_change = 0;

#if ecps_debug_prints >= 2
    fan::print_throttled("Decode thread started with enhanced error handling");
#endif

    while (1) {
      bool processed_frame = false;
      auto now = std::chrono::steady_clock::now();
      auto time_since_startup = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);
      auto time_since_success = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_successful_decode);
      auto time_since_decoder_change = std::chrono::duration_cast<std::chrono::seconds>(now - last_decoder_change);

      // Enhanced reset conditions based on error patterns
      bool should_reset = false;
      if (decoder_recently_changed) {
        // More lenient reset conditions after decoder change
        if (time_since_success > std::chrono::milliseconds(10000) && frames_since_decoder_change > 50) {
          should_reset = true;
        }
      }
      else {
        // Different reset triggers for different error patterns
        if ((consecutive_decoder_closed > 20) ||
          (consecutive_not_readable > 50 && time_since_success > std::chrono::milliseconds(2000)) ||
          (consecutive_read_failures > 30) ||
          (consecutive_stride_errors > 10) ||
          (consecutive_format_errors > 10) ||
          (time_since_success > std::chrono::milliseconds(5000) && successful_decodes > 0)) {
          should_reset = true;
        }
      }

      if (should_reset) {
        decoder_needs_reset = true;
#if ecps_debug_prints >= 2
        fan::print_throttled_format("Decoder reset needed: closed={}, not_readable={}, read_fail={}, stride_err={}, format_err={}",
          consecutive_decoder_closed, consecutive_not_readable, consecutive_read_failures,
          consecutive_stride_errors, consecutive_format_errors);
#endif
      }

      {
        std::unique_lock<std::timed_mutex> network_lock(network_decode_mutex, std::try_to_lock);
        if (network_lock.owns_lock() && !network_decode_queue.empty()) {

          frame_timing_t frame_to_decode;
          bool found_valid_frame = false;

          // Find the most recent valid frame
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
            frame_to_decode.decode_time = now;
            last_network_frame_time = now;

            if (frame_to_decode.data && !frame_to_decode.data->empty()) {
              uint64_t current_frame_id = frame_to_decode.frame_id;
              frames_since_decoder_change++;

              // Reset decoder BEFORE processing if needed
              if (decoder_needs_reset) {
                try {
#if ecps_debug_prints >= 2
                  fan::print_throttled("Resetting decoder state...");
#endif
                  screen_decode->close_decoder();
                  std::this_thread::sleep_for(std::chrono::milliseconds(50));
                  screen_decode->open_decoder();

                  // Clear render frame queue
                  std::lock_guard<std::timed_mutex> frame_lock(render_mutex);
                  while (render_thread->FrameList.Usage() > 0) {
                    auto old = render_thread->FrameList.GetNodeFirst();
                    render_thread->FrameList.unlrec(old);
                  }

                  // Reset all counters
                  decoder_needs_reset = false;
                  consecutive_decode_failures = 0;
                  consecutive_not_readable = 0;
                  consecutive_read_failures = 0;
                  consecutive_stride_errors = 0;
                  consecutive_format_errors = 0;
                  consecutive_decoder_closed = 0;
                  last_decode_type = 255;
                  decoder_recently_changed = false;
                  frames_since_decoder_change = 0;
#if ecps_debug_prints >= 2
                  fan::print_throttled("Decoder reset completed");
#endif
                }
                catch (...) {
#if ecps_debug_prints >= 2
                  fan::print_throttled("Decoder reset failed");
#endif
                }
              }

              try {
                // Call the decode function
                fan::graphics::screen_decode_t::decode_data_t decode_data = screen_decode->decode(
                  frame_to_decode.data->data(),
                  frame_to_decode.data->size(),
                  render_thread->screen_frame
                );

                last_decode_type = decode_data.type;

                // HANDLE ALL ERROR CODES WITH SPECIFIC STRATEGIES
                if (decode_data.type == 253) { // Decoder changed
                  decoder_recently_changed = true;
                  frames_since_decoder_change = 0;
                  last_decoder_change = now;
                  consecutive_decode_failures = 0;
                  consecutive_decoder_closed = 0;
                  consecutive_not_readable = 0;
                  consecutive_read_failures = 0;
                  consecutive_stride_errors = 0;
                  consecutive_format_errors = 0;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} - decoder changed successfully", current_frame_id);
#endif
                  continue; // Skip this frame, decoder was just changed
                }
                else if (decode_data.type == 254) { // Decoder closed
                  consecutive_decoder_closed++;
                  consecutive_not_readable = 0;
                  consecutive_read_failures = 0;
                  consecutive_stride_errors = 0;
                  consecutive_format_errors = 0;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} - decoder closed (count: {})", current_frame_id, consecutive_decoder_closed);
#endif

                  // Try to reopen decoder after multiple closes
                  if (consecutive_decoder_closed > 5) {
                    try {
                      screen_decode->open_decoder();
                      consecutive_decoder_closed = 0;
#if ecps_debug_prints >= 2
                      fan::print_throttled("Decoder reopened after multiple closes");
#endif
                    }
                    catch (...) {}
                  }
                  continue;
                }
                else if (decode_data.type == 252) { // Not readable
                  consecutive_not_readable++;
                  consecutive_decoder_closed = 0;
                  consecutive_read_failures = 0;
                  consecutive_stride_errors = 0;
                  consecutive_format_errors = 0;
#if ecps_debug_prints >= 2
                  if (consecutive_not_readable % 20 == 1) { // Throttled logging
                    fan::print_throttled_format("Frame {} - not readable (count: {})", current_frame_id, consecutive_not_readable);
                  }
#endif

                  // For not readable, just continue - this is often temporary
                  continue;
                }
                else if (decode_data.type == 251) { // Read failed
                  consecutive_read_failures++;
                  consecutive_decoder_closed = 0;
                  consecutive_not_readable = 0;
                  consecutive_stride_errors = 0;
                  consecutive_format_errors = 0;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} - read failed (count: {})", current_frame_id, consecutive_read_failures);
#endif

                  // Read failures often indicate corrupted data - request IDR
                  if (consecutive_read_failures % 10 == 0) {
                    static auto last_idr_request = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_idr_request).count() > 2) {
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
                          }
                        }
                        catch (...) {}
                        });
                      last_idr_request = now;
                    }
                  }
                  continue;
                }
                else if (decode_data.type == 250) { // Unsupported stride
                  consecutive_stride_errors++;
                  consecutive_decoder_closed = 0;
                  consecutive_not_readable = 0;
                  consecutive_read_failures = 0;
                  consecutive_format_errors = 0;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} - unsupported stride (count: {})", current_frame_id, consecutive_stride_errors);
#endif

                  // Stride errors indicate format mismatch - may need decoder change
                  if (consecutive_stride_errors > 5) {
                    decoder_needs_reset = true;
                  }
                  continue;
                }
                else if (decode_data.type == 249) { // Unsupported pixel format
                  consecutive_format_errors++;
                  consecutive_decoder_closed = 0;
                  consecutive_not_readable = 0;
                  consecutive_read_failures = 0;
                  consecutive_stride_errors = 0;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} - unsupported pixel format (count: {})", current_frame_id, consecutive_format_errors);
#endif

                  // Format errors definitely need decoder reset
                  if (consecutive_format_errors > 3) {
                    decoder_needs_reset = true;
                  }
                  continue;
                }

                // SUCCESS PATH - Reset all error counters
                consecutive_decoder_closed = 0;
                consecutive_not_readable = 0;
                consecutive_read_failures = 0;
                consecutive_stride_errors = 0;
                consecutive_format_errors = 0;

                // Enhanced validation based on type
                bool decode_successful = false;

                if (decode_data.type == 0) { // CUVID
                  decode_successful = true;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} CUVID decode successful", current_frame_id);
#endif
                }
                else if (decode_data.type == 1) { // YUV
                  // Relaxed validation right after decoder change
                  if (decoder_recently_changed && frames_since_decoder_change < 10) {
                    decode_successful = !decode_data.data[0].empty() &&
                      decode_data.image_size.x > 0 &&
                      decode_data.image_size.y > 0;
#if ecps_debug_prints >= 2
                    fan::print_throttled_format("Frame {} YUV decode (relaxed validation after change): success={}",
                      current_frame_id, decode_successful);
#endif
                  }
                  else {
                    // Full validation for stable decoder
                    bool has_y_data = !decode_data.data[0].empty();
                    bool has_u_data = !decode_data.data[1].empty();
                    bool has_v_data = !decode_data.data[2].empty();
                    bool valid_dimensions = decode_data.image_size.x > 0 &&
                      decode_data.image_size.y > 0 &&
                      decode_data.image_size.x <= 7680 &&
                      decode_data.image_size.y <= 4320;
                    bool valid_stride = decode_data.stride[0].x >= decode_data.image_size.x &&
                      decode_data.stride[0].x > 0;

                    decode_successful = has_y_data && has_u_data && has_v_data &&
                      valid_dimensions && valid_stride;
#if ecps_debug_prints >= 2
                    fan::print_throttled_format("Frame {} YUV decode (full validation): success={}",
                      current_frame_id, decode_successful);
#endif
                  }
                }
                else if (decode_data.type == 255) { // Error type
                  decode_successful = false;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} decode returned error type 255", current_frame_id);
#endif
                }

                if (decode_successful) {
                  std::lock_guard<std::timed_mutex> frame_lock(render_mutex);

                  // Keep only the latest frame
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
                  last_decoded_frame_id = current_frame_id;
                  last_successful_decode = now;
                  decoder_initialized = true;

                  // Clear "recently changed" flag after successful decode
                  if (decoder_recently_changed && frames_since_decoder_change > 5) {
                    decoder_recently_changed = false;
#if ecps_debug_prints >= 2
                    fan::print_throttled_format("Decoder change stabilized after {} frames", frames_since_decoder_change);
#endif
                  }

#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Frame {} decoded successfully (type: {})",
                    current_frame_id, decode_data.type);
#endif
                }
                else {
                  consecutive_decode_failures++;
                  if (current_frame_id > last_decoded_frame_id) {
                    last_decoded_frame_id = current_frame_id;
                  }
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("Decode validation failed for frame {} (type: {})",
                    current_frame_id, decode_data.type);
#endif
                }
              }
              catch (const std::exception& e) {
                consecutive_decode_failures++;
                if (current_frame_id > last_decoded_frame_id) {
                  last_decoded_frame_id = current_frame_id;
                }
#if ecps_debug_prints >= 2
                fan::print_throttled_format("Decode exception for frame {}: {}", current_frame_id, e.what());
#endif
              }
            }
          }
        }
      }

      // Enhanced IDR request logic based on error patterns
      static auto last_idr_request = std::chrono::steady_clock::now();
      auto time_since_last_idr = std::chrono::duration_cast<std::chrono::seconds>(now - last_idr_request);

      bool needs_idr = false;
      if (decoder_recently_changed) {
        // Less aggressive IDR requests right after decoder change
        needs_idr = (consecutive_decode_failures > 100 && time_since_decoder_change.count() > 5);
      }
      else {
        // Normal IDR request logic based on specific error patterns
        needs_idr = (consecutive_decode_failures > 30 && !decoder_initialized && time_since_startup.count() > 5) ||
          (consecutive_read_failures > 20) ||
          (consecutive_format_errors > 5) ||
          (consecutive_stride_errors > 8) ||
          (last_decode_type == 255 && time_since_last_idr.count() > 2);
      }

      if (needs_idr && time_since_last_idr.count() > 3) {
#if ecps_debug_prints >= 2
        fan::print_throttled_format("Requesting IDR: failures={}, read_fail={}, format_err={}, stride_err={}",
          consecutive_decode_failures, consecutive_read_failures, consecutive_format_errors, consecutive_stride_errors);
#endif
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
            }
          }
          catch (...) {}
          });
        last_idr_request = now;
        consecutive_decode_failures = 0;
      }

      if (!processed_frame) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
    }
    });

  auto frequent_idr_task = fan::event::task_timer(1000, []() -> fan::event::task_value_resume_t<bool> { // Changed from 250ms to 1000ms
    static auto startup_time = std::chrono::steady_clock::now();
    static bool startup_phase = true;

    auto now = std::chrono::steady_clock::now();
    auto time_since_startup = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);

    if (render_thread && render_thread->ecps_gui.is_streaming && screen_encode) {
      // STARTUP PHASE: First 10 seconds - minimal IDR
      if (startup_phase && time_since_startup.count() < 10) {
        // Only force IDR every 5 seconds during startup
        static auto last_startup_idr = startup_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_startup_idr).count() > 5000) {
          screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
          last_startup_idr = now;
#if ecps_debug_prints >= 2
          fan::print_throttled("Startup IDR (5s interval)");
#endif
        }
      }
      // NORMAL PHASE: After 10 seconds - moderate IDR for artifact prevention
      else {
        startup_phase = false;
        // Only every 1 second instead of 250ms
        screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
#if ecps_debug_prints >= 2
        fan::print_throttled("Normal IDR (1s interval)");
#endif
      }
    }

    // Reset startup timer when streaming stops/starts
    if (!render_thread || !render_thread->ecps_gui.is_streaming) {
      startup_time = now;
      startup_phase = true;
    }

    co_return 0;
    });


  auto motion_idr_task = fan::event::task_timer(100, []() -> fan::event::task_value_resume_t<bool> {
    static uint64_t last_frame_count = 0;
    static int motion_frames = 0;
    static auto startup_time = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    auto time_since_startup = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);

    if (render_thread && render_thread->ecps_gui.is_streaming && screen_encode) {
      uint64_t current_frame_count = screen_encode->frame_counter.load();
      uint64_t frames_in_100ms = current_frame_count - last_frame_count;

      // IGNORE MOTION DURING STARTUP (first 10 seconds)
      if (time_since_startup.count() > 10) {
        // Only trigger motion IDR after startup phase
        if (frames_in_100ms > 6) {
          motion_frames++;
          if (motion_frames >= 5) { // Changed from 3 to 5 (500ms instead of 300ms)
            screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
#if ecps_debug_prints >= 2
            fan::print_throttled_format("Motion IDR: {} fps detected", frames_in_100ms * 10);
#endif
            motion_frames = 0;
          }
        }
        else {
          motion_frames = 0;
        }
      }

      last_frame_count = current_frame_count;
    }
    else {
      // Reset startup timer when streaming stops
      startup_time = now;
    }

    co_return 0;
    });


  auto render_thread_id = fan::event::thread_create([&render_thread_promise, &encode_start] {
    render_thread = (render_thread_t*)malloc(sizeof(render_thread_t));
    std::construct_at(render_thread);

    render_thread_promise.set_value(); // signal

    while (!render_thread->engine.should_close() && !render_thread->should_stop.load()) {

      {
        std::lock_guard<std::timed_mutex> render_lock(render_mutex);
        auto flnr = render_thread->FrameList.GetNodeFirst();
        if (flnr == render_thread->FrameList.dst) {
          goto g_feed_frames_skip;
        }
        auto& node = render_thread->FrameList[flnr];

        bool frame_valid = false;

        for (auto& i : screen_decode->graphics_queue) {
          i();
        }

        //screen_decode->graphics_queue_callback();

        if (node.type == 0) { // cuvid
          try {
            screen_decode->decode_cuvid(render_thread->screen_frame);
            frame_valid = true;
#if ecps_debug_prints >= 2
            fan::print_throttled("RENDER: CUVID frame processed");
#endif
          }
          catch (...) {
#if ecps_debug_prints >= 2
            fan::print_throttled("RENDER: CUVID decode failed");
#endif
          }
        }
        else if (node.type == 1) {
          // YUV validation and processing
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

              if (raw_ptrs[0] != nullptr && raw_ptrs[1] != nullptr && raw_ptrs[2] != nullptr) {
                try {
                  render_thread->screen_frame.set_tc_size(fan::vec2(sx, 1));
                  render_thread->screen_frame.reload(
                    node.pixel_format,
                    raw_ptrs.data(),
                    fan::vec2ui(node.stride[0].x, node.image_size.y),
                    fan::graphics::image_filter::linear
                  );
                  frame_valid = true;
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("RENDER: YUV frame processed {}x{}, stride={}, sx={}",
                    node.image_size.x, node.image_size.y, node.stride[0].x, sx);
#endif
                }
                catch (const std::exception& e) {
#if ecps_debug_prints >= 2
                  fan::print_throttled_format("RENDER: YUV reload failed: {}", e.what());
#endif
                }
              }
            }
            else {
#if ecps_debug_prints >= 2
              fan::print_throttled_format("RENDER: Invalid sx ratio: {}", sx);
#endif
            }
          }
          else {
#if ecps_debug_prints >= 2
            fan::print_throttled_format("RENDER: YUV validation failed: Y={}, U={}, V={}, dims={}x{}, stride={}",
              y_valid, u_valid, v_valid, node.image_size.x, node.image_size.y, node.stride[0].x);
#endif
          }
        }

        render_thread->FrameList.unlrec(flnr);

        // Only request IDR if frame was invalid and we're not spamming
        if (!frame_valid) {
          static auto last_idr_request = std::chrono::steady_clock::now();
          auto now = std::chrono::steady_clock::now();
          if (std::chrono::duration_cast<std::chrono::seconds>(now - last_idr_request).count() > 5) {
            if (screen_encode) {
              screen_encode->encode_write_flags |= fan::graphics::codec_update_e::reset_IDR;
              last_idr_request = now;
#if ecps_debug_prints >= 2
              fan::print_throttled("RENDER: Requesting IDR due to invalid frame");
#endif
            }
          }
        }
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
#if ecps_debug_prints >= 2
                fan::print_throttled("Failed to request channel list");
#endif
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

  auto enhanced_network_task = fan::event::task_timer(1, []() -> fan::event::task_value_resume_t<bool> { // 1ms for maximum responsiveness
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

    // Faster bucket refill for responsive keyframe transmission
    ecps_backend.share.m_NetworkFlow.Bucket +=
      (f32_t)DeltaTime / 1000000000 * screen_encode->settings.RateControl.VBR.bps * 3.0; // Triple rate

    if (ecps_backend.share.m_NetworkFlow.Bucket > ecps_backend.share.m_NetworkFlow.BucketSize) {
      ecps_backend.share.m_NetworkFlow.Bucket = ecps_backend.share.m_NetworkFlow.BucketSize;
    }

    auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
    if (flnr == ecps_backend.share.m_NetworkFlow.FrameList.dst) {
      co_return 0;
    }

    auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];

    // Detect keyframes by size (they're much larger)
    bool is_likely_keyframe = f->vec.size() > 30000; // 30KB+ likely keyframe

    uint8_t Flag = 0;
    uint16_t Possible = (f->vec.size() / 0x400) + !!(f->vec.size() % 0x400);
    uint16_t sent_offset = f->SentOffset;

    // Send keyframes with highest priority - up to 20 chunks at once
    constexpr size_t MAX_CHUNKS = 20;
    size_t chunks_to_send = is_likely_keyframe ? MAX_CHUNKS : 8;
    size_t chunks_sent = 0;

    for (; sent_offset < Possible && chunks_sent < chunks_to_send; sent_offset++, chunks_sent++) {
      uintptr_t DataSize = f->vec.size() - sent_offset * 0x400;
      if (DataSize > 0x400) {
        DataSize = 0x400;
      }

      // Allow bucket deficit for keyframes (for artifact recovery)
      if (!is_likely_keyframe && ecps_backend.share.m_NetworkFlow.Bucket < DataSize * 8) {
        break;
      }

      bool ret = co_await ecps_backend.write_stream(sent_offset, Possible, Flag, &f->vec[sent_offset * 0x400], DataSize);
      if (ret != false) {
        break;
      }

      // Deduct from bucket, but allow deficit for keyframes
      if (ecps_backend.share.m_NetworkFlow.Bucket >= DataSize * 8) {
        ecps_backend.share.m_NetworkFlow.Bucket -= DataSize * 8;
      }
      else if (is_likely_keyframe) {
        // Allow negative bucket for keyframes
        ecps_backend.share.m_NetworkFlow.Bucket -= DataSize * 8;
      }
    }

    f->SentOffset = sent_offset;

    if (sent_offset >= Possible) {
      if (is_likely_keyframe) {
#if ecps_debug_prints >= 2
        fan::print_throttled_format("Keyframe transmitted: {} bytes in {} chunks",
          f->vec.size(), chunks_sent);
#endif
      }
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