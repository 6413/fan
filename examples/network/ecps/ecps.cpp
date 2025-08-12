#include <fan/types/types.h>
#include <fan/time/time.h>
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

extern "C" {
#include <libavutil/pixfmt.h>
}

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

struct render_thread_t;
#define ecps_debug_prints 0

std::atomic<render_thread_t*> render_thread_ptr{ nullptr };

render_thread_t* get_render_thread() {
  return render_thread_ptr.load(std::memory_order_acquire);
}

#define render_thread get_render_thread()

std::mutex render_thread_mutex;
std::condition_variable render_thread_cv;
std::atomic<bool> render_thread_ready{ false };

#include "backend.h"

struct dynamic_config_t {
  static uint32_t get_target_framerate();

  static uint64_t get_adaptive_bitrate() {
    uint64_t  fps = get_target_framerate();
    if (fps >= 120) return 12000000;
    else if (fps >= 90) return 9000000;
    else if (fps >= 60) return 6000000;
    else return 4000000;
  }

  static uint32_t get_adaptive_bucket_size() {
    uint32_t fps = get_target_framerate();
    uint32_t base_bitrate = get_adaptive_bitrate();
    if (fps >= 120) return base_bitrate * 3;
    else if (fps >= 60) return base_bitrate * 2;
    else return base_bitrate;
  }

  static size_t get_adaptive_pool_size() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 256;
    else if (fps >= 90) return 192;
    else if (fps >= 60) return 128;
    else return 64;
  }

  static size_t get_adaptive_queue_size() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 8;
    else if (fps >= 90) return 6;
    else if (fps >= 60) return 4;
    else return 3;
  }

  static uint32_t get_adaptive_motion_poll_ms() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 30;
    else if (fps >= 90) return 40;
    else if (fps >= 60) return 50;
    else return 80;
  }

  static size_t get_adaptive_chunk_count() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 40;
    else if (fps >= 90) return 30;
    else if (fps >= 60) return 25;
    else return 20;
  }

  static f32_t get_adaptive_bucket_multiplier() {
    uint32_t fps = get_target_framerate();
    if (fps >= 120) return 8.0f;
    else if (fps >= 90) return 6.5f;
    else if (fps >= 60) return 5.0f;
    else return 3.5f;
  }
};

void ecps_backend_t::share_t::CalculateNetworkFlowBucket() {
  uintptr_t MaxBufferSize = (sizeof(ScreenShare_StreamHeader_Head_t) + 0x400) * 8;

  m_NetworkFlow.BucketSize = dynamic_config_t::get_adaptive_bucket_size();
  m_NetworkFlow.Bucket = m_NetworkFlow.BucketSize;

#if ecps_debug_prints >= 1
  uint32_t fps = dynamic_config_t::get_target_framerate();
  fan::print_throttled_format("Adaptive bucket: {}fps -> {} bits ({} MB)",
    fps, m_NetworkFlow.BucketSize, m_NetworkFlow.BucketSize / 8 / 1024 / 1024);
#endif
}

struct render_thread_t {
  render_thread_t() { 
    engine.set_target_fps(0, false);
    
    fan::graphics::codec_config_t config;
    config.width = 1920;
    config.height = 1080; 
    config.frame_rate = 30;
    config.bitrate = 10000000;
    
    if (!screen_encoder.open(config)) {
      fan::print("Failed to open modern encoder");
    }
    
    if (!screen_decoder.open()) {
      fan::print("Failed to open modern decoder");
    }
    screen_decoder.reload_codec_cb = [this]() {
      ecps_gui.backend_queue([=]() -> fan::event::task_t {
        try {
          for (const auto& channel : ecps_backend.channel_info) {
            if (channel.is_viewing) {
              ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
              rest.ChannelID = channel.channel_id;
              rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
              co_await ecps_backend.tcp_write(
                ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
                &rest,
                sizeof(rest)
              );
            }
          }
        }
        catch (...) {}
        });
    };
  }

  engine_t engine;
#define engine OFFSETLESS(This, render_thread_t, ecps_gui)->engine
#include "gui.h"
  ecps_gui_t ecps_gui;

  fan::graphics::screen_encode_t screen_encoder;
  fan::graphics::screen_decode_t screen_decoder;

  std::condition_variable frame_cv;
  std::condition_variable task_cv;
  std::atomic<bool> has_task_work{ false };
  std::atomic<bool> should_stop{ false };

  fan::graphics::image_t screen_image = engine.image_create();

  fan::graphics::universal_image_renderer_t network_frame{ {
    .position = fan::vec3(fan::vec2(0), 1),
    .size = gloco->window.get_size() / 2,
  } };

  void render(auto l) {
    if (engine.process_loop([this, l] { ecps_gui.render(); l(); })) {
      std::exit(0);
    }
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

uint32_t dynamic_config_t::get_target_framerate() {
  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  if (rt && rt->screen_encoder.config_.frame_rate > 0) {
    return rt->screen_encoder.config_.frame_rate;
  }
  return 60;
}

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
    co_await backend.udp_write(0, ProtocolUDP::C2S_t::KeepAlive, {}, 0, 0);
    co_return;
    };

  __dme_get(Protocol_S2C, Response_Login) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Response_Login_t>();
    backend.session_id = msg->SessionID;
    co_await backend.udp_write(0, ProtocolUDP::C2S_t::KeepAlive, {}, 0, 0);
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
    for (auto& channel : channel_info) {
      if (channel.channel_id.i == msg->ChannelID.i) {
        channel.session_id = msg->ChannelSessionID;
        break;
      }
    }
    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (rt) {
      rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
    }
    };

  __dme_get(Protocol_S2C, JoinChannel_Error) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_Error_t>();
    };

  __dme_get(Protocol_S2C, Channel_ScreenShare_ViewToShare) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_ViewToShare_t>();
    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (!rt || !ecps_backend.is_channel_streaming(msg->ChannelID)) {
      co_return;
    }
    if (msg->Flag & ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR) {

      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      if (rt) {
        rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
      }
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

void ecps_backend_t::view_t::WriteFramePacket() {
  if (m_Possible == 0) return;

  uint32_t FramePacketSize = (uint32_t)(this->m_Possible - 1) * 0x400 + this->m_ModuloSize;
  if (FramePacketSize == 0 || FramePacketSize > 0x500000 || this->m_data.size() < FramePacketSize) {
    this->m_stats.Frame_Drop++;
#if ecps_debug_prints >= 1
    fan::print_format("NETWORK: Frame dropped - invalid size: {} (max: 0x500000, buffer: {})", 
                     FramePacketSize, this->m_data.size());
#endif
    return;
  }

#if ecps_debug_prints >= 2
  printf("NETWORK: Frame received at %lld ms, size: %u bytes\n",
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count(),
    FramePacketSize);
#endif

  bool is_keyframe = false;
  if (FramePacketSize >= 8) {
    for (size_t i = 0; i <= std::min(static_cast<size_t>(FramePacketSize - 4), size_t(64)); ++i) {
      if (this->m_data[i] == 0x00 && this->m_data[i + 1] == 0x00 && 
          this->m_data[i + 2] == 0x00 && this->m_data[i + 3] == 0x01) {
        uint8_t nal_type = this->m_data[i + 4] & 0x1F;
        if (nal_type == 7 || nal_type == 8) { // SPS or PPS
          is_keyframe = true;
#if ecps_debug_prints >= 2
          printf("NETWORK: Keyframe detected (NAL type: %d)\n", nal_type);
#endif
          break;
        }
      }
    }
  }

  if (!is_keyframe && FramePacketSize > 50000) {
    is_keyframe = true;
#if ecps_debug_prints >= 2
    printf("NETWORK: Large frame treated as keyframe (%u bytes)\n", FramePacketSize);
#endif
  }

  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  if (!rt) {
#if ecps_debug_prints >= 1
    fan::print_throttled("NETWORK: No render thread available");
#endif
    this->m_stats.Frame_Drop++;
    return;
  }

  std::vector<uint8_t> frame_data;
  
  if (FramePacketSize == this->m_data.size()) {
    frame_data.swap(this->m_data);
    this->m_data.clear();
    this->m_data.resize(0x400400);
#if ecps_debug_prints >= 3
    printf("NETWORK: Zero-copy swap for frame %llu\n", frame_index);
#endif
  } else {
    frame_data.resize(FramePacketSize);
    std::memcpy(frame_data.data(), this->m_data.data(), FramePacketSize);
#if ecps_debug_prints >= 3
    printf("NETWORK: Memcpy for frame %llu (%u bytes)\n", frame_index, FramePacketSize);
#endif
  }

  try {
    auto decode_result = rt->screen_decoder.decode(
      frame_data.data(),
      frame_data.size(),
      rt->network_frame
    );

    if (decode_result.type == 1) { // success
      bool has_valid_data = false;

      if (decode_result.pixel_format == fan::graphics::image_format::yuv420p) {
        has_valid_data = !decode_result.data[0].empty() &&
          !decode_result.data[1].empty() &&
          !decode_result.data[2].empty();
      }
      else if (decode_result.pixel_format == fan::graphics::image_format::nv12) {
        has_valid_data = !decode_result.data[0].empty() &&
          !decode_result.data[1].empty();
      }
      else {
        fan::print("weird format");
        has_valid_data = !decode_result.data[0].empty();
      }
      bool valid_dimensions = decode_result.image_size.x > 0 && 
                             decode_result.image_size.y > 0 &&
                             decode_result.image_size.x <= 7680 && 
                             decode_result.image_size.y <= 4320;
      bool valid_stride = decode_result.stride[0].x >= decode_result.image_size.x && 
                         decode_result.stride[0].x > 0;

      if (has_valid_data && valid_dimensions && valid_stride) {
        std::lock_guard<std::timed_mutex> frame_lock(render_mutex);
        
        while (rt->FrameList.Usage() > 0) {
          auto old = rt->FrameList.GetNodeFirst();
          rt->FrameList.unlrec(old);
        }

        rt->screen_decoder.decoded_size = decode_result.image_size;

        auto flnr = rt->FrameList.NewNodeLast();
        auto f = &rt->FrameList[flnr];
        *f = std::move(decode_result);

#if ecps_debug_prints >= 2
        printf("NETWORK: Frame %llu decoded and queued successfully (%ux%u)\n",
               frame_index, decode_result.image_size.x, decode_result.image_size.y);
#endif
      } else {
#if ecps_debug_prints >= 1
        fan::print_throttled_format("NETWORK: Decoded frame validation failed - data:{} dims:{}x{} stride:{}",
                                   has_valid_data, decode_result.image_size.x, 
                                   decode_result.image_size.y, decode_result.stride[0].x);
#endif
        this->m_stats.Frame_Drop++;
      }
    } else {
#if ecps_debug_prints >= 1
      fan::print_throttled_format("NETWORK: Decode failed with type: {}", decode_result.type);
#endif
      this->m_stats.Frame_Drop++;
    }
  } catch (const std::exception& e) {
#if ecps_debug_prints >= 1
    fan::print_throttled_format("NETWORK: Decode exception: {}", e.what());
#endif
    this->m_stats.Frame_Drop++;
  }

  frame_index++;
}

int main() {

  ecps_backend.login_fail_cb = [] (fan::exception_t e) {
    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (rt) {// might be able to be removed
      rt->screen_decoder.graphics_queue_callback([e] {
        fan::printcl("failed to connect to server:"_str + e.reason + ", retrying...");
      });
    }
  };

  std::promise<void> render_thread_promise;
  std::future<void> render_thread_future = render_thread_promise.get_future();

  fan::event::thread_create([&render_thread_promise] {
    render_thread_t render_thread_instance;

    render_thread_ptr.store(&render_thread_instance, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(render_thread_mutex);
      render_thread_ready.store(true, std::memory_order_release);
    }
    render_thread_cv.notify_all();
    render_thread_promise.set_value();

    while (!render_thread_instance.screen_decoder.is_initialized() ||
      render_thread_instance.should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    while (!render_thread_instance.engine.should_close() && !render_thread_instance.should_stop.load()) {
      {
        std::lock_guard<std::timed_mutex> render_lock(render_mutex);
        {
          std::lock_guard<std::mutex> lock(render_thread_instance.screen_decoder.mutex);
          for (auto& i : render_thread_instance.screen_decoder.graphics_queue) {
            i();
          }
          render_thread_instance.screen_decoder.graphics_queue.clear();
        }

        if (render_thread_instance.ecps_gui.show_own_stream == false) {
          auto flnr = render_thread_instance.FrameList.GetNodeFirst();
          if (flnr != render_thread_instance.FrameList.dst) {
            auto& node = render_thread_instance.FrameList[flnr];
            bool frame_valid = false;
#if ecps_debug_prints >= 2
            printf("RENDER: Frame displayed at %lld ms\n",
              std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
#endif

            if (ecps_backend.did_just_join) {
              auto* rt = render_thread_ptr.load(std::memory_order_acquire);
              if (rt) {
                rt->ecps_gui.window_handler.main_tab = 1;
              }
              ecps_backend.did_just_join = false;
            }

            // Handle different frame types from LibAV decoder
            if (node.type == 0) {
              // Hardware decoded frame (CUDA/NVENC) - if supported

            }
            else if (node.type == 1) {
              // Software decoded frame (YUV data)
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
                      render_thread_instance.network_frame.set_tc_size(fan::vec2(sx, 1));
                      render_thread_instance.network_frame.reload(
                        node.pixel_format,
                        raw_ptrs.data(),
                        fan::vec2ui(node.stride[0].x, node.image_size.y)
                      );
                      frame_valid = true;
#if ecps_debug_prints >= 2
                      static uint64_t yuv_render_count = 0;
                      if (++yuv_render_count % 30 == 1) {
                        fan::print_throttled_format("RENDER: LibAV YUV frame processed {}x{}, stride={}, sx={} ({})",
                          node.image_size.x, node.image_size.y, node.stride[0].x, sx, yuv_render_count);
                      }
#endif
                    }
                    catch (const std::exception& e) {
#if ecps_debug_prints >= 1
                      fan::print_throttled_format("RENDER: YUV reload failed for LibAV frame: {}", e.what());
#endif
                    }
                  }
                }
                else {
#if ecps_debug_prints >= 1
                  fan::print_throttled_format("RENDER: Invalid sx ratio for LibAV frame: {}", sx);
#endif
                }
              }
              else {
#if ecps_debug_prints >= 1
                fan::print_throttled_format("RENDER: YUV validation failed for LibAV frame: Y={}, U={}, V={}, dims={}x{}, stride={}",
                  y_valid, u_valid, v_valid, node.image_size.x, node.image_size.y, node.stride[0].x);
#endif
              }
            }
            else if (node.type >= 250) {
              // Error codes from LibAV decoder
              switch (node.type) {
              case 254:
#if ecps_debug_prints >= 1
                fan::print_throttled("RENDER: LibAV decoder failed to reopen");
#endif
                break;
              case 253:
#if ecps_debug_prints >= 1
                fan::print_throttled("RENDER: LibAV decoder changed");
#endif
                break;
              case 252:
#if ecps_debug_prints >= 1
                fan::print_throttled("RENDER: LibAV decoder not readable");
#endif
                break;
              case 251:
#if ecps_debug_prints >= 1
                fan::print_throttled("RENDER: LibAV decode failed");
#endif
                break;
              case 250:
#if ecps_debug_prints >= 1
                fan::print_throttled("RENDER: LibAV unsupported stride");
#endif
                break;
              case 249:
#if ecps_debug_prints >= 1
                fan::print_throttled("RENDER: LibAV unsupported pixel format");
#endif
                break;
              default:
#if ecps_debug_prints >= 1
                fan::print_throttled_format("RENDER: LibAV unknown error type: {}", node.type);
#endif
                break;
              }
            }

            render_thread_instance.FrameList.unlrec(flnr);

            if (!frame_valid) {
              static auto last_idr_request = std::chrono::steady_clock::now();
              auto now = std::chrono::steady_clock::now();
              if (std::chrono::duration_cast<std::chrono::seconds>(now - last_idr_request).count() > 3) {
                render_thread_instance.ecps_gui.backend_queue([=]() -> fan::event::task_t {
                  try {
                    for (const auto& channel : ecps_backend.channel_info) {
                      if (channel.is_viewing) {
                        ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
                        rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
                        rest.ChannelID = channel.channel_id;
                        co_await ecps_backend.tcp_write(
                          ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
                          &rest,
                          sizeof(rest)
                        );
#if ecps_debug_prints >= 1
                        fan::print_throttled_format("RENDER: Requesting LibAV IDR for channel {} due to invalid frame", channel.channel_id.i);
#endif
                      }
                    }
                  }
                  catch (...) {}
                  });
                last_idr_request = now;
              }
            }
          }
        }
      }

      render_thread_instance.render([] {
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

        auto* rt = render_thread_ptr.load(std::memory_order_acquire);
        if (!rt) return;

        std::vector<fan::graphics::image_t> img_list;
        img_list.emplace_back(rt->network_frame.get_image());
        auto images = rt->network_frame.get_images();
        img_list.insert(img_list.end(), images.begin(), images.end());
      });

      std::this_thread::yield();
    }

    render_thread_ptr.store(nullptr, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(render_thread_mutex);
      render_thread_ready.store(false, std::memory_order_release);
    }
    });

    render_thread_future.wait();

    fan::event::thread_create([] {
      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      while (!rt || !rt->screen_encoder.encoder_.is_initialized()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        rt = render_thread_ptr.load(std::memory_order_acquire);
      }

      uint64_t frame_id = 0;
      bool first_frame = true;
      auto last_idr_time = std::chrono::steady_clock::now();

      while (!rt->should_stop.load()) {
        rt = render_thread_ptr.load(std::memory_order_acquire);
        if (!rt) break;

        if (ecps_backend.is_streaming_to_any_channel()) {
          if (first_frame) {
            rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
            first_frame = false;
            last_idr_time = std::chrono::steady_clock::now();
          }

          // Force IDR frame periodically (every 2 seconds)
          auto now = std::chrono::steady_clock::now();
          auto time_since_idr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_idr_time);
          if (time_since_idr > std::chrono::milliseconds(2000)) {
            rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
            last_idr_time = now;
          }

          // Read screen data
          if (!rt->screen_encoder.screen_read()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }

          if (rt->ecps_gui.show_own_stream && rt->screen_encoder.screen_buffer) {
            uint32_t width = rt->screen_encoder.mdscr.Geometry.Resolution.x;
            uint32_t height = rt->screen_encoder.mdscr.Geometry.Resolution.y;

            if (width > 0 && height > 0) {
              rt->ecps_gui.backend_queue([=]() -> fan::event::task_t {
                auto* current_rt = render_thread_ptr.load(std::memory_order_acquire);
                if (!current_rt) co_return;

                try {
                  fan::graphics::image_load_properties_t props;
                  props.format = fan::graphics::image_format::b8g8r8a8_unorm;
                  props.visual_output = fan::graphics::image_filter::linear;

                  fan::image::info_t image_info;
                  image_info.data = current_rt->screen_encoder.screen_buffer;
                  image_info.size = fan::vec2ui(width, height);

                  current_rt->engine.image_reload(current_rt->screen_image, image_info, props);
                }
                catch (const std::exception& e) {
                  fan::print("LOCAL: Direct sprite update failed: " + std::string(e.what()));
                }
                co_return;
                });
            }
          }

          if (!rt->screen_encoder.encode_write()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }

          uint8_t* encoded_data = nullptr;
          size_t encoded_size = rt->screen_encoder.encode_read(&encoded_data);

          if (encoded_size > 0 && encoded_data) {
            std::unique_lock<std::timed_mutex> frame_list_lock(ecps_backend.share.frame_list_mutex, std::try_to_lock);
            if (frame_list_lock.owns_lock()) {
              bool is_idr = (rt->screen_encoder.encode_write_flags & fan::graphics::codec_update_e::force_keyframe) != 0;

              // Clear old frames for IDR
              if (is_idr) {
                while (ecps_backend.share.m_NetworkFlow.FrameList.Usage() > 0) {
                  auto old = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
                  ecps_backend.share.m_NetworkFlow.FrameList.unlrec(old);
                }
              }

              auto flnr = ecps_backend.share.m_NetworkFlow.FrameList.NewNodeLast();
              auto f = &ecps_backend.share.m_NetworkFlow.FrameList[flnr];
              f->vec.resize(encoded_size);
              std::memcpy(f->vec.data(), encoded_data, encoded_size);
              f->SentOffset = 0;
            }
          }

          rt->screen_encoder.sleep_thread();
        }
        else {
          first_frame = true;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      });

    fan::event::thread_create([] {
      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      while (!rt || !rt->screen_decoder.is_initialized()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        rt = render_thread_ptr.load(std::memory_order_acquire);
      }

      auto last_successful_decode = std::chrono::steady_clock::now();
      uint64_t consecutive_decode_failures = 0;
      uint64_t successful_decodes = 0;
      bool decoder_needs_reset = false;
      bool has_processed_any_frame = false;

      while (!rt->should_stop.load()) {
        rt = render_thread_ptr.load(std::memory_order_acquire);
        if (!rt) break;

        bool processed_frame = false;
        auto now = std::chrono::steady_clock::now();
        auto time_since_success = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_successful_decode);

        // Check if decoder needs reset
        if (has_processed_any_frame &&
          (consecutive_decode_failures > 30 ||
            (time_since_success > std::chrono::milliseconds(10000) && successful_decodes > 0))) {
          decoder_needs_reset = true;
        }

        bool should_process_network = !rt->ecps_gui.show_own_stream ||
          !ecps_backend.is_channel_streaming(rt->ecps_gui.window_handler.selected_channel_id);

        if (!processed_frame) {
          if (!has_processed_any_frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        }
      }
      });

  auto motion_idr_task = fan::event::task_timer(dynamic_config_t::get_adaptive_motion_poll_ms(), []() -> fan::event::task_value_resume_t<bool> {
    static uint64_t last_frame_count = 0;
    static int motion_frames = 0;
    static auto startup_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto time_since_startup = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);
    uint32_t fps = dynamic_config_t::get_target_framerate();
    uint32_t startup_duration = fps >= 120 ? 4 : (fps >= 60 ? 6 : 8);

    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (rt && ecps_backend.is_streaming_to_any_channel()) {
      uint64_t current_frame_count = rt->screen_encoder.frame_timestamp_;
      uint32_t poll_ms = dynamic_config_t::get_adaptive_motion_poll_ms();
      uint64_t frames_in_poll = current_frame_count - last_frame_count;
      if (time_since_startup.count() > startup_duration) {
        uint32_t motion_threshold = (poll_ms * fps / 1000) * 3 / 2;
        if (frames_in_poll > motion_threshold) {
          motion_frames++;
          uint32_t motion_trigger = fps >= 120 ? 8 : (fps >= 60 ? 6 : 4);
          if (motion_frames >= motion_trigger) {
            rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
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
      startup_time = now;
    }

    co_return 0;
    });

  fan::event::task_idle([]() -> fan::event::task_t {
    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (!rt || rt->should_stop.load()) {
      co_return;
    }

    constexpr size_t MAX_BATCH_SIZE = 4;
    std::vector<std::function<fan::event::task_t()>> local_tasks;

    {
      std::unique_lock<std::timed_mutex> task_lock(render_mutex, std::defer_lock);
      if (task_lock.try_lock_for(std::chrono::milliseconds(5)) && !rt->ecps_gui.task_queue.empty()) {
        size_t batch_size = std::min(rt->ecps_gui.task_queue.size(), MAX_BATCH_SIZE);
        local_tasks.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
          local_tasks.emplace_back(std::move(rt->ecps_gui.task_queue[i]));
        }

        rt->ecps_gui.task_queue.erase(
          rt->ecps_gui.task_queue.begin(),
          rt->ecps_gui.task_queue.begin() + batch_size
        );

        if (rt->ecps_gui.task_queue.empty()) {
          rt->has_task_work.store(false);
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

  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  while (!rt) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  rt = render_thread_ptr.load(std::memory_order_acquire);
}

  rt = render_thread_ptr.load(std::memory_order_acquire);
while (rt && rt->should_stop.load()) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  rt = render_thread_ptr.load(std::memory_order_acquire);
  continue;
}

  auto network_task = fan::event::task_timer(1, []() -> fan::event::task_value_resume_t<bool> {
    /*if (ecps_backend.cleanup_disconnected_channels()) {
      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      if (rt) {
        rt->ecps_gui.window_handler.main_tab = 0;
      }
    }*/
    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (rt) {
      if (rt->ecps_gui.window_handler.auto_refresh) {
        static fan::time::clock refresh_timer(rt->ecps_gui.window_handler.refresh_interval * 2e+9, true);
        if (refresh_timer.finished()) {
          rt->ecps_gui.backend_queue([]() -> fan::event::task_t {
            try {
              co_await ecps_backend.request_channel_list();
            }
            catch (...) {}
            });
          refresh_timer.restart();
        }
      }
    }

    if (ecps_backend.channel_info.empty()) {
      co_return 0;
    }

    static uint64_t skip_counter = 0;
    skip_counter++;

    if (skip_counter % 5 != 0) {
      auto flnr_check = ecps_backend.share.m_NetworkFlow.FrameList.GetNodeFirst();
      if (flnr_check == ecps_backend.share.m_NetworkFlow.FrameList.dst) {
        co_return 0;
      }
    }

    std::unique_lock<std::timed_mutex> frame_list_lock(ecps_backend.share.frame_list_mutex, std::try_to_lock);
    if (!frame_list_lock.owns_lock()) {
      co_return 0;
    }

    uint64_t ctime = fan::event::now();
    uint64_t DeltaTime = ctime - ecps_backend.share.m_NetworkFlow.TimerLastCallAt;

    if (DeltaTime < 100000) {
      frame_list_lock.unlock();
      co_return 0;
    }

    ecps_backend.share.m_NetworkFlow.TimerLastCallAt = ctime;

    f32_t bucket_multiplier = dynamic_config_t::get_adaptive_bucket_multiplier();

    if (!rt) {
      co_return 0;
    }

    double seconds_elapsed = static_cast<double>(DeltaTime) / 1000000000.0;
    double bits_to_add = seconds_elapsed * rt->screen_encoder.config_.bitrate * bucket_multiplier;

    ecps_backend.share.m_NetworkFlow.Bucket += static_cast<uint64_t>(bits_to_add);

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

    size_t max_chunks = dynamic_config_t::get_adaptive_chunk_count();

    size_t affordable_chunks = ecps_backend.share.m_NetworkFlow.Bucket / (0x400 * 8);

    if (is_likely_keyframe) {
      max_chunks *= 2;
      affordable_chunks = (ecps_backend.share.m_NetworkFlow.Bucket + ecps_backend.share.m_NetworkFlow.BucketSize * 0.1) / (0x400 * 8);
    }

    size_t chunks_to_send = std::min({
        static_cast<size_t>(Possible - sent_offset),
        max_chunks,
        affordable_chunks
      });

    if (is_likely_keyframe && chunks_to_send == 0 && sent_offset < Possible) {
      chunks_to_send = 1;
    }

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

      ecps_backend.share.m_NetworkFlow.Bucket -= DataSize * 8;
    }

    f->SentOffset = sent_offset;

    if (sent_offset >= Possible) {
      f->vec.clear();
      f->vec.shrink_to_fit();
      ecps_backend.share.m_NetworkFlow.FrameList.unlrec(flnr);
      ++ecps_backend.share.frame_index;

#if ecps_debug_prints >= 2
      static uint64_t completed_frames = 0;
      if (++completed_frames % 30 == 0) {
        printf("NETWORK: Completed frame %llu (%s)\n",
          ecps_backend.share.frame_index - 1,
          is_likely_keyframe ? "I-frame" : "P-frame");
      }
#endif
    }

    co_return 0;
    });

  fan::event::loop();

  auto* rt_final = render_thread_ptr.load(std::memory_order_acquire);
  if (rt_final) {
    rt_final->should_stop.store(true);
    rt_final->frame_cv.notify_all();
    rt_final->task_cv.notify_all();
  }

  std::exit(0);
}