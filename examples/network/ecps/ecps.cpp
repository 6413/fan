#include <fan/utility.h>
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
#include <cmath>

#include <WITCH/WITCH.h>
#include <WITCH/MD/Mice.h>
#include <WITCH/MD/Keyboard/Keyboard.h>

extern "C" {
#include <libavutil/pixfmt.h>
}

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include <vector>
#include <stdexcept>

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

  static uint32_t get_adaptive_bucket_size() {
    uint32_t fps = get_target_framerate();
    uint32_t base_bitrate = get_adaptive_bitrate();
    if (fps >= 144) return base_bitrate * 4;
    else if (fps >= 120) return base_bitrate * 3;
    else if (fps >= 60) return base_bitrate * 2;
    else return base_bitrate;
  }

  static uint64_t get_adaptive_bitrate();

  static size_t get_adaptive_pool_size() {
    uint32_t fps = get_target_framerate();
    if (fps >= 144) return 320;
    else if (fps >= 120) return 256;
    else if (fps >= 90) return 192;
    else if (fps >= 60) return 128;
    else return 64;
  }

  static size_t get_adaptive_queue_size() {
    uint32_t fps = get_target_framerate();
    if (fps >= 144) return 10;
    else if (fps >= 120) return 8;
    else if (fps >= 90) return 6;
    else if (fps >= 60) return 4;
    else return 3;
  }

  static uint32_t get_adaptive_motion_poll_ms() {
    uint32_t fps = get_target_framerate();
    if (fps >= 144) return 25;
    else if (fps >= 120) return 30;
    else if (fps >= 90) return 40;
    else if (fps >= 60) return 50;
    else return 80;
  }


  static size_t get_adaptive_chunk_count() {
    uint32_t fps = get_target_framerate();
    if (fps >= 144) return 50;
    else if (fps >= 120) return 40;
    else if (fps >= 90) return 30;
    else if (fps >= 60) return 25;
    else return 20;
  }

  static f32_t get_adaptive_bucket_multiplier() {
    uint32_t fps = get_target_framerate();
    if (fps >= 144) return 10.0f;
    else if (fps >= 120) return 8.0f;
    else if (fps >= 90) return 6.5f;
    else if (fps >= 60) return 5.0f;
    else return 3.5f;
  }
};

struct webrtc_stream_t {
  std::vector<fan::network::tcp_id_t> peer_connections;
  std::atomic<bool> enabled{ true };
  std::mutex peers_mutex;

  void push_frame(const std::vector<uint8_t>& h264_data,
    bool is_keyframe,
    bool has_sps_pps,
    int width,
    int height,
    int64_t pts90k,
    int64_t dts90k) {

    if (!enabled.load()) return;

    std::lock_guard<std::mutex> lock(peers_mutex);

    send_frame_to_peer(h264_data, is_keyframe, has_sps_pps, pts90k);
  }

  void add_peer(fan::network::tcp_id_t peer_id) {
    std::lock_guard<std::mutex> lock(peers_mutex);
    peer_connections.push_back(peer_id);
  }

  void remove_peer(fan::network::tcp_id_t peer_id) {
    std::lock_guard<std::mutex> lock(peers_mutex);
    peer_connections.erase(
      std::remove(peer_connections.begin(), peer_connections.end(), peer_id),
      peer_connections.end()
    );
  }

  // Send raw H.264 data instead of JSON
  void send_frame_to_peer(const std::vector<uint8_t>& data, bool keyframe, bool has_sps_pps, int64_t pts);

  void broadcast_to_websocket_clients(const std::string& message);
};

struct websocket_connection_t {
  fan::network::tcp_id_t socket_id;
  bool is_webrtc_peer = false;
};

struct websocket_server_t {
  std::vector<websocket_connection_t> connections;
  std::mutex connections_mutex;
  std::atomic<uint32_t> next_connection_id{ 0 };

  fan::event::task_t broadcast_async(const std::string& message) {
    std::vector<fan::event::task_t> send_tasks;
    {
      std::lock_guard<std::mutex> lock(connections_mutex);
      for (auto& conn : connections) {
        if (conn.is_webrtc_peer) {
          send_tasks.push_back(send_websocket_frame(conn.socket_id, message));
        }
      }
    }
    
    for (auto& task : send_tasks) {
      co_await task;
    }
    co_return;
  }

  void broadcast(const std::string& message) {
    static fan::event::idle_id_t id;
    id = fan::event::task_idle([this, message]() -> fan::event::task_t {
      try {
        co_await broadcast_async(message);
      }
      catch (const std::exception& e) {
        fan::print("error:", e.what());
      }
      catch (const fan::exception_t& e) {
        fan::print("error:", e.reason);
      }
      catch (...) {
      fan::print("error");
    }
      fan::event::idle_stop(id);
      co_return;
    });
  }

  void add_connection(fan::network::tcp_id_t socket_id) {
    std::string conn_id = "peer_" + std::to_string(next_connection_id.fetch_add(1));
    std::lock_guard<std::mutex> lock(connections_mutex);
    
    connections.push_back(websocket_connection_t{
      .socket_id = socket_id,
      .is_webrtc_peer = true
    });
  }


  void remove_connection(fan::network::tcp_id_t conn_id) {
    std::lock_guard<std::mutex> lock(connections_mutex);
    connections.erase(
      std::remove_if(connections.begin(), connections.end(),
        [conn_id](const auto& conn) { return conn.socket_id == conn_id; }),
      connections.end()
    );
  }

void broadcast_binary_to_websocket_clients(const std::vector<uint8_t>& binary_data) {
    static fan::event::idle_id_t id;
    id = fan::event::task_idle([this, binary_data]() -> fan::event::task_t {
        try {
            std::vector<fan::event::task_t> send_tasks;
            {
                std::lock_guard<std::mutex> lock(connections_mutex);
                for (auto& conn : connections) {
                    if (conn.is_webrtc_peer) {
                        send_tasks.push_back(send_binary_websocket_frame(conn.socket_id, binary_data));
                    }
                }
            }
            
            for (auto& task : send_tasks) {
                co_await task;
            }
        } catch (const std::exception& e) {
            fan::print("binary broadcast error:", e.what());
        } catch (const fan::exception_t& e) {
            fan::print("binary broadcast error:", e.reason);
        } catch (...) {
            fan::print("binary broadcast unknown error");
        }
        fan::event::idle_stop(id);
        co_return;
    });
}

private:
fan::event::task_t send_binary_websocket_frame(fan::network::tcp_id_t socket_id, const std::vector<uint8_t>& data) {
     // fan::print("Sending binary frame:", data.size(), "bytes to socket:", socket_id.NRI);

   std::vector<uint8_t> frame;
   frame.push_back(0x82); // Binary frame, FIN bit set

   if (data.size() < 126) {
       frame.push_back(static_cast<uint8_t>(data.size()));
   }
   else if (data.size() < 65536) {
       frame.push_back(126);
       frame.push_back((data.size() >> 8) & 0xFF);
       frame.push_back(data.size() & 0xFF);
   }
   else {
       frame.push_back(127);
       for (int i = 7; i >= 0; i--) {
           frame.push_back((data.size() >> (i * 8)) & 0xFF);
       }
   }

   frame.insert(frame.end(), data.begin(), data.end());
   
   co_await fan::network::get_client_handler()[socket_id].write_raw(
       fan::network::buffer_t(frame.begin(), frame.end()));
   co_return;
}
  fan::event::task_t send_websocket_frame(fan::network::tcp_id_t socket_id, const std::string& data) {
    std::vector<uint8_t> frame;
    frame.push_back(0x81); // Text frame, FIN bit set

    if (data.size() < 126) {
      frame.push_back(static_cast<uint8_t>(data.size()));
    }
    else if (data.size() < 65536) {
      frame.push_back(126);
      frame.push_back((data.size() >> 8) & 0xFF);
      frame.push_back(data.size() & 0xFF);
    }
    else {
      frame.push_back(127);
      for (int i = 7; i >= 0; i--) {
        frame.push_back((data.size() >> (i * 8)) & 0xFF);
      }
    }

    frame.insert(frame.end(), data.begin(), data.end());
    
    // This is the fix - await the write_raw call
    co_await fan::network::get_client_handler()[socket_id].write_raw(fan::network::buffer_t(frame.begin(), frame.end()));
    co_return;
  }
};

static websocket_server_t websocket_server;


void webrtc_stream_t::send_frame_to_peer(const std::vector<uint8_t>& data, bool keyframe, bool has_sps_pps, int64_t pts) {
    // Enhanced header: [1 byte flags][8 bytes pts][4 bytes frame size][data...]
    std::vector<uint8_t> frame;
    frame.reserve(data.size() + 13);
    
    uint8_t flags = (keyframe ? 0x01 : 0x00) | (has_sps_pps ? 0x02 : 0x00);
    if (keyframe) flags |= 0x04; // IDR frame marker
    
    frame.push_back(flags);
    
    // PTS as 8 bytes
    for (int i = 7; i >= 0; i--) {
        frame.push_back((pts >> (i * 8)) & 0xFF);
    }
    
    // Frame size as 4 bytes
    uint32_t frame_size = static_cast<uint32_t>(data.size());
    for (int i = 3; i >= 0; i--) {
        frame.push_back((frame_size >> (i * 8)) & 0xFF);
    }
    
    frame.insert(frame.end(), data.begin(), data.end());
    
    websocket_server.broadcast_binary_to_websocket_clients(frame);
}

void webrtc_stream_t::broadcast_to_websocket_clients(const std::string& message) {
  websocket_server.broadcast(message);
}

std::string WEBRTC_HTML;

struct render_thread_t {
  render_thread_t() {
    if (fan::io::file::read(fan::io::file::find_relative_path("webrtc.html"), &WEBRTC_HTML)) {
      fan::print_color(fan::colors::red, "failed to open html file");
    }
    
    window_icon = engine.image_load("icons/ecps_logo.png");

    engine.set_target_fps(0, false);
    engine.set_window_name("ECPS - Every Communication Program Sucks");
    engine.set_window_icon(window_icon);

    if (!screen_encoder.open()) {
      fan::print("Failed to open modern encoder");
    }
    screen_encoder.set_user_resolution(1280, 720);

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

  fan::graphics::image_t window_icon;

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
  f32_t displayed_fps = 0.0f;

  fan::vec2ui host_mouse_pos;

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

  webrtc_stream_t webrtc_stream;
  fan::network::http_server_t http_server;

  void setup_webrtc_routes() {
    http_server.get("/", [](const auto& req, auto& res) -> fan::event::task_t {
      res.html(WEBRTC_HTML);
      co_return;
      });

  
  }
  fan::event::task_t handle_websocket_frames(fan::network::tcp_id_t conn_id) {
    try {
      auto& client = fan::network::get_client_handler()[conn_id];
        while (true) {
          auto hdr = co_await client.read(2);
            if (hdr.status < 0) break;

            const uint8_t fin_opcode = hdr.buffer[0];
            const uint8_t mask_len   = hdr.buffer[1];

            bool fin    = (fin_opcode & 0x80) != 0;
            uint8_t opcode = fin_opcode & 0x0F;
            bool masked = (mask_len & 0x80) != 0;
            uint64_t payload_len = mask_len & 0x7F;

            // Extended payload length
            if (payload_len == 126) {
                auto ext = co_await client.read(2);
                if (ext.status < 0) break;
                payload_len = (ext.buffer[0] << 8) | ext.buffer[1];
            } else if (payload_len == 127) {
                auto ext = co_await client.read(8);
                if (ext.status < 0) break;
                payload_len = 0;
                for (int i = 0; i < 8; i++)
                    payload_len = (payload_len << 8) | ext.buffer[i];
            }

            // Masking key
            uint8_t mask_key[4] = {0};
            if (masked) {
                auto mask = co_await client.read(4);
                if (mask.status < 0) break;
                std::copy(mask.buffer.begin(), mask.buffer.end(), mask_key);
            }

            // Payload data
            auto payload = co_await client.read(payload_len);
            if (payload.status < 0) break;

            if (masked) {
                for (size_t i = 0; i < payload.buffer.size(); i++) {
                    payload.buffer[i] ^= mask_key[i % 4];
                }
            }

            // Handle opcodes
            if (opcode == 0x1) { // text frame
              std::string msg(payload.buffer.begin(), payload.buffer.end());
              try {
                auto json_msg = fan::json::parse(msg);
                if (json_msg["type"] == "request_keyframe") {
                  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
                  if (rt) {
                    rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
                  }
                }
              }
              catch (...) {
                fan::print("Text message from client:", msg);
              }
            }
            else if (opcode == 0x2) { // binary frame
              fan::print("Binary frame of size", payload.buffer.size());
              // TODO: handle binary data (e.g., video chunks)
            }
            else if (opcode == 0x8) { // close
                fan::print("Close frame received");
                break;
            }
            else if (opcode == 0x9) { // ping
                fan::print("Ping received, sending pong");
                // Send pong frame
                fan::network::buffer_t pong = {(char)0x8A, (char)0x00};
                co_await client.write_raw(pong);
            }
            else if (opcode == 0xA) { // pong
                fan::print("Pong received");
            }
        }
    }
    catch (...) { fan::print("Frame loop unknown error");}
    
    // Cleanup
    websocket_server.remove_connection(conn_id);
    webrtc_stream.remove_peer(conn_id);
    co_return;
}

#if 0
  fan::event::task_t start_websocket_server() {
    fan::network::tcp_t websocket_tcp;
    fan::print("Starting WebSocket server on port 9091...");
    
    co_await websocket_tcp.listen({"0.0.0.0", 9091}, [this](fan::network::tcp_t& client) -> fan::event::task_t {
        fan::print("WebSocket client connected!");
        try {
            // Read HTTP upgrade request
            std::string request_buffer;
            bool headers_done = false;
            
            while (!headers_done) {
                auto data = co_await client.read_raw();
                if (data.status < 0) {
                    fan::print("Failed to read from client");
                    co_return;
                }
                
                std::string chunk(data.buffer.begin(), data.buffer.end());
                request_buffer += chunk;
                
                if (request_buffer.find("\r\n\r\n") != std::string::npos) {
                    headers_done = true;
                }
            }
            
            fan::print("Request received:", request_buffer.substr(0, 200), "...");
            
            // Check for WebSocket upgrade
            if (request_buffer.find("Upgrade: websocket") != std::string::npos) {
                fan::print("WebSocket upgrade detected");
                
                std::string key = fan::network::extract_header(request_buffer, "Sec-WebSocket-Key");
                std::string accept_val = fan::network::websocket_accept_key(key);

                std::string response =
                  "HTTP/1.1 101 Switching Protocols\r\n"
                  "Upgrade: websocket\r\n"
                  "Connection: Upgrade\r\n"
                  "Sec-WebSocket-Accept: " + accept_val + "\r\n\r\n";

                co_await client.write_raw(response);
                fan::print("WebSocket handshake sent");
                
                websocket_server.add_connection(client);
                webrtc_stream.add_peer(client);
                
                fan::print("WebSocket connection established with ID:", client.nr.NRI);
                
                // Handle WebSocket frames
                co_await handle_websocket_frames(client);
            } else {
                fan::print("Not a WebSocket upgrade request");
                std::string error_response = 
                    "HTTP/1.1 400 Bad Request\r\n"
                    "Content-Length: 11\r\n"
                    "Connection: close\r\n\r\n"
                    "Bad Request";
                co_await client.write_raw(error_response);
            }
        }
        catch (const std::exception& e) {
            fan::print("WebSocket error:", e.what());
        }
        catch (const fan::exception_t& e) {
          fan::print("WebSocket error:", e.reason);
        }
        catch (...) {
          fan::print("WebSocket unknown error");
        }
        co_return;
    }, true);
}
#endif

  fan::event::task_t handle_websocket_connection(fan::network::tcp_id_t conn_id) {
    try {
      while (true) {
        auto data = co_await fan::network::get_client_handler()[conn_id].read_raw();
        if (data.status < 0) break;
      }
    }
    catch (...) {}

    websocket_server.remove_connection(conn_id);
    webrtc_stream.remove_peer(conn_id);
    co_return;
  }
};

  struct webrtc_rate_limiter_t {
    std::atomic<std::chrono::steady_clock::time_point> last_webrtc_send_;
    std::atomic<int> webrtc_frame_counter_;
    static constexpr int webrtc_max_fps_ = 50;
    static constexpr auto webrtc_frame_interval_ = std::chrono::microseconds(1000000 / webrtc_max_fps_);
    
    bool should_send_webrtc(bool is_keyframe) {
        auto now = std::chrono::steady_clock::now();
        auto last_time = last_webrtc_send_.load();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time);
        
        if (is_keyframe || elapsed >= webrtc_frame_interval_) {
            last_webrtc_send_.store(now);
            return true;
        }
        return false;
    }
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

  // input
  __dme_get(Protocol_S2C, Channel_ScreenShare_View_InformationToViewSetFlag) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_View_InformationToViewSetFlag_t>();

    for (auto& channel : backend.channel_info) {
      if (channel.channel_id.i == msg->ChannelID.i) {
        channel.flag = msg->Flag;
        break;
      }
    }
    co_return;
    };

  __dme_get(Protocol_S2C, Channel_ScreenShare_View_InformationToViewMouseCoordinate) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_View_InformationToViewMouseCoordinate_t>();

    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (rt) {
      backend.update_host_mouse_coordinate(msg->ChannelID, msg->pos);
    }
    co_return;
    };

  __dme_get(Protocol_S2C, Channel_ScreenShare_Share_ApplyToHostMouseCoordinate) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseCoordinate_t>();

    bool input_control_enabled = false;
    for (const auto& channel : backend.channel_info) {
      if (channel.channel_id.i == msg->ChannelID.i) {
        input_control_enabled = (channel.flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl) != 0;
        break;
      }
    }

    if (input_control_enabled) {
      MD_Mice_Error err = MD_Mice_Coordinate_Write(&mice, msg->pos.x, msg->pos.y);
      if (err != MD_Mice_Error_Success && err != MD_Mice_Error_Temporary) {
        fan::print("mouse coordinate write failed");
      }
    }
    co_return;
    };

  __dme_get(Protocol_S2C, Channel_ScreenShare_Share_ApplyToHostMouseMotion) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseMotion_t>();

    bool input_control_enabled = false;
    for (const auto& channel : backend.channel_info) {
      if (channel.channel_id.i == msg->ChannelID.i) {
        input_control_enabled = (channel.flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl) != 0;
        break;
      }
    }

    if (input_control_enabled) {
      MD_Mice_Error err = MD_Mice_Motion_Write(&mice, msg->Motion.x, msg->Motion.y);
      if (err != MD_Mice_Error_Success && err != MD_Mice_Error_Temporary) {
        fan::print("mouse motion write failed");
      }
    }
    co_return;
    };

  __dme_get(Protocol_S2C, Channel_ScreenShare_Share_ApplyToHostMouseButton) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseButton_t>();

    bool input_control_enabled = false;
    for (const auto& channel : backend.channel_info) {
      if (channel.channel_id.i == msg->ChannelID.i) {
        input_control_enabled = (channel.flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl) != 0;
        break;
      }
    }

    if (input_control_enabled) {
      if (msg->pos != fan::vec2i(-1)) {
        MD_Mice_Error err = MD_Mice_Coordinate_Write(&mice, msg->pos.x, msg->pos.y);
        if (err != MD_Mice_Error_Success && err != MD_Mice_Error_Temporary) {
          fan::print("mouse coordinate write failed");
        }
      }

      MD_Mice_Error err = MD_Mice_Button_Write(&mice, msg->key, msg->state);
      if (err != MD_Mice_Error_Success && err != MD_Mice_Error_Temporary) {
        fan::print("mouse button write failed");
      }
    }
    co_return;
    };

  __dme_get(Protocol_S2C, Channel_ScreenShare_Share_ApplyToHostKeyboard) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
    auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostKeyboard_t>();

    bool input_control_enabled = false;
    for (const auto& channel : backend.channel_info) {
      if (channel.channel_id.i == msg->ChannelID.i) {
        input_control_enabled = (channel.flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl) != 0;
        break;
      }
    }

    if (input_control_enabled) {
      MD_Keyboard_Error err = MD_Keyboard_WriteKey(&keyboard, msg->Scancode, msg->State);
      if (err == MD_Keyboard_Error_UnknownArgument) {
        fan::print("unknown keyboard scancode:", msg->Scancode);
      }
      else if (err != MD_Keyboard_Error_Success && err != MD_Keyboard_Error_Temporary) {
        fan::print("keyboard write failed");
      }
    }
    co_return;
    };

  MD_Mice_Error mice_err = MD_Mice_Open(&mice);
  if (mice_err != MD_Mice_Error_Success) {
    fan::print("Failed to initialize mouse device");
  }

  MD_Keyboard_Error keyboard_err = MD_Keyboard_open(&keyboard);
  if (keyboard_err != MD_Keyboard_Error_Success) {
    fan::print("Failed to initialize keyboard device");
  }
}

fan::event::task_t ecps_backend_t::default_s2c_cb(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
  co_await backend.tcp_client.read(backend.Protocol_S2C.NA(base.Command)->m_DSS);
  co_return;
}

void ecps_backend_t::update_host_mouse_coordinate(Protocol_ChannelID_t channel_id, const fan::vec2ui& pos) {
  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  if (!rt) return;

  for (auto& channel : channel_info) {
    if (channel.channel_id.i == channel_id.i && channel.is_viewing) {
      rt->host_mouse_pos = pos;
      break;
    }
  }
}

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

void ecps_backend_t::view_t::RequestKeyframe() {
  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  if (rt) {
    rt->ecps_gui.backend_queue([]() -> fan::event::task_t {
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
  }
}

void ecps_backend_t::view_t::WriteFramePacket() {
  if (m_Possible == 0) return;

  uint32_t FramePacketSize = (uint32_t)(this->m_Possible - 1) * 0x400 + this->m_ModuloSize;
  if (FramePacketSize < 32 || FramePacketSize > 0x500000) {
    this->m_stats.Frame_Drop++;
  #if ecps_debug_prints >= 1
    fan::print_throttled_format("NETWORK: Frame dropped - invalid size: {} bytes (valid range: 32 - 5MB)",
      FramePacketSize);
  #endif
    return;
  }

  if (this->m_data.size() < FramePacketSize) {
    this->m_stats.Frame_Drop++;
  #if ecps_debug_prints >= 1
    fan::print_throttled_format("NETWORK: Frame dropped - buffer underrun: need {} bytes, have {}",
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
        if (nal_type == 7 || nal_type == 8) {
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
  }
  else {
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
        std::unique_lock<std::timed_mutex> frame_lock(render_mutex, std::defer_lock);

        if (frame_lock.try_lock_for(std::chrono::milliseconds(5))) {

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
        }
        else {
          this->m_stats.Frame_Drop++;
        #if ecps_debug_prints >= 1
          fan::print_throttled("NETWORK: Frame dropped due to render lock timeout (deadlock prevention)");
        #endif
        }
      }
      else {
      #if ecps_debug_prints >= 1
        fan::print_throttled_format("NETWORK: Decoded frame validation failed - data:{} dims:{}x{} stride:{}",
          has_valid_data, decode_result.image_size.x,
          decode_result.image_size.y, decode_result.stride[0].x);
      #endif
        this->m_stats.Frame_Drop++;
      }
    }
    else {
    #if ecps_debug_prints >= 1
      fan::print_throttled_format("NETWORK: Decode failed with type: {}", decode_result.type);
    #endif
      this->m_stats.Frame_Drop++;
    }
  }
  catch (const std::exception& e) {
  #if ecps_debug_prints >= 1
    fan::print_throttled_format("NETWORK: Decode exception: {}", e.what());
  #endif
    this->m_stats.Frame_Drop++;
  }

  frame_index++;
}

void ecps_backend_t::view_t::FixFrameOnComplete() {
  UpdateMissingPackets();
  if (m_MissingPackets.empty()) {
    WriteFramePacket();
    return;
  }

  m_stats.Frame_Drop++;
  m_Possible = (uint16_t)-1;
  RequestKeyframe();

  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  if (rt && m_MissingPackets.size() > m_Possible / 4) {
    rt->screen_decoder.reload_codec_cb();
  }
}

uint32_t dynamic_config_t::get_target_framerate() {
  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  if (rt && rt->screen_encoder.config_.frame_rate > 0) {
    return rt->screen_encoder.config_.frame_rate;
  }
  return 60;
}

uint64_t dynamic_config_t::get_adaptive_bitrate() {
  auto* rt = render_thread_ptr.load(std::memory_order_acquire);
  if (rt && rt->ecps_gui.stream_settings.bitrate_mode == 1) {
    return rt->ecps_gui.stream_settings.bitrate_mbps * 1000000;
  }

  uint64_t fps = get_target_framerate();
  if (fps >= 144) return 25000000;
  else if (fps >= 120) return 20000000;
  else if (fps >= 90) return 15000000;
  else if (fps >= 60) return 12000000;
  else return 8000000;
}

#if 0
fan::event::task_t start_both_servers() {
    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (!rt) co_return;

    rt->setup_webrtc_routes();

    co_await fan::event::when_all(
        rt->http_server.listen({"0.0.0.0", 9090}),
        rt->start_websocket_server()
    );
}

#endif
int main() {

  ecps_backend.login_fail_cb = [](fan::exception_t e) {
    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (rt) {// might be able to be removed
      rt->screen_decoder.graphics_queue_callback([e] {
        fan::printcl("failed to connect to server:"_str + e.reason + ", retrying...");
        });
    }
    };
  ecps_backend.view.recovery_callback = [](std::vector<uint8_t> request_data) {
    auto* rt = get_render_thread();
    if (rt) {
      rt->ecps_gui.backend_queue([request_data = std::move(request_data)]() -> fan::event::task_t {
        try {
          ecps_backend_t::ProtocolUDP::C2S_t::Channel_ScreenShare_RecoveryRequest_t rest;
          rest.ChannelID = ecps_backend_t::Protocol_ChannelID_t(0);
          rest.ChannelSessionID = ecps_backend_t::Protocol_ChannelSessionID_t(0);

          for (const auto& channel : ecps_backend.channel_info) {
            if (channel.is_viewing) {
              rest.ChannelID = channel.channel_id;
              rest.ChannelSessionID = channel.session_id;
              break;
            }
          }

          co_await ecps_backend.udp_write(
            0,
            ecps_backend_t::ProtocolUDP::C2S_t::Channel_ScreenShare_RecoveryRequest,
            rest,
            request_data.data(),
            request_data.size()
          );
        }
        catch (...) {}
        });
    }
    };

  ecps_backend.share.resend_packet_callback = [](const std::vector<uint8_t>& packet_data) {
    auto* rt = get_render_thread();
    if (rt) {
      rt->ecps_gui.backend_queue([packet_data]() -> fan::event::task_t {
        try {
          ecps_backend_t::ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData_t rest;
          for (const auto& channel : ecps_backend.channel_info) {
            if (channel.is_streaming) {
              rest.ChannelID = channel.channel_id;
              rest.ChannelSessionID = channel.session_id;
              break;
            }
          }

          co_await ecps_backend.udp_write(
            0,
            ecps_backend_t::ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData,
            rest,
            packet_data.data(),
            packet_data.size()
          );
        }
        catch (...) {}
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

            {
              static fan::time::timer display_fps_timer;
              static int display_frame_count = 0;
              static float display_fps = 0.0f;
              static bool timer_started = false;

              if (!timer_started) {
                display_fps_timer.start();
                timer_started = true;
              }

              display_frame_count++;

              if (display_fps_timer.elapsed() >= 1e+9) {
                display_fps = static_cast<float>(display_frame_count) * 1e+9 / display_fps_timer.elapsed();
                display_frame_count = 0;
                display_fps_timer.restart();
              }

              render_thread_instance.displayed_fps = display_fps;
            }

            auto& node = render_thread_instance.FrameList[flnr];
            bool frame_valid = false;
          #if ecps_debug_prints >= 2
            printf("RENDER: Frame displayed at %lld ms\n",
              std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
          #endif

            if (ecps_backend.did_just_join && !ecps_backend.is_current_user_host_of_channel(ecps_backend.channel_info.back().channel_id)) {
              auto* rt = render_thread_ptr.load(std::memory_order_acquire);
              if (rt) {
                rt->ecps_gui.window_handler.main_tab = 1;
              }
              ecps_backend.did_just_join = false;
            }

            if (node.type == 0) {
              // Hardware decoded frame (CUDA/NVENC) - if supported
            }
            else if (node.type == 1) {
              bool has_valid_data = false;
              bool dims_valid = node.image_size.x > 0 && node.image_size.y > 0 &&
                node.image_size.x <= 7680 && node.image_size.y <= 4320;
              bool stride_valid = node.stride[0].x >= node.image_size.x && node.stride[0].x > 0;

              if (node.pixel_format == fan::graphics::image_format::yuv420p) {
                has_valid_data = !node.data[0].empty() && !node.data[1].empty() && !node.data[2].empty();

              #if ecps_debug_prints >= 3
                static uint64_t yuv420_count = 0;
                if (++yuv420_count % 60 == 1) {
                  fan::print_throttled_format("RENDER: YUV420P frame - Y:{} U:{} V:{}",
                    node.data[0].size(), node.data[1].size(), node.data[2].size());
                }
              #endif
              }
              else if (node.pixel_format == fan::graphics::image_format::nv12) {
                has_valid_data = !node.data[0].empty() && !node.data[1].empty();

              #if ecps_debug_prints >= 3
                static uint64_t nv12_count = 0;
                if (++nv12_count % 60 == 1) {
                  fan::print_throttled_format("RENDER: NV12 frame - Y:{} UV:{}",
                    node.data[0].size(), node.data[1].size());
                }
              #endif
              }
              else {
                has_valid_data = !node.data[0].empty();

              #if ecps_debug_prints >= 1
                fan::print_throttled_format("RENDER: Unknown pixel format {}, trying Y-plane only",
                  node.pixel_format);
              #endif
              }

              if (has_valid_data && dims_valid && stride_valid) {
                f32_t sx = (f32_t)node.image_size.x / node.stride[0].x;
                if (sx > 0 && sx <= 1.0f) {
                  std::array<void*, 4> raw_ptrs = { nullptr, nullptr, nullptr, nullptr };

                  if (node.pixel_format == fan::graphics::image_format::yuv420p) {
                    raw_ptrs[0] = static_cast<void*>(node.data[0].data());
                    raw_ptrs[1] = static_cast<void*>(node.data[1].data());
                    raw_ptrs[2] = static_cast<void*>(node.data[2].data());
                    raw_ptrs[3] = nullptr;
                  }
                  else if (node.pixel_format == fan::graphics::image_format::nv12) {
                    raw_ptrs[0] = static_cast<void*>(node.data[0].data());
                    raw_ptrs[1] = static_cast<void*>(node.data[1].data());
                    raw_ptrs[2] = nullptr;
                    raw_ptrs[3] = nullptr;
                  }
                  else {
                    raw_ptrs[0] = static_cast<void*>(node.data[0].data());
                  }

                  bool pointers_valid = false;
                  if (node.pixel_format == fan::graphics::image_format::yuv420p) {
                    pointers_valid = (raw_ptrs[0] && raw_ptrs[1] && raw_ptrs[2]);
                  }
                  else if (node.pixel_format == fan::graphics::image_format::nv12) {
                    pointers_valid = (raw_ptrs[0] && raw_ptrs[1]);
                  }
                  else {
                    pointers_valid = (raw_ptrs[0] != nullptr);
                  }

                  if (pointers_valid) {
                    try {
                      render_thread_instance.network_frame.set_tc_size(fan::vec2(sx, 1));
                      render_thread_instance.network_frame.reload(
                        node.pixel_format,
                        raw_ptrs.data(),
                        fan::vec2ui(node.stride[0].x, node.image_size.y)
                      );
                      frame_valid = true;

                    #if ecps_debug_prints >= 2
                      static uint64_t render_count = 0;
                      if (++render_count % 30 == 1) {
                        const char* format_name = (node.pixel_format == fan::graphics::image_format::yuv420p) ? "YUV420P" :
                          (node.pixel_format == fan::graphics::image_format::nv12) ? "NV12" : "Unknown";
                        fan::print_throttled_format("RENDER: {} frame processed {}x{}, stride={}, sx={} ({})",
                          format_name, node.image_size.x, node.image_size.y, node.stride[0].x, sx, render_count);
                      }
                    #endif
                    }
                    catch (const std::exception& e) {
                    #if ecps_debug_prints >= 1
                      const char* format_name = (node.pixel_format == fan::graphics::image_format::yuv420p) ? "YUV420P" :
                        (node.pixel_format == fan::graphics::image_format::nv12) ? "NV12" : "Unknown";
                      fan::print_throttled_format("RENDER: {} reload failed for LibAV frame: {}", format_name, e.what());
                    #endif
                    }
                  }
                  else {
                  #if ecps_debug_prints >= 1
                    fan::print_throttled_format("RENDER: Invalid pointers for format {} - Y:{} UV/U:{} V:{}",
                      node.pixel_format, (void*)raw_ptrs[0], (void*)raw_ptrs[1], (void*)raw_ptrs[2]);
                  #endif
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
                const char* format_name = (node.pixel_format == fan::graphics::image_format::yuv420p) ? "YUV420P" :
                  (node.pixel_format == fan::graphics::image_format::nv12) ? "NV12" : "Unknown";
                fan::print_throttled_format("RENDER: {} validation failed - data:{} dims:{}x{} stride:{}",
                  format_name, has_valid_data, node.image_size.x, node.image_size.y, node.stride[0].x);
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
    webrtc_rate_limiter_t webrtc_limiter{};

    while (!rt->should_stop.load()) {
        rt = render_thread_ptr.load(std::memory_order_acquire);
        if (!rt) break;

        if (ecps_backend.is_streaming_to_any_channel()) {
            if (first_frame) {
                rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
                first_frame = false;
                last_idr_time = std::chrono::steady_clock::now();
            }

            auto now = std::chrono::steady_clock::now();
            auto time_since_idr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_idr_time);
            if (time_since_idr > std::chrono::milliseconds(5000)) {
                rt->screen_encoder.encode_write_flags |= fan::graphics::codec_update_e::force_keyframe;
                last_idr_time = now;
            }

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
                        } catch (...) {}
                        co_return;
                    });
                }
            }

            if (!rt->screen_encoder.encode_write()) {
                rt->screen_encoder.sleep_thread();
                continue;
            }

            uint8_t* encoded_data = nullptr;
            size_t encoded_size = rt->screen_encoder.encode_read(&encoded_data);

            if (encoded_size > 0 && encoded_data) {
                static std::vector<uint8_t> last_sps;
                static std::vector<uint8_t> last_pps;
                static int64_t pts90k = 0;
                static int64_t dts90k = 0;
                
                const int fps = rt->screen_encoder.config_.frame_rate > 0 ? rt->screen_encoder.config_.frame_rate : 30;
                const int64_t frame_duration = 90000 / fps;

                auto find_nal_units = [&](const uint8_t* data, size_t size) {
                    std::vector<size_t> pos;
                    for (size_t i = 0; i + 4 <= size; ++i) {
                        if (data[i] == 0x00 && data[i+1] == 0x00 && data[i+2] == 0x00 && data[i+3] == 0x01) {
                            pos.push_back(i);
                            i += 3;
                        }
                    }
                    pos.push_back(size);
                    return pos;
                };

                auto nal_pos = find_nal_units(encoded_data, encoded_size);
                bool is_keyframe = false;
                
                for (size_t k = 0; k + 1 < nal_pos.size(); ++k) {
                    const size_t start = nal_pos[k];
                    const uint8_t nal_type = encoded_data[start + 4] & 0x1F;
                    if (nal_type == 7) {
                        last_sps.assign(encoded_data + start, encoded_data + nal_pos[k+1]);
                    } else if (nal_type == 8) {
                        last_pps.assign(encoded_data + start, encoded_data + nal_pos[k+1]);
                    } else if (nal_type == 5) {
                        is_keyframe = true;
                    }
                }

                bool has_webrtc_connections = false;
                {
                    std::lock_guard<std::mutex> lock(websocket_server.connections_mutex);
                    has_webrtc_connections = !websocket_server.connections.empty();
                }

                if (has_webrtc_connections && webrtc_limiter.should_send_webrtc(is_keyframe)) {
                    std::vector<uint8_t> out_frame;
                    bool has_sps_pps = false;

                    if (is_keyframe && !last_sps.empty() && !last_pps.empty()) {
                        out_frame.insert(out_frame.end(), last_sps.begin(), last_sps.end());
                        out_frame.insert(out_frame.end(), last_pps.begin(), last_pps.end());
                        out_frame.insert(out_frame.end(), encoded_data, encoded_data + encoded_size);
                        has_sps_pps = true;
                    } else {
                        out_frame.assign(encoded_data, encoded_data + encoded_size);
                    }
                        //fan::print("Sending WebRTC frame:", out_frame.size(), "bytes, keyframe:", is_keyframe);

                    rt->webrtc_stream.push_frame(out_frame, is_keyframe, has_sps_pps, 
                        rt->screen_encoder.config_.width, rt->screen_encoder.config_.height, pts90k, dts90k);
                }

                pts90k += frame_duration;
                dts90k += frame_duration;

                std::unique_lock<std::timed_mutex> frame_list_lock(ecps_backend.share.frame_list_mutex, std::try_to_lock);
                if (frame_list_lock.owns_lock()) {
                    bool is_idr = (rt->screen_encoder.encode_write_flags & fan::graphics::codec_update_e::force_keyframe) != 0;

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
        } else {
            first_frame = true;
            const int fps = rt ? rt->screen_encoder.config_.frame_rate : 30;
            const auto sleep_time = std::chrono::microseconds(1000000 / fps);
            std::this_thread::sleep_for(sleep_time);
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
        !ecps_backend.is_channel_streaming(rt->ecps_gui.selected_channel_id);

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
    uint32_t startup_duration = fps >= 144 ? 3 : (fps >= 120 ? 4 : (fps >= 60 ? 6 : 8));

    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (rt && ecps_backend.is_streaming_to_any_channel()) {
      uint64_t current_frame_count = rt->screen_encoder.frame_timestamp_;
      uint32_t poll_ms = dynamic_config_t::get_adaptive_motion_poll_ms();
      uint64_t frames_in_poll = current_frame_count - last_frame_count;
      if (time_since_startup.count() > startup_duration) {
        uint32_t motion_threshold = (poll_ms * fps / 1000) * 3 / 2;
        if (frames_in_poll > motion_threshold) {
          motion_frames++;
          uint32_t motion_trigger = fps >= 144 ? 10 : (fps >= 120 ? 8 : (fps >= 60 ? 6 : 4));
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
    try {
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
    }
    catch (...) {
      fan::print("error");
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
        static fan::time::timer refresh_timer(rt->ecps_gui.window_handler.refresh_interval * 2e+9, true);
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

    size_t max_chunks = dynamic_config_t::get_adaptive_chunk_count();

    uint8_t Flag = 0;
    uint16_t Possible = (f->vec.size() / 0x400) + !!(f->vec.size() % 0x400);
    uint16_t sent_offset = f->SentOffset;
    bool is_likely_keyframe = f->vec.size() > 30000;
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
      size_t remaining_chunks = static_cast<size_t>(Possible - sent_offset);
      chunks_to_send = std::min(remaining_chunks, static_cast<size_t>(8));

      if (ecps_backend.share.m_NetworkFlow.Bucket < (0x400 * 8)) {
        ecps_backend.share.m_NetworkFlow.Bucket += (0x400 * 8 * chunks_to_send);
      }
    }

    if (is_likely_keyframe && chunks_to_send < 3 && sent_offset < Possible) {
      size_t remaining_chunks = static_cast<size_t>(Possible - sent_offset);
      chunks_to_send = std::min(remaining_chunks, static_cast<size_t>(3));
    }

    size_t chunks_sent = 0;
    bool transmission_failed = false;

    for (; sent_offset < Possible && chunks_sent < chunks_to_send; sent_offset++, chunks_sent++) {
      uintptr_t DataSize = f->vec.size() - sent_offset * 0x400;
      if (DataSize > 0x400) {
        DataSize = 0x400;
      }

      bool can_afford = true;
      if (!is_likely_keyframe) {
        can_afford = (ecps_backend.share.m_NetworkFlow.Bucket >= DataSize * 8);
      }
      else {
        int64_t deficit_threshold = static_cast<int64_t>(ecps_backend.share.m_NetworkFlow.BucketSize) * 0.1;
        can_afford = (static_cast<int64_t>(ecps_backend.share.m_NetworkFlow.Bucket) >=
          -deficit_threshold);
      }

      if (!can_afford) {
        if (is_likely_keyframe && chunks_sent == 0 && sent_offset == 0) {
        }
        else {
          break;
        }
      }

      bool ret = co_await ecps_backend.write_stream(sent_offset, Possible, Flag,
        &f->vec[sent_offset * 0x400], DataSize);
      if (ret != false) {
        transmission_failed = true;
        break;
      }

      if (is_likely_keyframe) {
        ecps_backend.share.m_NetworkFlow.Bucket =
          static_cast<int64_t>(ecps_backend.share.m_NetworkFlow.Bucket) - static_cast<int64_t>(DataSize * 8);
      }
      else {
        ecps_backend.share.m_NetworkFlow.Bucket -= DataSize * 8;
      }

      int64_t max_deficit = static_cast<int64_t>(ecps_backend.share.m_NetworkFlow.BucketSize) / -2; // -50% max deficit
      if (static_cast<int64_t>(ecps_backend.share.m_NetworkFlow.Bucket) < max_deficit) {
        ecps_backend.share.m_NetworkFlow.Bucket = static_cast<uint64_t>(max_deficit);
      }
    }

    if (is_likely_keyframe && !transmission_failed && chunks_sent > 0 && sent_offset < Possible) {
      size_t keyframe_completion = (chunks_sent * 100) / static_cast<size_t>(Possible);
      if (keyframe_completion < 30) {
      #if ecps_debug_prints >= 1
        fan::print_throttled_format("NETWORK: Incomplete keyframe transmission: {}% ({}/{})",
          keyframe_completion, chunks_sent, Possible);
      #endif

        static uint64_t failed_keyframe_count = 0;
        if (++failed_keyframe_count % 3 == 0) {
          ecps_backend.share.m_NetworkFlow.Bucket += static_cast<uint64_t>(
            static_cast<double>(ecps_backend.share.m_NetworkFlow.BucketSize) * 0.2
            );
        }
      }
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
    #if 0
  fan::event::task_t http_server_task;
  fan::event::task_idle([&http_server_task]() -> fan::event::task_t {
    static bool servers_started = false;
    if (servers_started) co_return;

    auto* rt = render_thread_ptr.load(std::memory_order_acquire);
    if (!rt) co_return;

    servers_started = true;

  
    // Start both servers
    http_server_task = start_both_servers();
    fan::print("HTTP server started on port 9090");
    fan::print("WebSocket server started on port 9091");
});
#endif

  fan::event::loop();

  auto* rt_final = render_thread_ptr.load(std::memory_order_acquire);
  if (rt_final) {
    rt_final->should_stop.store(true);
    rt_final->frame_cv.notify_all();
    rt_final->task_cv.notify_all();
  }

  std::exit(0);
}