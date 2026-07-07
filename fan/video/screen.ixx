module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#include <cstring>

#if defined(FAN_2D)
#include <fan/utility.h>
#include <WITCH/WITCH.h>
#include <WITCH/PR/PR.h>
#include <WITCH/MD/SCR/SCR.h>
#endif

#endif

export module fan.graphics.video.screen;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

import std;
import fan.graphics.video.codec;

#if defined(FAN_2D)
import fan.graphics.common_context;
import fan.graphics.shapes;

import fan.time;
import fan.print;
import fan.event;

export namespace fan {
  namespace graphics {

    struct screen_encode_t;

    struct resolution_manager_t {
      struct detected_info_t {
        std::uint32_t screen_width, screen_height;

        f32_t screen_aspect;
        resolution_system_t::resolution_t optimal_resolution;
        std::vector<resolution_system_t::resolution_t> matching_aspect_resolutions;
      };

      detected_info_t detected_info;
      bool auto_detected = false;

      void detect_and_set_optimal(screen_encode_t& encoder);

      std::vector<std::pair<std::string, f32_t>> get_aspect_ratio_options() {
        return {
          {"16:9", 16.0f / 9.0f},
          {"16:10", 16.0f / 10.0f},
          {"4:3", 4.0f / 3.0f},
          {"21:9", 21.0f / 9.0f},
          {"Auto (Screen Native)", detected_info.screen_aspect}
        };

      }
    };

    struct screen_encode_t {
      bool update_scaling_quality(codec_config_t::scaling_quality_e new_quality) {
        config_.scaling_quality = new_quality;

        std::lock_guard<std::mutex> lock(mutex);
        update_flags |= codec_update_e::scaling_quality;
        return true;
      }

      bool open(const codec_config_t& default_config = codec_config_t()) {
        mdscr_initialized_ = false;

        config_.frame_rate = 30;
        config_.codec = codec_config_t::H264;
        config_.rate_control = codec_config_t::VBR;
        config_.bitrate = 10000000;
        config_.width = 1280;
        config_.height = 720;

        if (!encoder_.open(config_, fan::vec2(config_.width, config_.height))) {
          fan::print_impl("Failed to open encoder");
          return false;

        }

        std::lock_guard<std::mutex> lock(mutex);
        update_flags = 0;
        return true;

      }

      bool ensure_mdscr_initialized() {
        if (mdscr_initialized_) return true;

        std::memset(&mdscr, 0, sizeof(mdscr));
        if (int ret = MD_SCR_open(&mdscr); ret != 0) {
          fan::print_impl("failed to open screen:" + std::to_string(ret));

          return false;
        }

        resolution_manager.detect_and_set_optimal(*this);

        if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) {
          fan::print_impl("Failed to reopen encoder with screen resolution");

          return false;
        }

        mdscr_initialized_ = true;
        return true;

      }

      ~screen_encode_t() { encoder_.close(); }
      bool is_initialized() const { return encoder_.is_initialized();

      }

      static std::vector<std::string> get_encoders() { return libav_encoder_t::get_available_encoders();

      }

      struct encode_data_t {
        std::vector<std::uint8_t> data;

        bool is_keyframe = false;
        std::int64_t timestamp = 0;
      };

      bool encode_write() {
        if (!ensure_mdscr_initialized()) return false;

        if (!user_set_resolution &&
          (config_.width != mdscr.Geometry.Resolution.x || config_.height != mdscr.Geometry.Resolution.y)) {
          resolution_manager.detect_and_set_optimal(*this);

          config_.width = mdscr.Geometry.Resolution.x;
          config_.height = mdscr.Geometry.Resolution.y;

          {
            std::lock_guard<std::mutex> lock(mutex);

            update_flags &= ~codec_update_e::codec;
          }

          if (!encoder_.open(config_, fan::vec2(mdscr.Geometry.Resolution.x, mdscr.Geometry.Resolution.y))) return false;

        }

        if (update_flags != 0) {
          std::lock_guard<std::mutex> lock(mutex);

          if (update_flags & codec_update_e::scaling_quality) {
            if (!encoder_.update_scaling_quality(config_.scaling_quality)) {
              fan::print_impl("Failed to update scaling quality");

            }
          }
          if (update_flags & ~codec_update_e::scaling_quality) {
            encoder_.update_config(config_, update_flags & ~codec_update_e::scaling_quality);

          }
          update_flags = 0;

        }

        if (!screen_buffer) {
          fan::print_impl("Error: screen_buffer is null");

          return false;
        }

        if (mdscr.Geometry.Resolution.x <= 0 || mdscr.Geometry.Resolution.y <= 0) {
          fan::print_impl("Error: invalid screen resolution");

          return false;
        }

        if (mdscr.Geometry.LineSize <= 0) {
          fan::print_impl("Error: invalid line size");

          return false;
        }

        int min_line_size = mdscr.Geometry.Resolution.x * 4;

        if (mdscr.Geometry.LineSize < min_line_size) {
          fan::print_impl("Error: LineSize too small for BGRA.");

          return false;
        }

        bool force_keyframe = (encode_write_flags & codec_update_e::force_keyframe) != 0;

        auto results = encoder_.encode_frame(screen_buffer, mdscr.Geometry.LineSize, frame_timestamp_, force_keyframe);
        
        if (encode_write_flags & codec_update_e::force_keyframe) encode_write_flags &= ~codec_update_e::force_keyframe;

        std::lock_guard<std::mutex> lock(data_mutex);

        for (const auto& result : results) {
          encode_data_t packet;

          packet.data = std::move(result.data);
          packet.is_keyframe = result.is_keyframe;
          packet.timestamp = result.pts;
          encoded_packets_.push_back(std::move(packet));
        }

        frame_timestamp_++;

        return !results.empty();
      }

      std::size_t encode_read(std::uint8_t** data) {
        std::lock_guard<std::mutex> lock(data_mutex);

        if (encoded_packets_.empty()) return 0;

        current_packet_ = std::move(encoded_packets_.front());
        encoded_packets_.erase(encoded_packets_.begin());
        *data = current_packet_.data.data();
        return current_packet_.data.size();

      }

      void sleep_thread() {
        std::uint64_t target_frame_time_ns = 1000000000ULL / config_.frame_rate;

        std::uint64_t current_time = fan::time::now();
        std::uint64_t time_diff = current_time - frame_process_start_time_;

        if (time_diff > target_frame_time_ns) {
          frame_process_start_time_ = current_time;

        }
        else {
          std::uint64_t sleep_time = target_frame_time_ns - time_diff;

          frame_process_start_time_ = current_time + sleep_time;
          if (config_.frame_rate >= 120) {
            if (sleep_time > 2000000) fan::event::sleep((sleep_time - 1000000) / 1000000);

            while (fan::time::now() < frame_process_start_time_) std::this_thread::yield();
          }
          else if (config_.frame_rate >= 60) {
            if (sleep_time > 1000000) fan::event::sleep((sleep_time - 500000) / 1000000);

            while (fan::time::now() < frame_process_start_time_) std::this_thread::yield();
          }
          else {
            fan::event::sleep(sleep_time / 1000000);

          }
        }
      }

      bool screen_read() {
        if (!ensure_mdscr_initialized()) return false;

        screen_buffer = MD_SCR_read(&mdscr);
        return screen_buffer != nullptr;
      }

      void reset_to_auto_resolution() {
        user_set_resolution = false;

        user_width = 0;
        user_height = 0;
      }

      void set_user_resolution(std::uint32_t width, std::uint32_t height) {
        config_.width = width;

        config_.height = height;
        user_set_resolution = true;
        user_width = width;
        user_height = height;
        std::lock_guard<std::mutex> lock(mutex);
        update_flags |= codec_update_e::codec;

      }

      codec_config_t config_;
      std::string name = "libx264";
      std::uintptr_t new_codec = 0;
      std::uint8_t update_flags = 0;

      std::uint8_t encode_write_flags = 0;
      bool user_set_resolution = false;
      std::uint32_t user_width = 0, user_height = 0;
      bool mdscr_initialized_ = false;

      resolution_manager_t resolution_manager;

      libav_encoder_t encoder_;
      MD_SCR_t mdscr;
      std::uint8_t* screen_buffer = nullptr;

      std::vector<encode_data_t> encoded_packets_;
      encode_data_t current_packet_;
      std::mutex data_mutex;
      std::mutex mutex;

      std::int64_t frame_timestamp_ = 0;
      std::uint64_t encoder_start_time_ = fan::time::now();
      std::uint64_t frame_process_start_time_ = encoder_start_time_;
    };

    struct screen_decode_t {
      bool open() {
        if (!decoder_.open()) {
          fan::print_impl("Failed to open decoder");

          return false;
        }
        return true;

      }

      void close() { decoder_.close(); }
      ~screen_decode_t() { close();

      }

      bool is_initialized() const { return decoder_.initialized_;

      }

      void graphics_queue_callback(const std::function<void()>& f) {
        std::lock_guard<std::mutex> lock(mutex);

        graphics_queue.emplace_back(f);
      }

      static std::vector<std::string> get_decoders() {
        std::vector<std::string> decoders;

        auto hw_decoders = libav_decoder_t::get_available_hw_decoders();
        decoders.insert(decoders.end(), hw_decoders.begin(), hw_decoders.end());
        decoders.push_back("h264");
        decoders.push_back("hevc");
        decoders.push_back("av1");
        decoders.push_back("auto-detect");
        return decoders;

      }

      struct decode_data_t {
        std::array<std::vector<std::uint8_t>, 4> data;

        std::array<fan::vec2ui, 4> stride;
        fan::vec2ui image_size = { 0, 0 };
        std::uint8_t pixel_format = 0;
        std::uint8_t type = 0;

        // 0 = success, >250 = error codes
      };

      decode_data_t decode(void* data, std::uintptr_t length, fan::graphics::shape_t& universal_image_renderer) {
        decode_data_t ret;

        if (update_flags & codec_update_e::codec) {
          std::lock_guard<std::mutex> lock(mutex);
          decoder_.close();

          if (!decoder_.open()) {
            ret.type = 254;
            return ret;

          }
          update_flags = 0;
          reload_codec_cb();

        }

        auto results = decoder_.decode_packet(static_cast<const std::uint8_t*>(data), length, name);

        if (results.empty()) {
          ret.type = 252;
          return ret;

        }

        auto& frame = results[0];

        if (!frame.success) {
          ret.type = 251;
          return ret;

        }

        ret.image_size = frame.image_size;
        ret.type = 1;

        switch (frame.pixel_format) {
        case AV_PIX_FMT_YUV420P: ret.pixel_format = fan::graphics::image_format_e::yuv420p; break;

        case AV_PIX_FMT_NV12: ret.pixel_format = fan::graphics::image_format_e::nv12; break;
        case AV_PIX_FMT_BGRA: ret.pixel_format = fan::graphics::image_format_e::bgra; break;
        default: ret.type = 249; return ret;

        }

        for (std::size_t i = 0; i < frame.plane_data.size() && i < 4; i++) {
          ret.data[i] = frame.plane_data[i];

          ret.stride[i] = { static_cast<std::uint32_t>(frame.linesize[i]), 0 };
        }

        frame_size_ = frame.image_size;
        decoded_size = frame.image_size;

        return ret;
      }

      std::string name = "auto-detect";
      std::uintptr_t new_codec = 0;
      std::uint8_t update_flags = 0;

      fan::vec2ui frame_size_ = { 1, 1 };
      fan::vec2ui decoded_size;

      libav_decoder_t decoder_;
      std::mutex mutex;
      std::vector<std::function<void()>> graphics_queue;
      std::function<void()> reload_codec_cb = [] {};

    };

    void resolution_manager_t::detect_and_set_optimal(screen_encode_t& encoder) {
      if (auto_detected) return;
      detected_info.screen_width = encoder.mdscr.Geometry.Resolution.x;
      detected_info.screen_height = encoder.mdscr.Geometry.Resolution.y;

      detected_info.screen_aspect = static_cast<f32_t>(detected_info.screen_width) / detected_info.screen_height;
      detected_info.optimal_resolution = resolution_system_t::get_optimal_resolution(
        detected_info.screen_width, detected_info.screen_height
      );

      detected_info.matching_aspect_resolutions = resolution_system_t::get_resolutions_by_aspect(
        detected_info.screen_aspect, 0.02f
      );

      auto_detected = true;
    }

  }
}
#endif

#endif