module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#include <cstring>
#include <libavformat/avformat.h>

#if defined(FAN_2D)
#include <fan/utility.h>
#include <WITCH/WITCH.h>
#include <WITCH/PR/PR.h>
#include <WITCH/MD/SCR/SCR.h>
#endif

#endif

module fan.graphics.video.screen;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#if defined(FAN_2D)

bool fan::graphics::screen_encode_t::encode_write() {
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

void fan::graphics::screen_encode_t::sleep_thread() {
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

fan::graphics::screen_decode_t::decode_data_t fan::graphics::screen_decode_t::decode(void* data, std::uintptr_t length, fan::graphics::shape_t& universal_image_renderer) {
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

#endif

#endif
