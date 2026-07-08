module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#if defined(FAN_2D)
  extern "C" {
    #include <libavformat/avformat.h>
  }
#endif

#endif

export module fan.graphics.video.renderer;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

import std;
import fan.time;
import fan.formatter;
import fan.graphics;
import fan.graphics.video.codec;
import fan.graphics.gui.base;

using namespace fan::graphics;

export namespace fan {
  namespace graphics {
    namespace video {

      struct renderer_t {
        universal_image_renderer_t shape;
        std::array<image_t, 4> images;

        void open(const fan::vec2& position, const fan::vec2& size) {
          shape = universal_image_renderer_t{{
            .position = fan::vec3(position, 0),
            .size = size / 2.f,
          }};
        }
        void open(const fan::vec2& size) {
          shape = universal_image_renderer_t{{
            .size = size / 2.f,
          }};
        }

        void close() {
          if (shape) shape.erase();
          for (auto& img : images) {
            if (img.valid()) img.unload();
          }
        }

        void update(const fan::vec2ui& frame_size, const std::vector<std::vector<std::uint8_t>>& planes,
            const std::vector<int>& linesizes, AVPixelFormat av_fmt) {
          std::uint8_t fan_fmt = convert_pixel_format(av_fmt);
          if (!fan_fmt) return;
          shape.reload(fan_fmt, fan::image::plane_split(planes), frame_size);
        }
      };

      struct player_t {
        bool open(const std::string& path, const fan::vec2& position, const fan::vec2& size) {
          if (!open_common(path)) return false;
          renderer.open(position, size);
          return true;
        }
        bool open(const std::string& path, const fan::vec2& size) {
          if (!open_common(path)) return false;
          renderer.open(size);
          return true;
        }
        bool open(const std::string& path) {
          if (!open_common(path)) return false;
          renderer.open(demuxer.get_video_size());
          return true;
        }
        bool reopen(const std::string& path) {
          close();
          return open(path);
        }

        void close() {
          renderer.close();
          decoder.close();
          demuxer.close();
          frames.clear();
        }

        void toggle_pause() { paused = !paused; }
        void skip(f32_t seconds) { seek(current_time + seconds); }

        bool seek(f32_t seconds) {
          seconds = std::clamp(seconds, 0.f, duration);
          if (!demuxer.seek(seconds)) return false;
          decoder.flush();
          frames.clear();
          frames.shrink_to_fit();
          buf.clear();
          packet_pts = std::numeric_limits<std::int64_t>::min();
          seek_target_time = seconds;
          dropping_seek_frames = true;
          current_time = seconds;
          frame_timer.restart();
          return true;
        }

        void update();

        void show(bool controls = true);
        void show_debug() {
          auto res = !frames.empty() ? frames.front().image_size : fan::vec2ui {0, 0};
          gui::text(fan::format_args_no_space("decoder res: ", res.x, "x", res.y));
          gui::text(fan::format_args_no_space("render res: ",
            static_cast<int>(renderer.shape.get_size().x * 2.f), "x",
            static_cast<int>(renderer.shape.get_size().y * 2.f)));
          gui::text("fps: ", demuxer.get_fps());
        }

        std::string format_time() const {
          return fan::time::format_seconds(current_time) + " / " + fan::time::format_seconds(duration);
        }

        f32_t current_time = 0.f;
        f32_t duration = 0.f;
        f64_t time_base = 0.0;
        f32_t display_time = 0.f;
        std::int64_t packet_pts = std::numeric_limits<std::int64_t>::min();
        f32_t seek_target_time = 0.f;
        inline static f32_t seek_time_skip = 10.f;
        bool dropping_seek_frames = false;
        bool paused = false;
        std::string codec_name = "h264";

        video::renderer_t renderer;
        fan::mp4_demuxer_t demuxer;
        libav_decoder_t decoder;
        fan::time::timer frame_timer;
        std::vector<std::uint8_t> buf;
        std::vector<libav_decoder_t::decode_result_t> frames;

        bool read_packet() {
          return demuxer.needs_annexb()
            ? demuxer.read_annexb_packet(buf, packet_pts)
            : demuxer.read_raw_packet(buf, packet_pts);
        }

        bool open_common(const std::string& path) {
          if (!demuxer.open(path.c_str())) return false;
          decoder.open();
          duration = demuxer.get_duration();
          time_base = demuxer.get_time_base();
          current_time = 0.f;
          codec_name = demuxer.get_codec_name();
          frame_timer = fan::time::seconds_timer(1.f / demuxer.get_fps());
          return true;
        }
      };

    }
  }
}


#endif