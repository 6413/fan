module;

#if defined (FAN_WINDOW)

#if defined(FAN_2D)
  extern "C" {
    #include <libavformat/avformat.h>
  }
#endif

#endif

export module fan.graphics.video.renderer;

#if defined (FAN_WINDOW)

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

        void update() {
          if (fan::window::is_key_clicked(fan::key_space)) { toggle_pause(); }
          if (fan::window::is_key_clicked(fan::key_left)) { skip(-seek_time_skip); }
          if (fan::window::is_key_clicked(fan::key_right)) { skip(seek_time_skip); }

          if (dropping_seek_frames) {
            for (int i = 0; i < 16; i++) {
              if (read_packet()) {
                auto out = decoder.decode_packet(buf.data(), buf.size(), codec_name, false);
                if (!out.empty()) {
                  frames.insert(frames.end(), std::make_move_iterator(out.begin()), std::make_move_iterator(out.end()));
                }
              }
              while (!frames.empty()) {
                if (static_cast<f32_t>(packet_pts * time_base) >= seek_target_time) {
                  dropping_seek_frames = false;
                  break;
                }
                frames.erase(frames.begin());
              }
              if (!dropping_seek_frames) break;
            }
            frame_timer.restart();
            return;
          }

          if (frames.size() < 8) {
            if (read_packet()) {
              auto out = decoder.decode_packet(buf.data(), buf.size(), codec_name, false);
              if (!out.empty()) {
                frames.insert(frames.end(), std::make_move_iterator(out.begin()), std::make_move_iterator(out.end()));
              }
            }
          }

          if (paused) return;

          if (frame_timer && !frames.empty()) {
            frame_timer.restart();
            auto& frame = frames.front();
            if (frame.success && !frame.plane_data.empty()) {
              renderer.update(frame.image_size, frame.plane_data, frame.linesize, frame.pixel_format);
              if (packet_pts != std::numeric_limits<std::int64_t>::min()) {
                current_time = static_cast<f32_t>(packet_pts * time_base);
              }
            }
            frames.erase(frames.begin());
          }
        }

        void show(bool controls = true) {
          f32_t padding = 20.f;
          f32_t control_height = 150.f;

          fan::vec2 vp = gui::get_window_size().offset_y(-control_height);
          fan::vec2 fit = fan::vec2(demuxer.get_video_size()).fit(vp);
          renderer.shape.set_size(fit / 2.f);
          renderer.shape.set_position(gui::get_window_pos() + vp / 2.f);
          if (!controls) return;
          fan::vec2 avail = gui::get_window_size();
          fan::vec2 ws_pos = gui::get_window_pos();
          gui::anchor_bottom_left(fan::vec2(padding, -control_height));
          if (auto cw = gui::child_window("##controls", fan::vec2(avail.x, control_height))) {
            gui::set_cursor_pos_y(gui::get_cursor_pos_y() + padding);
            { // times + slider
              f32_t width = gui::get_content_region_avail().x;
              std::string cur = fan::time::format_seconds(current_time);
              std::string dur = fan::time::format_seconds(duration);
              fan::vec2 cur_size = gui::get_text_size(cur);
              fan::vec2 dur_size = gui::get_text_size(dur);
              gui::text(cur);
              gui::same_line();
              gui::spacing();
              gui::same_line();
              gui::set_cursor_pos_y(gui::get_cursor_pos_y() - cur_size.y / 4.f);
              gui::set_next_item_width(width - (cur_size.x + dur_size.x) - padding * 3.f);
              gui::slider("##seek_bar", &display_time, 0.f, duration);
              gui::same_line();
              gui::spacing();
              gui::same_line();
              gui::set_cursor_pos_y(gui::get_cursor_pos_y() - dur_size.y / 4.f);
              gui::text(dur);
            }

            if (gui::is_item_edited()) {
              seek(display_time);
            }
            else if (!gui::is_item_active()) {
              display_time = current_time;
            }
            gui::style_scope_t s;
            s.color(gui::col_button, {0, 0, 0, 0})
             .color(gui::col_button_hovered, {0.25, 0.25, 0.25, 0.25})
             .color(gui::col_button_active, {0.3, 0.3, 0.3, 0.3});
            if (gui::image_button(image_t{ paused ? "icons/wplay.webp" : "icons/wpause.webp", image_presets::pixel_art()}, {28.f, 28.f})) toggle_pause();
            gui::same_line();
            if (gui::button("-" + std::to_string(int(seek_time_skip)) + "s")) { skip(-seek_time_skip); }
            gui::same_line();
            if (gui::button("+" + std::to_string(int(seek_time_skip)) + "s")) { skip(seek_time_skip); }

            //show_debug();

          } // cw controls
        }
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