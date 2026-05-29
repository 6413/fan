module;

export module fan.graphics.video.renderer;
import std;
import fan.time;
import fan.graphics;
import fan.graphics.video.codec;

export namespace fan {
  namespace graphics {
    namespace video {

      struct renderer_t {
        fan::graphics::sprite_t sprite;
        fan::graphics::image_t image;

        void open(const fan::vec2& position, const fan::vec2& size) {
          image = fan::graphics::image_t(size, 4, fan::graphics::image_presets::smooth());
          sprite = fan::graphics::sprite_t(position, size / 2.f, image);
        }
        void open(const fan::vec2& size) {
          image = fan::graphics::image_t(size, 4, fan::graphics::image_presets::smooth());
        }

        void close() {
          if (sprite) {
            sprite.erase();
          }
          if (image.valid()) {
            image.unload();
          }
        }

        void update(const fan::vec2ui& new_size, const std::uint8_t* pixel_data, int channels = 4) {
          if (!image.valid() || image.get_size() != new_size) {
            if (image.valid()) {
              image.unload();
            }
            image = fan::graphics::image_t(new_size, channels, fan::graphics::image_presets::smooth());
          }
          image.update(pixel_data, channels);
        }
      };

      struct player_t {
        f32_t current_time = 0.f;
        f32_t duration = 0.f;
        f64_t time_base = 0.0;
        std::int64_t packet_pts = std::numeric_limits<std::int64_t>::min();
        f32_t seek_target_time = 0.f;
        bool dropping_seek_frames = false;

        bool open(const std::string& path, const fan::vec2& position, const fan::vec2& size) {
          if (!demuxer.open(path.c_str())) return false;
          decoder.open();
          duration = demuxer.get_duration();
          time_base = demuxer.get_time_base();
          current_time = 0.f;
          frame_timer = fan::time::seconds_timer(1.f / demuxer.get_fps());
          renderer.open(position, size);
          return true;
        }

        bool open(const std::string& path, const fan::vec2& size) {
          if (!demuxer.open(path.c_str())) return false;
          decoder.open();
          duration = demuxer.get_duration();
          time_base = demuxer.get_time_base();
          current_time = 0.f;
          frame_timer = fan::time::seconds_timer(1.f / demuxer.get_fps());
          renderer.open(size);
          return true;
        }

        bool open(const std::string& path) {
          if (!demuxer.open(path.c_str())) return false;
          decoder.open();
          duration = demuxer.get_duration();
          time_base = demuxer.get_time_base();
          current_time = 0.f;
          frame_timer = fan::time::seconds_timer(1.f / demuxer.get_fps());
          return true;
        }

        void close() {
          renderer.close();
          decoder.close();
          demuxer.close();
          frames.clear();
        }

        bool seek(f32_t seconds) {
          seconds = std::clamp(seconds, 0.f, duration);
          if (!demuxer.seek(seconds)) return false;
          decoder.flush();
          frames.clear();
          buf.clear();
          packet_pts = std::numeric_limits<std::int64_t>::min();
          seek_target_time = seconds;
          dropping_seek_frames = true;
          current_time = seconds;
          frame_timer.restart();
          return true;
        }

        void update() {
          if (dropping_seek_frames) {
            for (int i = 0; i < 16; i++) {
              if (demuxer.read_annexb_packet(buf, packet_pts)) {
                auto out = decoder.decode_packet(buf.data(), buf.size(), "h264", true);
                if (!out.empty())
                  frames.insert(frames.end(), std::make_move_iterator(out.begin()), std::make_move_iterator(out.end()));
              }
              while (!frames.empty()) {
                f32_t frame_time = static_cast<f32_t>(packet_pts * time_base);
                if (frame_time >= seek_target_time) {
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
            if (demuxer.read_annexb_packet(buf, packet_pts)) {
              auto out = decoder.decode_packet(buf.data(), buf.size(), "h264", true);
              if (!out.empty())
                frames.insert(frames.end(), std::make_move_iterator(out.begin()), std::make_move_iterator(out.end()));
            }
          }

          if (frame_timer && !frames.empty()) {
            frame_timer.restart();
            auto& frame = frames.front();
            if (frame.success && !frame.plane_data.empty()) {
              renderer.update(frame.image_size, frame.plane_data[0].data());
              if (packet_pts != std::numeric_limits<std::int64_t>::min())
                current_time = static_cast<f32_t>(packet_pts * time_base);
            }
            frames.erase(frames.begin());
          }
        }

        fan::graphics::video::renderer_t renderer;
        fan::mp4_demuxer_t demuxer;
        fan::graphics::libav_decoder_t decoder;
        fan::time::timer frame_timer;
        std::vector<std::uint8_t> buf;
        std::vector<fan::graphics::libav_decoder_t::decode_result_t> frames;
      };

    }
  }
}