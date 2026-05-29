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
            fan::vec3 pos = sprite ?
              sprite.get_position() : fan::vec3(0, 0, 0);
            if (sprite) {
              sprite.erase();
            }
            if (image.valid()) {
              image.unload();
            }
            image = fan::graphics::image_t(new_size, channels, fan::graphics::image_presets::smooth());
            sprite = fan::graphics::sprite_t(pos, new_size / 2.f, image);
          }
          image.update(pixel_data, channels);
        }
      };

      struct player_t {
        bool open(const std::string& path, const fan::vec2& position, const fan::vec2& size) {
          if (!demuxer.open(path.c_str())) {
            return false;
          }
          decoder.open();
          f32_t fps = demuxer.get_fps();
          frame_timer = fan::time::seconds_timer(1.f / fps);
          renderer.open(position, size);
          return true;
        }

        void close() {
          renderer.close();
          decoder.close();
          frames.clear();
        }

        void update() {
          if (frames.size() < 8) {
            if (demuxer.read_annexb_packet(buf)) {
              auto out = decoder.decode_packet(buf.data(), buf.size(), "h264", true);
              if (!out.empty()) {
                frames.insert(frames.end(), std::make_move_iterator(out.begin()), std::make_move_iterator(out.end()));
              }
            }
          }
          if (frame_timer && !frames.empty()) {
            frame_timer.restart();
            auto& frame = frames.front();
            if (frame.success && !frame.plane_data.empty()) {
              renderer.update(frame.image_size, frame.plane_data[0].data());
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