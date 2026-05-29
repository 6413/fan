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
        bool open(const std::string& path, const fan::vec2& size) {
          if (!demuxer.open(path.c_str())) {
            return false;
          }
          decoder.open();
          f32_t fps = demuxer.get_fps();
          frame_timer = fan::time::seconds_timer(1.f / fps);
          renderer.open(size);
          return true;
        }

        bool open(const std::string& path) {
          if (!demuxer.open(path.c_str())) return false;
          decoder.open();
          f32_t fps = demuxer.get_fps();
          frame_timer = fan::time::seconds_timer(1.f / fps);
          // no renderer.open() — first frame will allocate via renderer.update()
          return true;
        }

        void close() {
          renderer.close();
          decoder.close();
          demuxer.close();
          frames.clear();
        }

        void update() {
          if (frames.size() < 8) {
            AVCodecID cid = demuxer.get_codec_id();
            bool is_h264 = (cid == AV_CODEC_ID_H264);
            std::string codec_name = (cid == AV_CODEC_ID_VP9) ? "vp9" :
              (cid == AV_CODEC_ID_AV1) ? "av1" : "h264";
            bool ok = is_h264 ? demuxer.read_annexb_packet(buf)
              : demuxer.read_raw_packet(buf);
            if (ok) {
              auto out = decoder.decode_packet(buf.data(), buf.size(), codec_name, true);
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