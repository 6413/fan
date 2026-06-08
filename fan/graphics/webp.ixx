module;

export module fan.graphics.webp;

import std;

import fan.types;
import fan.types.compile_time_string;
import fan.types.vector;

export namespace fan {
  namespace webp {

    struct info_t {
      void* data;
      fan::vec2i size;
      int channels;
      std::uint8_t type;
    };

    bool get_image_size(fan::str_view_t path, fan::vec2ui* size, const std::source_location& callers_path = std::source_location::current());
    bool decode(const std::uint8_t* webp_data, std::size_t size, info_t* image_info);
    bool load(fan::str_view_t path, info_t* image_info, const std::source_location& callers_path = std::source_location::current());

    bool write(fan::str_view_t path, const info_t& image_info, f32_t quality = 80.f);
    bool write(fan::str_view_t path, void* data, fan::vec2i size, int channels, f32_t quality = 80.f);
    bool write(fan::str_view_t path, std::span<const std::uint8_t> data, fan::vec2i size, int channels, f32_t quality = 80.f);

    std::size_t encode_rgba(const std::uint8_t* in, const fan::vec2& size, f32_t quality, std::uint8_t** out);
    std::size_t encode_lossless_rgba(const std::uint8_t* in, const fan::vec2& size, std::uint8_t** out);
    void free_image(void* ptr);
    bool validate(fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());
  }
}