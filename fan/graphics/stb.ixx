module;

export module fan.graphics.stb;

import std;

import fan.types;
import fan.types.vector;

import fan.types.compile_time_string;

export namespace fan {
  namespace stb {
    struct info_t {
      unsigned char* data;
      fan::vec2i size;
      int channels;
      std::uint8_t type;
    };

    bool validate(fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());
    bool load(fan::str_view_t path, info_t* image_info, const std::source_location& callers_path = std::source_location::current());
    bool write(fan::str_view_t path, const info_t& image_info, f32_t quality = 80.f);
    bool write(fan::str_view_t path, void* data, fan::vec2i size, int channels, f32_t quality = 80.f);
    bool write(fan::str_view_t path, std::span<const std::uint8_t> data, fan::vec2i size, int channels, f32_t quality = 80.f);
    void free_image(void* data);
  }
}