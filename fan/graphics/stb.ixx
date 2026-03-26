module;

#include <cstdint>
#include <source_location>
#include <string>

export module fan.graphics.stb;

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
    void free_image(void* data);
  }
}