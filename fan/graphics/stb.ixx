module;

#include <cstdint>
#include <source_location>
#include <string>

export module fan.graphics.stb;

import fan.types.vector;

export namespace fan {
  namespace stb {
    struct info_t {
      unsigned char* data;
      fan::vec2i size;
      int channels;
      std::uint8_t type;
    };

    bool validate(const std::string& file, const std::source_location& callers_path = std::source_location::current());
    bool load(const std::string& file, info_t* image_info, const std::source_location& callers_path = std::source_location::current());
    void free_image(void* data);
  }
}