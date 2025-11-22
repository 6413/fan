module;

#include <cstdint>
#include <source_location>
#include <string>

export module fan.graphics.image_load;

#if !defined(loco_no_stb)
  import fan.graphics.stb;
#endif

import fan.print;
import fan.utility;
import fan.types.vector;
import fan.graphics.webp;

export namespace fan {
  namespace image {
    struct image_type_e {
      enum {
        webp,
        stb
      };
    };

    struct info_t {
      void* data;
      fan::vec2i size;
      int channels = -1;
      std::uint8_t type;
    };

    bool valid(const std::string& path, const std::source_location& callers_path = std::source_location::current());
    bool load(const std::string& file, info_t* image_info, const std::source_location& callers_path = std::source_location::current());
    void free(info_t* image_info);

    inline constexpr std::uint8_t missing_texture_pixels[16] = {
      0, 0, 0, 255,
      255, 0, 220, 255,
      255, 0, 220, 255,
      0, 0, 0, 255
    };
    inline constexpr std::uint8_t transparent_texture_pixels[16] = {
      60, 60, 60, 255,
      40, 40, 40, 255,
      40, 40, 40, 255,
      60, 60, 60, 255
    };
  }
}

#undef loco_no_stb