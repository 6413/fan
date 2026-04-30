module;

export module fan.graphics.image_load;

import std;

#if !defined(loco_no_stb)
  import fan.graphics.stb;
#endif

import fan.types.compile_time_string;
import fan.print.error;
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
    bool load(fan::str_view_t path, info_t* image_info, const std::source_location& callers_path = std::source_location::current());
    void free(info_t* image_info);
    
    void convert_channels(const std::uint8_t* src, std::uint8_t* dst, std::size_t pixels, int src_channels, int dst_channels, std::uint8_t default_alpha = 255);

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