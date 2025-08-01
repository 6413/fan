module;

#include <fan/types/types.h>

#include <string>

export module fan.graphics.image_load;

#if !defined(loco_no_stb)
  import fan.graphics.stb;
#endif

import fan.print;
import fan.types.vector;
import fan.graphics.webp;

export namespace fan {
  namespace image {
    struct image_type_e{
      enum  {
        webp,
        stb
      };
    };

    struct info_t {
      void* data;
      fan::vec2i size;
      int channels = -1;
      std::uint8_t type; // webp, stb
    };

    bool valid(const std::string& path) {
      if (fan::webp::validate(path)) {
        return true;
      }
      else if (fan::stb::validate(path)) {
        return true;
      }
      return false;
    }

    bool load(const std::string& file, info_t* image_info) {
      bool ret;
      if (fan::webp::validate(file)) {
        ret = fan::webp::load(file, (fan::webp::info_t*)image_info);
        image_info->type = image_type_e::webp;
      }
      else {
        #if !defined(loco_no_stb)
          ret = fan::stb::load(file, (fan::stb::info_t*)image_info);
          image_info->type = image_type_e::stb;
        #endif
      }
#if fan_debug >= fan_debug_low
      if (ret) {
        fan::print_warning("failed to load image data from path:" + file);
      }
#endif
      return ret;
    }
    fan_module_api void free(info_t* image_info) {
      if (image_info->type == image_type_e::webp) {
        fan::webp::free_image(image_info->data);
      }
      else if (image_info->type == image_type_e::stb) {
        #if !defined(loco_no_stb)
        fan::stb::free_image(image_info->data);
        #endif
      }
    }
    inline constexpr uint8_t missing_texture_pixels[16] = {
      0, 0, 0, 255,
      255, 0, 220, 255,
      255, 0, 220, 255,
      0, 0, 0, 255
    };
    inline constexpr uint8_t transparent_texture_pixels[16] = {
      60, 60, 60, 255,
      40, 40, 40, 255,
      40, 40, 40, 255,
      60, 60, 60, 255
    };
  }
}

#undef loco_no_stb