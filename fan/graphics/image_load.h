#pragma once

#include <fan/graphics/webp.h>
#if !defined(loco_no_stb)
  #include <fan/graphics/stb.h>
#endif

namespace fan {
  namespace image {

    struct image_info_t {
      void* data;
      fan::vec2i size;
      int channels;
      uint8_t type; // webp, stb
    };

    static bool load(const std::string& file, image_info_t* image_info) {
      bool ret;
      if (fan::webp::validate_webp(file)) {
        ret = fan::webp::load(file, (fan::webp::image_info_t*)image_info);
        image_info->type = 0;
      }
      else {
        #if !defined(loco_no_stb)
          ret = fan::stb::load(file, (fan::stb::image_info_t*)image_info);
          image_info->type = 1;
        #endif
      }
#if fan_debug >= fan_debug_low
      if (ret) {
        fan::print_warning("failed to load image data from path:" + file);
      }
#endif
      return ret;
    }
    static void free(image_info_t* image_info) {
      if (image_info->type == 0) { // webp
        fan::webp::free_image(image_info->data);
      }
      else if (image_info->type == 1) { // stb
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