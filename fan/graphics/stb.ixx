module;

#include <fan/types/types.h>

#ifndef fan_platform_android

#if defined(fan_compiler_msvc)
  
#endif

#include <string>

// if windows fails with duplicat resource, remove mux folder from libwebp
#define STB_IMAGE_IMPLEMENTATION
#include <fan/stb/stb_image.h>

//#include <fan/stb/stb_image_write.h>

export module fan.graphics.stb;

import fan.print;
import fan.types.vector;
import fan.io.file;

export namespace fan {
    namespace stb {

    struct info_t {
      unsigned char* data;
      fan::vec2i size;
      int channels;
      std::uint8_t type = 1; // webp, stb
    };

    fan_module_api bool validate(const std::string& file) {
      int x, y, channels;
      return stbi_info(file.c_str(), &x, &y, &channels);
    }

    fan_module_api bool load(const std::string& file, info_t* image_info) {
    
      image_info->data = stbi_load(file.c_str(), &image_info->size.x, &image_info->size.y, &image_info->channels, 0);
      if (!image_info->data) {
        fan::print_warning(std::string("failed to load image:") + std::string(file) + ", error:" + stbi_failure_reason());
        return true;
      }

      return false;
    }

    // static bool encode_rgba(const fan::string& file, const info_t& image_info) {
    //   return stbi_write_png(file.c_str(), image_info.size.x, image_info.size.y, image_info.channels, image_info.data, image_info.size.x * image_info.channels);
    // }

    // static bool encode_lossless_rgba(const fan::string& file, const info_t& image_info) {
    //   return stbi_write_png(file.c_str(), image_info.size.x, image_info.size.y, image_info.channels, image_info.data, image_info.size.x * image_info.channels);
    // }

    fan_module_api void free_image(void* data) {
      stbi_image_free(data);
    }
  
  }
}
#endif