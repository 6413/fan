#pragma once

#ifndef fan_platform_android

#if defined(fan_compiler_msvc)
  
#endif

// if windows fails with duplicat resource, remove mux folder from libwebp
#define STB_IMAGE_IMPLEMENTATION
#include <fan/stb/stb_image.h>
//#include <fan/stb/stb_image_write.h>

#include <fan/io/file.h>

namespace fan {
    namespace stb {

    struct image_info_t {
      unsigned char* data;
      fan::vec2i size;
      int channels;
    };

    static bool load(const std::string& file, image_info_t* image_info) {
    
      image_info->data = stbi_load(file.c_str(), &image_info->size.x, &image_info->size.y, &image_info->channels, 0);
      if (!image_info->data) {
        fan::print_warning(std::string("failed to load image:") + std::string(file));
        return true;
      }

      return false;
    }

    // static bool encode_rgba(const fan::string& file, const image_info_t& image_info) {
    //   return stbi_write_png(file.c_str(), image_info.size.x, image_info.size.y, image_info.channels, image_info.data, image_info.size.x * image_info.channels);
    // }

    // static bool encode_lossless_rgba(const fan::string& file, const image_info_t& image_info) {
    //   return stbi_write_png(file.c_str(), image_info.size.x, image_info.size.y, image_info.channels, image_info.data, image_info.size.x * image_info.channels);
    // }

    static void free_image(image_info_t* image_info) {
      stbi_image_free(image_info->data);
    }
  
  }
}
#endif