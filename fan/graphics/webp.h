#pragma once

#include <unordered_map>

#ifndef fan_platform_android

#if defined(fan_compiler_msvc)
  
#endif

// if windows fails with duplicate resource, remove mux folder from libwebp
#include <webp/encode.h>
#include <webp/decode.h>

#include <fan/io/file.h>

namespace fan {
	namespace webp {

    struct image_info_t {
      void* data;
      fan::vec2i size;
      int channels;
      uint8_t type = 0; // webp, stb
    };

    static bool get_image_size(const fan::string& file, fan::vec2ui* size) {
      fan::string data;
      fan::io::file::read(file, &data);
      return WebPGetInfo((uint8_t*)data.data(), data.size(), (int*)&size->x, (int*)&size->y) != 1;
    }

    // if fails, try encode with -pix_fmt yuv420p
    static bool decode(const uint8_t* webp_data, std::size_t size, image_info_t* image_info) {
      image_info->data = WebPDecodeRGBA(webp_data, size, &image_info->size.x, &image_info->size.y);
      image_info->channels = 4;
      return image_info->data == 0;
    }

    static bool load(const std::string& file, image_info_t* image_info) {
    
      fan::string data;
      fan::io::file::read(file, &data);

      bool failed = decode((const uint8_t*)data.data(), data.size(), image_info);
      if (failed) {
        fan::print_warning(fan::string("failed to load image:") + fan::string(file));
        return true;
      }

      return false;
    }
    static std::size_t encode_rgba(const uint8_t* in, const fan::vec2& size, f32_t quality, uint8_t** out) {
      return WebPEncodeRGBA(in, size.x, size.y, size.x * 4, quality, out);
    }
    static std::size_t encode_lossless_rgba(const uint8_t* in, const fan::vec2& size, uint8_t** out) {
      return WebPEncodeLosslessRGBA(in, size.x, size.y, size.x * 4, out);
    }

    static void free_image(void* ptr) {
      WebPFree(ptr);
    }

    static bool validate_webp(const std::string& file_path) {
      std::string data;
      static constexpr uint32_t webp_header_size = 32;
      if (fan::io::file::read(file_path, &data, webp_header_size)) {
        return false;
      }
      int width, height;
      if (WebPGetInfo((const uint8_t*)data.c_str(), webp_header_size, &width, &height)) {
        return true;
      }
      return false;
    }
  
  }
}

#endif