module;

#include <fan/types/types.h>

#ifndef fan_platform_android

#if defined(fan_compiler_msvc)
  
#endif

// if windows fails with duplicate resource, remove mux folder from libwebp
#include <webp/encode.h>
#include <webp/decode.h>

#include <unordered_map>
#include <string>

export module fan:graphics.webp;

import :types.vector;
import :print;
import :types.vector;
import :io.file;

export namespace fan {
	namespace webp {

    struct image_info_t {
      void* data;
      fan::vec2i size;
      int channels;
      uint8_t type = 0; // webp, stb
    };

    fan_api bool get_image_size(const std::string& file, fan::vec2ui* size) {
      std::string data;
      fan::io::file::read(file, &data);
      return WebPGetInfo((uint8_t*)data.data(), data.size(), (int*)&size->x, (int*)&size->y) != 1;
    }

    // if fails, try encode with -pix_fmt yuv420p
    fan_api bool decode(const uint8_t* webp_data, std::size_t size, image_info_t* image_info) {
      image_info->data = WebPDecodeRGBA(webp_data, size, &image_info->size.x, &image_info->size.y);
      image_info->channels = 4;
      return image_info->data == 0;
    }

    fan_api bool load(const std::string& file, image_info_t* image_info) {
    
      std::string data;
      fan::io::file::read(file, &data);

      bool failed = decode((const uint8_t*)data.data(), data.size(), image_info);
      if (failed) {
        fan::print_warning(std::string("failed to load image:") + std::string(file));
        return true;
      }

      return false;
    }
    fan_api std::size_t encode_rgba(const uint8_t* in, const fan::vec2& size, f32_t quality, uint8_t** out) {
      return WebPEncodeRGBA(in, size.x, size.y, size.x * 4, quality, out);
    }
    fan_api std::size_t encode_lossless_rgba(const uint8_t* in, const fan::vec2& size, uint8_t** out) {
      return WebPEncodeLosslessRGBA(in, size.x, size.y, size.x * 4, out);
    }

    fan_api void free_image(void* ptr) {
      WebPFree(ptr);
    }

    fan_api bool validate(const std::string& file_path) {
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