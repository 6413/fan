#pragma once

#include _FAN_PATH(types/types.h)

#ifndef fan_platform_android

#if defined(fan_compiler_visual_studio)
  
#endif

// if windows fails with duplicat resource, remove mux folder from libwebp
#include <webp/encode.h>
#include <webp/decode.h>

#include _FAN_PATH(io/file.h)

namespace fan {
	namespace webp {

    struct image_info_t {
      void* data;
      fan::vec2i size;
    };

    static bool get_image_size(const std::string_view file, fan::vec2ui* size) {
      fan::string data;
      fan::io::file::read(fan::string(file), &data);
      return WebPGetInfo((uint8_t*)data.data(), data.size(), (int*)&size->x, (int*)&size->y) != 1;
    }

	  static bool decode(const uint8_t* webp_data, std::size_t size, image_info_t* image_info) {
      image_info->data = WebPDecodeRGBA(webp_data, size, &image_info->size.x, &image_info->size.y);
      return image_info->data == 0;
    }

    static bool load(const std::string_view file, image_info_t* image_info) {
    
      fan::string data;
      fan::io::file::read(fan::string(file), &data);

      bool failed = decode((const uint8_t*)data.data(), data.size(), image_info);
      if (failed) {
        fan::print_warning(fan::string("failed to load image:") + fan::string(file));
        return true;
      }

      return false;
    }
    static uint32_t encode_rgba(const uint8_t* in, const fan::vec2& size, f32_t quality, uint8_t** out) {
      return WebPEncodeRGBA(in, size.x, size.y, size.x * 4, quality, out);
    }
    static uint32_t encode_lossless_rgba(const uint8_t* in, const fan::vec2& size, uint8_t** out) {
      return WebPEncodeLosslessRGBA(in, size.x, size.y, size.x * 4, out);
    }

    static void free_image(void* ptr) {
      WebPFree(ptr);
    }
  
  }
}

#endif