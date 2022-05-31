#pragma once

#include _FAN_PATH(types/types.h)


#if defined(fan_compiler_visual_studio)
  
#endif

// if windows fails with duplicat resource, remove mux folder from libwebp
#include <webp/encode.h>
#include <webp/decode.h>

#include _FAN_PATH(io/file.h)

namespace fan {
	namespace webp {

    struct image_info_t {
      uint8_t* data;
      fan::vec2i size;
    };

    static bool get_image_size(const std::string_view file, fan::vec2i* size) {
      auto data = fan::io::file::read(std::string(file));
      return WebPGetInfo((uint8_t*)data.data(), data.size(), &size->x, &size->y) != 1;
    }

	  static image_info_t decode(const uint8_t* webp_data, std::size_t size) {
      
      image_info_t image_info;
      image_info.data = WebPDecodeRGBA(webp_data, size, &image_info.size.x, &image_info.size.y);

      return image_info;
    }

    static image_info_t load(const std::string_view file) {
    
      auto data = fan::io::file::read(std::string(file));

      auto image_info = decode((const uint8_t*)data.data(), data.size());

      if (image_info.data == nullptr) {
        fan::throw_error("failed to open image " + std::string(file));
      }

      return image_info;
    }

    static void free_image(void* ptr) {
      WebPFree(ptr);
    }
  
  }
}