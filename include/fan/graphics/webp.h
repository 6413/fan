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
      std::string data;
      fan::io::file::read(std::string(file), &data);
      return WebPGetInfo((uint8_t*)data.data(), data.size(), &size->x, &size->y) != 1;
    }

	  static bool decode(const uint8_t* webp_data, std::size_t size, image_info_t* image_info) {
      image_info->data = WebPDecodeRGBA(webp_data, size, &image_info->size.x, &image_info->size.y);
      return image_info->data == 0;
    }

    static bool load(const std::string_view file, image_info_t* image_info) {
    
      std::string data;
      fan::io::file::read(std::string(file), &data);

      bool failed = decode((const uint8_t*)data.data(), data.size(), image_info);
      if (failed) {
        fan::print_warning(std::string("failed to load image:") + std::string(file));
        return false;
      }

      return true;
    }

    static void free_image(void* ptr) {
      WebPFree(ptr);
    }
  
  }
}