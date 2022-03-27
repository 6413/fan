#pragma once

#include <fan/types/types.h>

// if windows fails with duplicat resource, remove mux folder from libwebp
#include <webp/encode.h>
#include <webp/decode.h>

#include <fan/io/file.h>

namespace fan {
	namespace webp {

    struct image_info_t {
      uint8_t* data;
      fan::vec2i size;
    };

	  static image_info_t decode(const uint8_t* webp_data, std::size_t size) {

      image_info_t image_info;
    
      image_info.data = WebPDecodeRGBA(webp_data, size, &image_info.size.x, &image_info.size.y);

      return image_info;
    }

    static image_info_t load_image(const std::string_view file) {
    
      auto data = fan::io::file::read(std::string(file));

      auto image_info = decode((const uint8_t*)data.data(), data.size());

      if (image_info.data == nullptr) {
        fan::throw_error("failed to open image " + std::string(file));
      }

      return image_info;
    }

    static void free_image(void* ptr) {
      WebPFree(ptr);
      ptr = nullptr;
    }
  
  }
}