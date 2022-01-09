#pragma once

// if windows fails with duplicat resource, remove mux folder from libwebp
#include <webp/encode.h>
#include <webp/decode.h>

#include <fan/io/file.hpp>

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

    static image_info_t load_image(const std::string& file) {
    
      auto data = fan::io::file::read(file);

      return decode((const uint8_t*)data.data(), data.size());
    }

    static void free_image(void* ptr) {
      WebPFree(ptr);
      ptr = nullptr;
    }
  
  }
}