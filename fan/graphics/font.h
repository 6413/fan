#pragma once

#include <fan/font.h>
#include <fan/graphics/opengl/gl_image.h>

namespace fan {
  namespace graphics {
    namespace gl_font_impl {
      struct font_t {
        void open(const fan::string& image_path);
        void close();

        //uint16_t decode_letter(uint16_t c) const {
        //  return info.get_font_index(c);
        //}

        fan::font::font_t info;
        fan::graphics::image_t image;
      };
    }
  }
}