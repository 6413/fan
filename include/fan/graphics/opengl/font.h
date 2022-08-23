#pragma once

#include _FAN_PATH(font.h)
#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_image.h)

namespace fan_2d {
  namespace graphics {

    struct font_t {

      void open(fan::opengl::context_t* context, const std::string& image_path) {
        fan::opengl::image_t::load_properties_t lp;
        lp.filter = fan::opengl::GL_LINEAR;
        image.load(context, image_path + ".webp", lp);
        info = fan::font::parse_font(image_path + "_metrics.txt");
      }
      void close(fan::opengl::context_t* context) {
        image.unload(context);
      }

      //uint32_t decode_letter(wchar_t c) const {
      //  return info.get_font_index(c);
      //}

      fan::font::font_t info;
      fan::opengl::image_t image;
    };

  }
}