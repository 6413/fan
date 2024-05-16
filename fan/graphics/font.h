#include <fan/font.h>

namespace fan {
  namespace graphics {
    namespace gl_font_impl {
      struct font_t {
        void open(const fan::string& image_path);
        void close();

        //uint16_t decode_letter(uint16_t c) const {
        //  return info.get_font_index(c);
        //}

        fan::vec2 get_text_size(const fan::string& text, f32_t font_size);

        fan::font::font_t info;
        fan::opengl::context_t::image_nr_t image;
      };
    }
  }
}