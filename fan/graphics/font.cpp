#include "font.h"

void fan::graphics::gl_font_impl::font_t::open(const fan::string& image_path) {
  fan::graphics::image_t::load_properties_t lp;
  #if defined(loco_opengl)
  lp.min_filter = fan::opengl::GL_LINEAR;
  lp.mag_filter = fan::opengl::GL_LINEAR;
  #elif defined(loco_vulkan)
  // fill here
  #endif
  image.load(image_path + ".webp", lp);
  fan::font::parse_font(info, image_path + "_metrics.txt");
}

void fan::graphics::gl_font_impl::font_t::close() {
  image.unload();
}
