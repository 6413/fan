struct font_t {

  void open(loco_t* loco, const fan::string& image_path) {
    loco_t::image_t::load_properties_t lp;
    #if defined(loco_opengl)
      lp.filter = fan::opengl::GL_LINEAR;
    #elif defined(loco_vulkan)
      // fill here
    #endif
    image.load(loco, image_path + ".webp", lp);
    info = fan::font::parse_font(image_path + "_metrics.txt");
  }
  void close(loco_t* loco) {
    image.unload(loco);
  }

  //uint16_t decode_letter(uint16_t c) const {
  //  return info.get_font_index(c);
  //}

  fan::font::font_t info;
  loco_t::image_t image;
};