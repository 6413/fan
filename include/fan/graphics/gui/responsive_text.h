struct responsive_text_t : loco_t::shape_t {
   //static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::rectangle;
  template <typename T>
  struct properties_t : loco_t::text_t::properties_t {
    T shape_info;
  };

  template <typename T>
  responsive_text_t(const properties_t<T>& properties) 
    : shape_t(properties.shape_info){
    auto& p = *(loco_t::text_t::properties_t*)&properties;
    p.camera = properties.shape_info.camera;
    p.viewport = properties.shape_info.viewport;
    p.position = properties.shape_info.position;
    p.position.z += 1;

    current_font_size = properties.font_size;
    f32_t scaler = properties.shape_info.size.y * 2 / gloco->font.info.height;
    p.font_size = gloco->font.info.size * scaler * current_font_size;
    m_text = p;
  }
  responsive_text_t() {

  }
  ~responsive_text_t() {

  }

  void set_size(const fan::vec2& size) {
    f32_t scaler = size.y * 2 / gloco->font.info.height;
    shape_t::set_size(size);
    fan::print(current_font_size, gloco->font.info.size * scaler * current_font_size);
    m_text.set_font_size(gloco->font.info.size * scaler * current_font_size);
    //p.font_size = ;
  }

  template <typename T>
  static responsive_text_t::properties_t<T> make_properties(T shape_info, const loco_t::text_t::properties_t& tp) {
      return responsive_text_t::properties_t<T>{tp, shape_info};
  }
  f32_t current_font_size = 0;
  loco_t::shape_t m_text;
};