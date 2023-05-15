struct responsive_text_custom_t : loco_t::shape_t {
   //static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::rectangle;
  template <typename T>
  struct properties_t : loco_t::text_t::properties_t {
    T shape_info;
  };

  //bool does_text_fit(const fan::string& text) {
  //  f32_t text_width = m_text.get_text_size().x;
  //  return 
  //}

  template <typename T>
  responsive_text_custom_t(const properties_t<T>& properties) 
    : shape_t(properties.shape_info){
    auto& p = *(loco_t::text_t::properties_t*)&properties;
    p.camera = properties.shape_info.camera;
    p.viewport = properties.shape_info.viewport;
    p.position = properties.shape_info.position;
    p.position.z += 1;

    current_font_size = properties.font_size;

    m_text_lines.resize(1.0 / current_font_size);

    for (uint32_t i = 0; i < m_text_lines.size(); ++i) { //                                                                                   fix when font size 1
      p.position = *(fan::vec2*)&properties.shape_info.position + fan::vec2(0, current_font_size * i * properties.shape_info.size.y * 2 /*- properties.shape_info.size.y / 2*/);
      m_text_lines[i] = p;
    }

    set_size(properties.shape_info.size);
  }
  responsive_text_custom_t() {

  }

  bool does_text_fit(const fan::string& text) {
    fan::vec2 size = get_size();
    f32_t font_size = m_text_lines[line_index].get_font_size();
    f32_t line0_width = gloco->text.get_text_size(m_text_lines[line_index].get_text(), font_size).x;
    f32_t line1_width = gloco->text.get_text_size(text, font_size).x;
    f32_t total = line0_width + line1_width;
    if (total <= size.x) {
      max_line_width = std::max(total, max_line_width);
      return true;
    }
    if (line_index + 1 < uint32_t(1.0 / current_font_size)) {
      ++line_index;
      max_line_width = std::max(total, max_line_width);
      return true;
    }
    return false;
  }

  void calculate_font_size() {

    fan::vec2 size = get_size();
    f32_t scaler = size.y * 2 / gloco->font.info.height;
    for (uint32_t i = 0; i < uint32_t(1.0 / current_font_size); ++i) {
      f32_t new_font_size = gloco->font.info.size * scaler * current_font_size;
      f32_t text_width = gloco->text.get_text_size(m_text_lines[i].get_text(), new_font_size).x / 2;
      new_font_size *= std::min(1.f, size.x / text_width);
      m_text_lines[i].set_font_size(new_font_size);
    }
  }

  void set_size(const fan::vec2& size) {
    shape_t::set_size(size);
    calculate_font_size();
  }

  // todo remake add push_letter to text renderer
  void push_letter(wchar_t wc) {
    std::wstring ws(&wc, &wc + 1);
    fan::string utf8(ws.begin(), ws.end());
    push_back(utf8);
   // calculate_font_size();
  }

  void push_back(const fan::string& text) {
    m_text_lines[line_index].set_text(m_text_lines[line_index].get_text() + text);
  }

  void erase() {
    shape_t::erase();
    m_text_lines.clear();
  }

  template <typename T>
  static responsive_text_custom_t::properties_t<T> make_properties(T shape_info, const loco_t::text_t::properties_t& tp) {
      return responsive_text_custom_t::properties_t<T>{tp, shape_info};
  }
  f32_t current_font_size = 0;
  std::vector<loco_t::shape_t> m_text_lines;
  uint32_t line_index = 0;
  f32_t max_line_width = 0;
};

struct responsive_text_t {

  struct properties_t : loco_t::text_t::properties_t {
    fan::vec2 boundary;
  };

  //bool does_text_fit(const fan::string& text) {
  //  f32_t text_width = m_text.get_text_size().x;
  //  return 
  //}

  responsive_text_t(const properties_t& properties) {
    auto& p = *(loco_t::text_t::properties_t*)&properties;
    p.position.z += 1;

    current_font_size = properties.font_size;

    m_text_lines.resize(1.0 / current_font_size);

    m_boundary = properties.boundary;

    for (uint32_t i = 0; i < m_text_lines.size(); ++i) { //                                                                                   fix when font size 1
      p.position = *(fan::vec2*)&properties.position + fan::vec2(0, current_font_size * i * m_boundary.y * 2 /*- properties.shape_info.size.y / 2*/);
      m_text_lines[i] = p;
    }

    set_size(m_boundary);
  }
  responsive_text_t() {

  }

  bool does_text_fit(const fan::string& text) {
    fan::vec2 size = m_boundary;
    f32_t font_size = m_text_lines[line_index].get_font_size();
    f32_t line0_width = gloco->text.get_text_size(m_text_lines[line_index].get_text(), font_size).x;
    f32_t line1_width = gloco->text.get_text_size(text, font_size).x;
    f32_t total = line0_width + line1_width;
    if (total <= size.x) {
      max_line_width = std::max(total, max_line_width);
      return true;
    }
    if (line_index + 1 < uint32_t(1.0 / current_font_size)) {
      ++line_index;
      max_line_width = std::max(total, max_line_width);
      return true;
    }
    return false;
  }

  void calculate_font_size() {

    fan::vec2 size = m_boundary;
    f32_t scaler = size.y * 2 / gloco->font.info.height;
    for (uint32_t i = 0; i < uint32_t(1.0 / current_font_size); ++i) {
      f32_t new_font_size = gloco->font.info.size * scaler * current_font_size;
      f32_t text_width = gloco->text.get_text_size(m_text_lines[i].get_text(), new_font_size).x / 2;
      new_font_size *= std::min(1.f, size.x / text_width);
      m_text_lines[i].set_font_size(new_font_size);
    }
  }

  void set_position(const fan::vec3& v) {
    // todo fix
    for (auto& i : m_text_lines) {
      i.set_position(v);
    }
  }
  void set_size(const fan::vec2& size) {
    m_boundary = size;
    calculate_font_size();
  }

  // todo remake add push_letter to text renderer
  void push_letter(wchar_t wc) {
    std::wstring ws(&wc, &wc + 1);
    fan::string utf8(ws.begin(), ws.end());
    push_back(utf8);
   // calculate_font_size();
  }

  void push_back(const fan::string& text) {
    m_text_lines[line_index].set_text(m_text_lines[line_index].get_text() + text);
  }

  void erase() {
    m_text_lines.clear();
  }

  f32_t current_font_size = 0;
  std::vector<loco_t::shape_t> m_text_lines;
  uint32_t line_index = 0;
  f32_t max_line_width = 0;
  fan::vec2 m_boundary;
};