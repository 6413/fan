struct responsive_text_t {

  struct properties_t : loco_t::text_t::properties_t {
    fan::vec2 boundary;
  };

  //bool does_text_fit(const fan::string& text) {
  //  f32_t text_width = m_text.get_text_size().x;
  //  return 
  //}

  void calculate_text_positions() {

    for (uint32_t i = 0; i < m_text_lines.size(); ++i) {
      f32_t yo = current_font_size * i * m_boundary.y * 2 - m_boundary.y * (1 - current_font_size);
      fan::vec3 p = *(fan::vec2*)&m_position + fan::vec2(0, yo);
      p.z = m_position.z;
      m_text_lines[i].set_position(p);
    }
  }

  responsive_text_t(const properties_t& properties) {
    m_boundary = properties.boundary;
    m_position = properties.position;

    auto p = *(loco_t::text_t::properties_t*)&properties;
    p.position.z += 1;
    m_position.z = p.position.z;

    current_font_size = properties.font_size;
    p.text.clear();
    m_text_lines.resize(1.0 / current_font_size, p);

    fan::time::clock c;
    f32_t left = 400;
    f32_t advance = 0;
    // feed letters in loop
    loco_t::text_t::tlist_NodeReference_t internal_id;
    for (uint32_t i = 0; i < m_text_lines.size(); ++i) {
      loco_t::text_t::tlist_NodeReference_t ii = gloco->text.tlist.NewNodeLast();
      if (i == line_index) {
        internal_id = ii;
      }
      gloco->text.tlist[ii].p = *(loco_t::text_t::properties_t*)&properties;
      m_text_lines[i]->shape_type = loco_t::shape_type_t::text;
      *m_text_lines[i].gdp4() = ii.NRI;
    }

    for (auto& i : properties.text) {
      push_letter_force(internal_id, left, advance, i);
    }

    set_size(m_boundary);
  }
  responsive_text_t() = default;

  bool does_letter_fit(uint32_t wc, bool force = false) {
    f32_t scaler = get_scaler();
    fan::vec2 size = get_size() * 2;
    f32_t font_size = gloco->font.info.size * scaler;
    f32_t line0_width = gloco->text.get_text_size(m_text_lines[line_index].get_text(),
      font_size).x;
    f32_t line1_width = gloco->font.info.get_letter_info(wc, font_size).metrics.size.x;
    f32_t total = line0_width + line1_width;
    if (total <= size.x) {
      return true;
    }
    if (line_index + 1 < uint32_t(1.0 / current_font_size)) {
      ++line_index;
      return true;
    }

    if (force == true) {
      return true;
    }

    return false;
  }

  bool does_text_fit(const fan::string& text, bool force = false) {
    f32_t scaler = get_scaler();
    fan::vec2 size = get_size() * 2;
    f32_t font_size = gloco->font.info.size * scaler;
    f32_t line0_width = gloco->text.get_text_size(m_text_lines[line_index].get_text(), font_size).x;
    f32_t line1_width = gloco->text.get_text_size(text, font_size).x;
    f32_t total = line0_width + line1_width;
    if (total <= size.x) {
      return true;
    }
    if (line_index + 1 < uint32_t(1.0 / current_font_size)) {
      ++line_index;
      return true;
    }

    if (force == true) {
      return true;
    }

    return false;
  }

  f32_t get_scaler() {
    fan::vec2 size = get_size();
    f32_t scaler = current_font_size * size.y * 2 / gloco->font.info.height;
    f32_t biggest = 0;
    for (uint32_t i = 0; i < uint32_t(1.0 / current_font_size); ++i) {
      f32_t new_font_size = gloco->font.info.size * scaler;
      f32_t text_width = gloco->text.get_text_size(m_text_lines[i].get_text(), new_font_size).x / 2;
      biggest = std::max(biggest, text_width);
    }
    scaler *= std::min(size.x / biggest, 1.f);
    return scaler;
  }

  void calculate_font_size() {
    f32_t scaler = get_scaler();
    for (uint32_t i = 0; i < uint32_t(1.0 / current_font_size); ++i) {
      m_text_lines[i].set_font_size(gloco->font.info.size * scaler);
    }
  }

  void set_position(const fan::vec3& v) {
    m_position = v;

    calculate_text_positions();
  }
  fan::vec2 get_size() const {
    return m_boundary;
  }
  void set_size(const fan::vec2& size) {
    m_boundary = size;
    calculate_font_size();
    calculate_text_positions();
  }

  void push_letter_force(loco_t::text_t::tlist_NodeReference_t nr, f32_t left, f32_t& advance, uint32_t wc) {
    does_letter_fit(wc, true);
    // might need properties no idea
    gloco->text.append_letter(m_text_lines[line_index], wc, nr, left, advance);
  }

  bool push_back(const fan::string& text) {
    if (does_text_fit(text) == false) {
      return false;
    }
    m_text_lines[line_index].set_text(m_text_lines[line_index].get_text() + text);
    return true;
  }

  void erase() {
    m_text_lines.clear();
  }

  f32_t current_font_size = 0;
  std::vector<loco_t::shape_t> m_text_lines;
  uint32_t line_index = 0;
  fan::vec3 m_position;
  fan::vec2 m_boundary;
};