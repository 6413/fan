struct text_renderer_t {

  struct properties_t {

    using type_t = text_renderer_t;

    f32_t font_size = 0.1;
    fan::vec3 position = 0;
    fan::color color = fan::colors::white;
    fan::color outline_color;
    f32_t outline_size;
    fan::utf16_string text;

    union {
      struct {
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };
      loco_t::letter_t::instance_properties_t instance_properties;
    };
  };

  /* struct id_t{
    id_t(fan::opengl::cid_t* cid) {
      block = cid->id / letter_t::max_instance_size;
      instance = cid->id % letter_t::max_instance_size;
    }
    uint32_t block;
    uint32_t instance;
  };*/

  void open(loco_t* loco) {
    letter_ids.open();
    e.amount = 0;
  }
  void close(loco_t* loco) {
    for (uint32_t i = 0; i < letter_ids.size(); i++) {
      letter_ids[i].close();
    }
    letter_ids.close();
  }

  f32_t convert_font_size(loco_t* loco, f32_t font_size) {
    return font_size / loco->font.info.size;
  }

  fan::vec2 get_text_size(loco_t* loco, const fan::utf16_string& text, f32_t font_size) {
    fan::vec2 text_size = 0;

    text_size.y = loco->font.info.line_height;

    f32_t width = 0;

    for (int i = 0; i < text.size(); i++) {

      auto letter = loco->font.info.characters[text[i]];

      if (i == text.size() - 1) {
        width += letter.glyph.size.x;
      }
      else {
        width += letter.metrics.advance;
      }
    }

    text_size.x = std::max(width, text_size.x);

    return text_size * convert_font_size(loco, font_size);
  }

  uint32_t push_back(loco_t* loco, properties_t properties) {
    typename loco_t::letter_t::properties_t p;
    p.color = properties.color;
    p.font_size = properties.font_size;
    p.viewport = properties.viewport;
    p.matrices = properties.matrices;
    uint32_t id;
    if (e.amount != 0) {
      id = e.id0;
      e.id0 = *(uint32_t*)&letter_ids[e.id0];
      e.amount--;
    }
    else {
      id = letter_ids.resize(letter_ids.size() + 1);
    }
    letter_ids[id].open();

    fan::vec2 text_size = get_text_size(loco, properties.text, properties.font_size);
    f32_t left = properties.position.x - text_size.x / 2;

    for (uint32_t i = 0; i < properties.text.size(); i++) {
      p.letter_id = loco->font.decode_letter(properties.text[i]);
      auto letter_info = loco->font.info.get_letter_info(properties.text[i], properties.font_size);

      p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, properties.font_size - letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));
      p.position.z = properties.position.z;

      letter_ids[id].resize(letter_ids[id].size() + 1);
      loco->letter.push_back(loco, &letter_ids[id][letter_ids[id].size() - 1], p);
      left += letter_info.metrics.advance;
    }
    return id;
  }
  void erase(loco_t* loco, uint32_t id) {
    for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
      loco->letter.erase(loco, &letter_ids[id][i]);
    }
    letter_ids[id].close();
    *(uint32_t*)&letter_ids[id] = e.id0;
    e.id0 = id;
    e.amount++;
  }

  template <typename T>
  T get(loco_t* loco, uint32_t id, T loco_t::letter_t::instance_t::*member) {
    return loco->letter.get(loco, &letter_ids[id][0], member);
  }
  template <typename T, typename T2>
  void set(loco_t* loco, uint32_t id, T loco_t::letter_t::instance_t::*member, const T2& value) {
    for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
      auto current = loco->letter.get(loco, &letter_ids[id][i], member);
      loco->letter.set(loco, &letter_ids[id][i], member, current - (current - value));
    }
  }

  struct{
    uint16_t id0;

    uint32_t amount;
  }e;

  fan::hector_t<fan::hector_t<fan::opengl::cid_t>> letter_ids;
};