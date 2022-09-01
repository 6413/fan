struct text_renderer_t {

  struct properties_t {

    using type_t = text_renderer_t;

    f32_t font_size = 0.1;
    fan::vec3 position = 0;
    fan::color color = fan::colors::white;
    fan::color outline_color = fan::colors::black;
    f32_t outline_size = 0.5;
    std::string text;

    union {
      struct {
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };
      loco_t::letter_t::instance_properties_t instance_properties;
    };
  };

  loco_t* get_loco() {
    loco_t* loco = OFFSETLESS(this, loco_t, sb_shape_var_name);
    return loco;
  }

  /* struct id_t{
    id_t(fan::opengl::cid_t* cid) {
      block = cid->id / letter_t::max_instance_size;
      instance = cid->id % letter_t::max_instance_size;
    }
    uint32_t block;
    uint32_t instance;
  };*/

  void open() {
    letter_ids.open();
    e.amount = 0;
  }
  void close() {
    for (uint32_t i = 0; i < letter_ids.size(); i++) {
      cid_close(&letter_ids[i]);
    }
    letter_ids.close();
  }

  f32_t convert_font_size(f32_t font_size) {
    loco_t* loco = get_loco();
    return font_size / loco->font.info.size;
  }

  fan::vec2 get_text_size(const std::string& text, f32_t font_size) {
    loco_t* loco = get_loco();
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

    return text_size * convert_font_size(font_size);
  }
  fan::vec2 get_text_size(uint32_t id) {
    loco_t* loco = get_loco();
    fan::vec2 text_size = 0;

    text_size.y = loco->font.info.line_height;

    f32_t width = 0;
    f32_t font_size = 0;

    auto it = cid_GetNodeFirst(&letter_ids[id]);

    while (it != letter_ids[id].dst) {
      auto node = cid_GetNodeByReference(&letter_ids[id], it);

      auto p = loco->letter.get_properties(&node->data.cid);

      font_size = p.font_size;
      auto letter =  loco->font.info.get_letter_info(p.letter_id, font_size);
      if (node->NextNodeReference == letter_ids[id].dst) {
        width += letter.glyph.size.x;
      }
      else {
        width += letter.metrics.advance / convert_font_size(font_size);
      }

      it = node->NextNodeReference;
    }

    text_size.x = std::max(width, text_size.x);

    return text_size * convert_font_size(font_size);
  }

  uint32_t push_back(properties_t properties) {
    loco_t* loco = get_loco();
    typename loco_t::letter_t::properties_t p;
    p.color = properties.color;
    p.font_size = properties.font_size;
    p.viewport = properties.viewport;
    p.matrices = properties.matrices;
    p.outline_color = properties.outline_color;
    p.outline_size = properties.outline_size;
    uint32_t id;
    if (e.amount != 0) {
      id = e.id0;
      e.id0 = *(uint32_t*)&letter_ids[e.id0];
      e.amount--;
    }
    else {
      id = letter_ids.resize(letter_ids.size() + 1);
    }
    cid_open(&letter_ids[id]);

    fan::vec2 text_size = get_text_size(properties.text, properties.font_size);
    f32_t left = properties.position.x - text_size.x / 2;

    for (uint32_t i = 0; i < properties.text.size(); i++) {
      p.letter_id = loco->font.decode_letter(properties.text[i]);
      auto letter_info = loco->font.info.get_letter_info(p.letter_id, properties.font_size);

      p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, properties.font_size - letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));
      p.position.z = properties.position.z;

      auto nr = cid_NewNodeLast(&letter_ids[id]);
      auto n = cid_GetNodeByReference(&letter_ids[id], nr);
      loco->letter.push_back(&n->data.cid, p);
      left += letter_info.metrics.advance;
    }
    return id;
  }

  void erase(uint32_t id) {
    loco_t* loco = get_loco();

    auto it = cid_GetNodeFirst(&letter_ids[id]);

    while (it != letter_ids[id].dst) {
      auto node = cid_GetNodeByReference(&letter_ids[id], it);
      loco->letter.erase(&node->data.cid);

      it = node->NextNodeReference;
    }
    cid_close(&letter_ids[id]);
    *(uint32_t*)&letter_ids[id] = e.id0;
    e.id0 = id;
    e.amount++;
  }

  template <typename T>
  T get(uint32_t id, T loco_t::letter_t::instance_t::*member) {
    loco_t* loco = get_loco();
    return loco->letter.get(&letter_ids[id][0], member);
  }
  template <typename T, typename T2>
  void set(uint32_t id, T loco_t::letter_t::instance_t::*member, const T2& value) {
    loco_t* loco = get_loco();
    fan::vec2 text_size;
    f32_t left;
    if constexpr (std::is_same<T, fan::vec3>::value) {
      text_size = get_text_size(id);
      left = ((f32_t*)&value)[0] - text_size.x / 2;
    }
      
    auto it = cid_GetNodeFirst(&letter_ids[id]);

    while (it != letter_ids[id].dst) {
      auto node = cid_GetNodeByReference(&letter_ids[id], it);

      auto p = loco->letter.get_properties(&node->data.cid);
      loco->letter.erase(&node->data.cid);
      ;
      if constexpr(std::is_same<T, fan::vec3>::value)
      if (fan::ofof(member) == fan::ofof(&loco_t::letter_t::instance_t::position)) {
        auto letter_info = loco->font.info.get_letter_info(p.letter_id, p.font_size);
        p.position = fan::vec2(left - letter_info.metrics.offset.x, ((f32_t*)&value)[1]) + (fan::vec2(letter_info.metrics.size.x, p.font_size - letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));
        p.position.z = value.z;
        loco->letter.push_back(&node->data.cid, p);
        left += letter_info.metrics.advance;
      }
      if (fan::ofof(member) != fan::ofof(&loco_t::letter_t::instance_t::position)) {
        p.*member = value;
        loco->letter.push_back(&node->data.cid, p);
      }
      it = node->NextNodeReference;
    }
  }

  void set_matrices(uint32_t id, fan::opengl::matrices_list_NodeReference_t n) {
    loco_t* loco = get_loco();

    auto it = cid_GetNodeFirst(&letter_ids[id]);

    while (it != letter_ids[id].dst) {
      auto node = cid_GetNodeByReference(&letter_ids[id], it);
      loco->letter.set_matrices(&node->data.cid, n);
      it = node->NextNodeReference;
    }
  }

  void set_viewport(uint32_t id, fan::opengl::viewport_list_NodeReference_t n) {
    loco_t* loco = get_loco();

    auto it = cid_GetNodeFirst(&letter_ids[id]);

    while (it != letter_ids[id].dst) {
      auto node = cid_GetNodeByReference(&letter_ids[id], it);
      loco->letter.set_viewport(&node->data.cid, n);
      it = node->NextNodeReference;
    }
  }

  //void set_position(loco_t* loco, uint32_t id, const fan::vec2& position) {
  //  for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
  //    auto p = loco->letter.get_properties(loco, &letter_ids[id][i]);
  //    loco->letter.erase(loco, &letter_ids[id][i]);
  //    auto letter_info = loco->font.info.get_letter_info(p.letter_id, p.font_size);

  //    p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, properties.font_size - letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));
  //    p.position.z = properties.position.z;
  //    p.*member = value;
  //    push_back(loco, p, &letter_ids[id][i]);
  //  }
  //}

  struct{
    uint16_t id0;

    uint32_t amount;
  }e;


  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid
  #define BLL_set_type_node uint16_t
  #define BLL_set_node_data fan::opengl::cid_t cid;
  #define BLL_set_Link 1
  #define BLL_set_StoreFormat 1
  #include _FAN_PATH(BLL/BLL.h)

  fan::hector_t<cid_t> letter_ids;
};