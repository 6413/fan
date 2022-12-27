struct text_renderer_t {

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : loco_t::letter_t::ri_t{

    properties_t() = default;
    properties_t(const properties_t& p) : properties_t() {
      *(loco_t::letter_t::ri_t*)this = *(loco_t::letter_t::ri_t*)&p;
      position = p.position;
      color = p.color;
      outline_color = p.outline_color;
      outline_size = p.outline_size;
      text = p.text;
    }
    properties_t& operator=(const properties_t& p) {
      this->~properties_t();
      new (this) properties_t(p);
      return *this;
    }

    make_key_value(loco_t::matrices_list_NodeReference_t, matrices);
    make_key_value(fan::graphics::viewport_list_NodeReference_t, viewport);

    fan::vec3 position = 0;
    fan::color color = fan::colors::white;
    fan::color outline_color = fan::colors::black;
    f32_t outline_size = 0.5;
    fan::wstring text;
  };

  #undef make_key_value

  loco_t* get_loco() {
    loco_t* loco = OFFSETLESS(this, loco_t, sb_shape_var_name);
    return loco;
  }

  /* struct nr_t{
    nr_t(fan::opengl::cid_t* cid) {
      block = cid->id / letter_t::max_instance_size;
      instance = cid->id % letter_t::max_instance_size;
    }
    uint32_t block;
    uint32_t instance;
  };*/

  text_renderer_t() {
    e.amount = 0;
  }
  ~text_renderer_t() {
    for (uint32_t i = 0; i < letter_ids.size(); i++) {
      letter_ids[i].cid_list.Close();
    }
  }

  f32_t convert_font_size(f32_t font_size) {
    loco_t* loco = get_loco();
    return font_size / loco->font.info.size;
  }

  fan::vec2 get_text_size(const fan::wstring& text, f32_t font_size) {
    loco_t* loco = get_loco();
    fan::vec2 text_size = 0;

    text_size.y = loco->font.info.line_height;

    f32_t width = 0;

    for (int i = 0; i < text.size(); i++) {

      auto letter = loco->font.info.characters[text[i]];

      if (i == text.size() - 1) {
        width += letter.metrics.advance;
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

    for (uint32_t i = 0; i < letter_ids[id].p.text.size(); i++) {
      font_size = letter_ids[id].p.font_size;
      auto letter = loco->font.info.get_letter_info(loco->font.decode_letter(letter_ids[id].p.text[i]), font_size);
      if (i == letter_ids[id].p.text.size() - 1) {
        width += letter.glyph.size.x;
      }
      else {
        width += letter.metrics.advance / convert_font_size(font_size);
      }
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
      letter_ids.resize(letter_ids.size() + 1);
      id = letter_ids.size() - 1;
    }
    letter_ids[id].cid_list.Open();
    letter_ids[id].p = properties;

    fan::vec2 text_size = get_text_size(properties.text, properties.font_size);
    f32_t left = properties.position.x - text_size.x / 2;

    for (uint32_t i = 0; i < properties.text.size(); i++) {
      p.letter_id = loco->font.decode_letter(properties.text[i]);
      auto letter_info = loco->font.info.get_letter_info(p.letter_id, properties.font_size);

      p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, properties.font_size - letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));
      p.position.z = properties.position.z;

      auto nr = letter_ids[id].cid_list.NewNodeLast();
      auto n = letter_ids[id].cid_list.GetNodeByReference(nr);
      loco->letter.push_back(&n->data.cid, p);
      left += letter_info.metrics.advance;
    }
    return id;
  }

  void erase(uint32_t id) {
    loco_t* loco = get_loco();

    auto it = letter_ids[id].cid_list.GetNodeFirst();

    while (it != letter_ids[id].cid_list.dst) {
      auto node = letter_ids[id].cid_list.GetNodeByReference(it);
      loco->letter.erase(&node->data.cid);

      it = node->NextNodeReference;
    }
    letter_ids[id].cid_list.Close();
    *(uint32_t*)&letter_ids[id] = e.id0;
    e.id0 = id;
    e.amount++;
  }

  template <typename T>
  T get(uint32_t id, T loco_t::letter_t::vi_t::*member) {
    loco_t* loco = get_loco();
    return loco->letter.get(id, member); // ?
  }
 /* template <typename T, typename T2>
  void set(uint32_t id, T loco_t::letter_t::instance_t::*member, const T2& value) {
    loco_t* loco = get_loco();
    fan::vec2 text_size;
    f32_t left;
    if constexpr (std::is_same<T, fan::vec3>::value) {
      text_size = get_text_size(id);
      left = ((f32_t*)&value)[0] - text_size.x / 2;
    }
      
    auto it = letter_ids[id].cid.GetNodeFirst();
    while (it != letter_ids[id].cid.dst) {
      auto node = letter_ids[id].cid.GetNodeByReference(it);

      auto p = get_properties(id);
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
  }*/

  // do not use with set_position
  void set(uint32_t id, auto member, auto value) {
    loco_t* loco = get_loco();
    auto it = letter_ids[id].cid_list.GetNodeFirst();

    while (it != letter_ids[id].cid_list.dst) {
      auto node = letter_ids[id].cid_list.GetNodeByReference(it);
      loco->letter.set(&node->data.cid, member, value);
      it = node->NextNodeReference;
    }
  }

  void set_matrices(uint32_t id, loco_t::matrices_list_NodeReference_t n) {
    loco_t* loco = get_loco();

    auto it = letter_ids[id].cid_list.GetNodeFirst();

    while (it != letter_ids[id].cid_list.dst) {
      auto node = letter_ids[id].cid_list.GetNodeByReference(it);
      loco->letter.set_matrices(&node->data.cid, n);
      it = node->NextNodeReference;
    }
  }

  void set_viewport(uint32_t id, fan::graphics::viewport_list_NodeReference_t n) {
    loco_t* loco = get_loco();

    auto it = letter_ids[id].cid_list.GetNodeFirst();

    while (it != letter_ids[id].cid_list.dst) {
      auto node = letter_ids[id].cid_list.GetNodeByReference(it);
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

  //f32_t get_font_size(uint32_t id) {
  //  auto loco = get_loco();
  //  auto it = letter_ids[id].GetNodeFirst();
  //  auto node = letter_ids[id].GetNodeByReference(it);
  //  return loco->letter.get_properties(&node->data.cid).font_size;
  //}

  properties_t get_properties(uint32_t id) {
    return letter_ids[id].p;
  }
  void set_text(uint32_t* id, const fan::wstring& text) {
    properties_t p = letter_ids[*id].p;
    erase(*id);
    p.text = text;

    *id = push_back(p);
  }

  void set_position(uint32_t* id, const fan::vec3& position) {
    properties_t p = letter_ids[*id].p;
    erase(*id);
    p.position = position;
    *id = push_back(p);
  }

  struct{
    uint32_t id0;
    uint32_t amount;
  }e;

  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData fan::graphics::cid_t cid;
  #define BLL_set_Link 1
  #define BLL_set_StoreFormat 1
  #include _FAN_PATH(BLL/BLL.h)

  struct instance_t {
    cid_list_t cid_list;
    properties_t p;
  };

  std::vector<instance_t> letter_ids;
};