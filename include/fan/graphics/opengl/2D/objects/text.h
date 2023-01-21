struct text_renderer_t {

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  //using ri_t = loco_t::letter_t::ri_t;
  using vi_t = loco_t::letter_t::vi_t;
  using ri_t = loco_t::letter_t::ri_t;

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
    fan::string text;
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

  fan::vec2 get_text_size(const fan::string& text, f32_t font_size) {
    loco_t* loco = get_loco();
    fan::vec2 text_size = 0;

    text_size.y = loco->font.info.get_line_height(font_size);


    for (int i = 0; i < text.utf8_size(); i++) {
      auto letter = loco->font.info.get_letter_info(text.get_utf8(i), font_size);

      //auto p = letter_info.metrics.offset.x + letter_info.metrics.size.x / 2 + letter_info.metrics.offset.x;

      text_size.x += letter.metrics.size.x + letter.metrics.offset.x;
      if (i + 1 != text.size()) {
        text_size.x += letter.metrics.offset.x;
      }
    }

    return text_size;
  }
  fan::vec2 get_text_size(fan::graphics::cid_t* id) {
    return get_text_size(letter_ids[*(uint32_t*)id].p.text, letter_ids[*(uint32_t*)id].p.font_size);
   /* loco_t* loco = get_loco();
    fan::vec2 text_size = 0;

    text_size.y = loco->font.info.line_height;

    f32_t font_size = 0;

    for (uint32_t i = 0; i < letter_ids[id].p.text.size(); i++) {
      font_size = letter_ids[id].p.font_size;
      auto letter = loco->font.info.get_letter_info(loco->font.decode_letter(letter_ids[id].p.text[i]), font_size);

      text_size.x += letter.metrics.size.x + letter.metrics.offset.x;
      if (i + 1 != letter_ids[id].p.text.size()) {
        text_size.x += letter.metrics.offset.x;
      }
    }

    return text_size;*/
  }

  void push_back(properties_t properties, fan::graphics::cid_t* cid) {
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
    f32_t advance = 0;
    for (uint32_t i = 0; i < properties.text.utf8_size(); i++) {
      p.letter_id = properties.text.get_utf8(i);
      auto letter_info = loco->font.info.get_letter_info(p.letter_id, properties.font_size);

      //uint8_t t = properties.text[i];
      //properties.text[i] = 0;
      p.position = fan::vec2(
        left + advance + letter_info.metrics.size.x / 2,
        properties.position.y + (properties.font_size - letter_info.metrics.size.y) / 2 - letter_info.metrics.offset.y
      );
      //fan::print(letter_info.metrics.size.x, letter_info.metrics.offset.x, left, get_text_size(&properties.text[0], properties.font_size).x, p.position.x);
      //properties.text[i] = t;
      p.position.z = properties.position.z;

      /*if (i == 0) {
        p.position.x -= letter_info.metrics.size.x;
      }*/
      auto nr = letter_ids[id].cid_list.NewNodeLast();
      auto n = letter_ids[id].cid_list.GetNodeByReference(nr);
      loco->letter.push_back(&n->data.cid, p);
      advance += letter_info.metrics.advance;
      //left += letter_info.metrics.advance;
    }
    *cid = (fan::opengl::cid_t)id;
  }

  void erase(fan::graphics::cid_t* cid) {
    loco_t* loco = get_loco();

    auto it = letter_ids[*(uint32_t*)cid].cid_list.GetNodeFirst();

    while (it != letter_ids[*(uint32_t*)cid].cid_list.dst) {
      auto node = letter_ids[*(uint32_t*)cid].cid_list.GetNodeByReference(it);
      loco->letter.erase(&node->data.cid);

      it = node->NextNodeReference;
    }
    letter_ids[*(uint32_t*)cid].cid_list.Close();
    *(uint32_t*)&letter_ids[*(uint32_t*)cid] = e.id0;
    e.id0 = *(uint32_t*)cid;
    e.amount++;
  }

  //template <typename T>
  //T get(fan::graphics::cid_t* cid, T loco_t::letter_t::vi_t::*member) {
  //  loco_t* loco = get_loco();
  //  return loco->letter.get(*(uint32_t*)cid, member); // ?
  //}
  // do not use with set_position
  void set(fan::graphics::cid_t* cid, auto member, auto value) {
    loco_t* loco = get_loco();
    auto it = letter_ids[*(uint32_t*)cid].cid_list.GetNodeFirst();

    while (it != letter_ids[*(uint32_t*)cid].cid_list.dst) {
      auto node = letter_ids[*(uint32_t*)cid].cid_list.GetNodeByReference(it);
      loco->letter.set(&node->data.cid, member, value);
      it = node->NextNodeReference;
    }
  }

  void set_matrices(fan::graphics::cid_t* cid, loco_t::matrices_list_NodeReference_t n) {
    loco_t* loco = get_loco();

    auto it = letter_ids[*(uint32_t*)cid].cid_list.GetNodeFirst();

    while (it != letter_ids[*(uint32_t*)cid].cid_list.dst) {
      auto node = letter_ids[*(uint32_t*)cid].cid_list.GetNodeByReference(it);
      loco->letter.set_matrices(&node->data.cid, n);
      it = node->NextNodeReference;
    }
  }

  void set_viewport(fan::graphics::cid_t* cid, fan::graphics::viewport_list_NodeReference_t n) {
    loco_t* loco = get_loco();

    auto it = letter_ids[*(uint32_t*)cid].cid_list.GetNodeFirst();

    while (it != letter_ids[*(uint32_t*)cid].cid_list.dst) {
      auto node = letter_ids[*(uint32_t*)cid].cid_list.GetNodeByReference(it);
      loco->letter.set_viewport(&node->data.cid, n);
      it = node->NextNodeReference;
    }
  }

  void set_depth(fan::graphics::cid_t* cid, f32_t depth) {
    loco_t* loco = get_loco();

    auto it = letter_ids[*(uint32_t*)cid].cid_list.GetNodeFirst();

    while (it != letter_ids[*(uint32_t*)cid].cid_list.dst) {
      auto node = letter_ids[*(uint32_t*)cid].cid_list.GetNodeByReference(it);
      loco->letter.set_depth(&node->data.cid, depth);
      it = node->NextNodeReference;
    }
  }

  //void set_position(loco_t* loco, uint32_t id, const fan::vec2& position) {
  //  for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
  //    auto p = loco->letter.get_instance(loco, &letter_ids[id][i]);
  //    loco->letter.erase(loco, &letter_ids[id][i]);
  //    auto letter_info = loco->font.info.get_letter_info(p.letter_id, p.font_size);

  //    p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, properties.font_size - letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));
  //    p.position.z = properties.position.z;
  //    p.*member = value;
  //    push_back(loco, p, &letter_ids[id][i]);
  //  }
  //}

  f32_t get_font_size(fan::graphics::cid_t* cid) {
    auto loco = get_loco();
    auto it = letter_ids[*(uint32_t*)cid].cid_list.GetNodeFirst();
    auto node = letter_ids[*(uint32_t*)cid].cid_list.GetNodeByReference(it);
    return loco->letter.sb_get_ri(&node->data.cid).font_size;
  }

  auto get_matrices(fan::graphics::cid_t* cid) {
    auto loco = get_loco();
    auto it = letter_ids[*(uint32_t*)cid].cid_list.GetNodeFirst();
    auto node = letter_ids[*(uint32_t*)cid].cid_list.GetNodeByReference(it);
    return loco->letter.get_matrices(&node->data.cid);
  }

  properties_t get_instance(fan::graphics::cid_t* cid) {
    return letter_ids[*(uint32_t*)cid].p;
  }
  void set_text(fan::graphics::cid_t* cid, const fan::string& text) {
    properties_t p = letter_ids[*(uint32_t*)cid].p;
    erase(cid);
    p.text = text;

    push_back(p, cid);
  }

  void set_position(fan::graphics::cid_t* cid, const fan::vec3& position) {
    properties_t p = letter_ids[*(uint32_t*)cid].p;
    erase(cid);
    p.position = position;
    push_back(p, cid);
  }

  void set_font_size(fan::graphics::cid_t* cid, f32_t font_size) {
    properties_t p = letter_ids[*(uint32_t*)cid].p;
    erase(cid);
    p.font_size = font_size;
    push_back(p, cid);
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