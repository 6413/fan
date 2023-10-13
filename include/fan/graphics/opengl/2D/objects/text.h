struct text_renderer_t {

  static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::text;

  //using ri_t = loco_t::letter_t::ri_t;
  using vi_t = loco_t::letter_t::vi_t;
  using ri_t = loco_t::letter_t::ri_t;

  struct properties_t : vi_t, ri_t{
    using type_t = text_renderer_t;

    loco_text_properties_t
  };

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData loco_t::shape_t shape;
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  #define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix tlist
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData \
    text_renderer_t::cid_list_t cid_list; \
    properties_t p;
    
    
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  tlist_t tlist;

  text_renderer_t() {

  }

  f32_t convert_font_size(f32_t font_size) {
    return font_size / gloco->font.info.size;
  }

  fan::string get_text(loco_t::cid_nt_t& id) {
    return get_properties(id).text;
  }

  fan::vec2 get_text_size(const fan::string& text, f32_t font_size) {
    fan::vec2 text_size = 0;

    text_size.y = gloco->font.info.get_line_height(font_size);


    for (int i = 0; i < text.utf8_size(); i++) {
      auto letter = gloco->font.info.get_letter_info(text.get_utf8(i), font_size);
      
      //auto p = letter_info.metrics.offset.x + letter_info.metrics.size.x / 2 + letter_info.metrics.offset.x;
      text_size.x += letter.metrics.advance;
      //text_size.x += letter.metrics.size.x + letter.metrics.offset.x;
      //if (i + 1 != text.size()) {
      //  text_size.x += letter.metrics.offset.x;
      //}
    }

    return text_size;
  }
  fan::vec2 get_text_size(loco_t::cid_nt_t& id) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    auto& instance = tlist[internal_id];
    if (!instance.cid_list.Usage()) {
      return 0;
    }
    auto& src_id = instance.cid_list[instance.cid_list.GetNodeFirst()].shape;
    auto& dst_id = instance.cid_list[instance.cid_list.GetNodeLast()].shape;
    fan::vec2 src = src_id.get_position();
    fan::vec2 dst = dst_id.get_position();
    return (dst + dst_id.get_size()) - (src - src_id.get_size());
  }

  void push_back(loco_t::cid_nt_t& id, properties_t properties) {
   
    tlist_NodeReference_t internal_id = tlist.NewNodeLast();
    tlist[internal_id].p = properties;

    fan::vec2 text_size = get_text_size(properties.text, properties.font_size);
    f32_t left = properties.position.x - text_size.x / 2;
    f32_t advance = 0;
    for (uint32_t i = 0; i < properties.text.utf8_size(); i++) {
    
      append_letter(id, properties.text.get_utf8(i), internal_id, left, advance);
      //left += letter_info.metrics.advance;
    }
    id->shape_type = loco_t::shape_type_t::text;
    *id.gdp4() = internal_id.NRI;
  }

  void erase(loco_t::cid_nt_t& id) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    tlist.unlrec(internal_id);
  }

  void append_letter(loco_t::cid_nt_t& id, uint32_t letter, tlist_NodeReference_t internal_id, f32_t left, f32_t& advance){

    const auto& instance_properties = tlist[internal_id].p;

    typename loco_t::letter_t::properties_t p;
    p.color = instance_properties.color;
    p.font_size = instance_properties.font_size;
    p.viewport = instance_properties.viewport;
    p.camera = instance_properties.camera;
    p.outline_color = instance_properties.outline_color;
    p.outline_size = instance_properties.outline_size;

    p.letter_id = letter;
    auto letter_info = gloco->font.info.get_letter_info(p.letter_id, instance_properties.font_size);
    p.position = fan::vec2(
      left + advance + letter_info.metrics.size.x / 2,
      instance_properties.position.y + (instance_properties.font_size - letter_info.metrics.size.y) / 2 - letter_info.metrics.offset.y
    );
    p.position.z = instance_properties.position.z;
    auto nr = tlist[internal_id].cid_list.NewNodeLast();
    auto n = tlist[internal_id].cid_list.GetNodeByReference(nr);
    n->data.shape = p;
    advance += letter_info.metrics.advance;
  }

  // do not use with set_position
  void set(loco_t::cid_nt_t& id, auto member, auto value) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();

    tlist[internal_id].p.*member = value;

    auto it = tlist[internal_id].cid_list.GetNodeFirst();

    while (it != tlist[internal_id].cid_list.dst) {
      auto node = tlist[internal_id].cid_list.GetNodeByReference(it);
      gloco->letter.set(node->data.shape, member, value);
      it = node->NextNodeReference;
    }
  }

  void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    auto it = tlist[internal_id].cid_list.GetNodeFirst();

    while (it != tlist[internal_id].cid_list.dst) {
      auto node = tlist[internal_id].cid_list.GetNodeByReference(it);
      gloco->letter.set_camera(node->data.shape, n);
      it = node->NextNodeReference;
    }
  }

  void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    auto it = tlist[internal_id].cid_list.GetNodeFirst();

    while (it != tlist[internal_id].cid_list.dst) {
      auto node = tlist[internal_id].cid_list.GetNodeByReference(it);
      gloco->letter.set_viewport(node->data.shape, n);
      it = node->NextNodeReference;
    }
  }

  void sb_set_depth(loco_t::cid_nt_t& id, f32_t depth) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    auto it = tlist[internal_id].cid_list.GetNodeFirst();

    while (it != tlist[internal_id].cid_list.dst) {
      auto node = tlist[internal_id].cid_list.GetNodeByReference(it);
      gloco->letter.sb_set_depth(node->data.shape, depth);
      it = node->NextNodeReference;
    }
  }

  tlist_NodeReference_t get_internal_id(loco_t::cid_nt_t& id) {
    return *(tlist_NodeReference_t *)id.gdp4();
  }

  void set_depth(loco_t::cid_nt_t& id, f32_t depth) {
    sb_set_depth(id, depth);
  }

  f32_t get_font_size(loco_t::cid_nt_t& id) {
    return get_instance(id).font_size;
  }

  auto get_camera(loco_t::cid_nt_t& id) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    auto it = tlist[internal_id].cid_list.GetNodeFirst();
    #if fan_debug >= 2
    if (it == tlist[internal_id].cid_list.dst) {
      fan::throw_error("empty string");
    }
    #endif
    auto node = tlist[internal_id].cid_list.GetNodeByReference(it);
    return gloco->letter.get_camera(node->data.shape);
  }

  properties_t get_instance(loco_t::cid_nt_t& id) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    return tlist[internal_id].p;
  }
  void set_text(loco_t::cid_nt_t& id, const fan::string& text) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    properties_t p = tlist[internal_id].p;
    erase(id);
    p.text = text;

    push_back(id, p);
  }

  void set_position(loco_t::cid_nt_t& id, const fan::vec3& position) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    properties_t p = tlist[internal_id].p;
    erase(id);
    p.position = position;
    push_back(id, p);
  }

  void set_font_size(loco_t::cid_nt_t& id, f32_t font_size) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    properties_t p = tlist[internal_id].p;
    erase(id);
    p.font_size = font_size;
    push_back(id, p);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    auto internal_id = *(tlist_NodeReference_t *)id.gdp4();
    return tlist[internal_id].p;
  }

  fan::vec2 get_size(loco_t::cid_nt_t& id) {
    return gloco->text.get_text_size(id) / 2;
  }
  

};