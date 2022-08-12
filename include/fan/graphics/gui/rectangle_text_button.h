struct button_t {

  using be_t = fan_2d::graphics::gui::be_t;

  struct instance_t {
    fan::vec3 position = 0;
    f32_t angle = 0;
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::color outline_color;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t outline_size;
  };

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(instance_t) / 4));

  struct instance_properties_t {
    struct key_t : fan::masterpiece_t<
      fan::opengl::matrices_list_NodeReference_t,
      fan::opengl::viewport_list_NodeReference_t
    > {}key;

    fan::opengl::theme_list_NodeReference_t theme;
    uint32_t text_id;
    uint32_t be_id;
    uint32_t depth;
  };

  struct properties_t : instance_t {
    properties_t() : depth(0) {}

    fan::utf16_string text;
    f32_t font_size = 0.1;
    be_t::on_input_cb_t mouse_input_cb = [](const be_t::mouse_input_data_t&) -> uint8_t { return 1; };
    be_t::on_mouse_move_cb_t mouse_move_cb = [](const be_t::mouse_move_data_t&) -> uint8_t { return 1; };

    void* userptr;
    bool disable_highlight = false;

    union {
      struct {
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
        fan::opengl::theme_list_NodeReference_t theme;
        uint32_t text_id;
        uint32_t be_id;
        uint32_t depth;
      };
      instance_properties_t instance_properties;
    };
  };

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    auto theme = loco->button.get_theme(loco, p.theme);
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 0.001;
    tp.viewport = p.viewport;
    tp.matrices = p.matrices;
    auto block = sb_push_back(loco, cid, p);
    block->p[cid->instance_id].text_id = loco->text.push_back(loco, tp);

    set_theme(loco, cid, theme, 1);

    fan_2d::graphics::gui::be_t::properties_t be_p;
    be_p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
    be_p.hitbox_rectangle.position = p.position;
    be_p.hitbox_rectangle.size = p.size;
    be_p.on_input_cb = p.mouse_input_cb;
    be_p.on_mouse_event_cb = p.mouse_move_cb;
    be_p.userptr = p.userptr;
    be_p.shape_type = loco_t::shapes::button;
    fan::print("warning we do not want to allocate");
    fan::opengl::cid_t* c = new fan::opengl::cid_t(*cid);
    be_p.cid = c;
    be_p.viewport = fan::opengl::viewport_list_GetNodeByReference(&loco->get_context()->viewport_list, p.viewport)->data.viewport_id;
    #if fan_debug >= fan_debug_low
      if (p.depth >= loco->max_depths) {
        fan::throw_error("invalid access");
      }
    #endif
    if (p.disable_highlight) {
      block->p[cid->instance_id].be_id = loco->element_depth[p.depth].input_hitbox.push_back(
        be_p, 
        [](const be_t::mouse_input_data_t&)->uint8_t { return 1; /* continue*/ },
        [](const be_t::mouse_move_data_t&)->uint8_t { return 1; /* continue*/ }
      );
    }
    else {
      block->p[cid->instance_id].be_id = loco->element_depth[p.depth].input_hitbox.push_back(be_p, mouse_input_cb, mouse_move_cb);
    }
  }
  void erase(loco_t* loco, fan::opengl::cid_t* cid) {
    auto block = sb_get_block(loco, cid);
    instance_properties_t* p = &block->p[cid->instance_id];
    loco->text.erase(loco, p->text_id);
    delete loco->element_depth[p->depth].input_hitbox.m_button_data[p->be_id].properties.cid;
    loco->element_depth[p->depth].input_hitbox.erase(p->be_id);

    sb_erase(loco, cid);
  }

  void draw(loco_t* loco) {
    sb_draw(loco);
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open(loco_t* loco) {
    sb_open(loco);
  }
  void close(loco_t* loco) {
    sb_close(loco);
  }

  fan_2d::graphics::gui::theme_t* get_theme(loco_t* loco, fan::opengl::theme_list_NodeReference_t nr) {
    return fan::opengl::theme_list_GetNodeByReference(&loco->get_context()->theme_list, nr)->data.theme_id;
  }
  fan_2d::graphics::gui::theme_t* get_theme(loco_t* loco, fan::opengl::cid_t* cid) {
    auto block_node = bll_block_GetNodeByReference(&blocks, *(bll_block_NodeReference_t*)&cid->block_id);
    return get_theme(loco, block_node->data.block.p[cid->instance_id].theme);
  }
  void set_theme(loco_t* loco, fan::opengl::cid_t* cid, fan_2d::graphics::gui::theme_t* theme, f32_t intensity) {
    fan_2d::graphics::gui::theme_t t = *theme;
    t = t * intensity;
    set(loco, cid, &instance_t::color, t.button.color);
    set(loco, cid, &instance_t::outline_color, t.button.outline_color);
    set(loco, cid, &instance_t::outline_size, t.button.outline_size);
    auto block = sb_get_block(loco, cid);
    block->p[cid->instance_id].theme = theme;
    loco->text.set(loco, block->p[cid->instance_id].text_id, 
      &loco_t::letter_t::instance_t::outline_color, t.button.text_outline_color);
    loco->text.set(loco, block->p[cid->instance_id].text_id, 
      &loco_t::letter_t::instance_t::outline_size, t.button.text_outline_size);
  }

  template <typename T>
  T get_button(loco_t* loco, fan::opengl::cid_t* cid, T instance_t::* member) {
    return loco->button.get(loco, cid, member);
  }
  template <typename T, typename T2>
  void set_button(loco_t* loco, fan::opengl::cid_t* cid, T instance_t::*member, const T2& value) {
    loco->button.set(loco, cid, member, value);
  }

  template <typename T>
  T get_text(loco_t* loco, fan::opengl::cid_t* cid, T loco_t::letter_t::instance_t::* member) {
    auto block = sb_get_block(loco, cid);
    return loco->text.get(loco, block->p[cid->instance_id].text_id, member);
  }
  template <typename T, typename T2>
  void set_text(loco_t* loco, fan::opengl::cid_t* cid, T loco_t::letter_t::instance_t::*member, const T2& value) {
    auto block = sb_get_block(loco, cid);
    loco->text.set(loco, block->p[cid->instance_id].text_id, member, value);
  }

  void set_position(loco_t* loco, fan::opengl::cid_t* cid, const fan::vec2& position) {
    auto block = sb_get_block(loco, cid);
    set_text(loco, cid, &loco_t::letter_t::instance_t::position, position);
    set_button(loco, cid, &instance_t::position, position);
    loco->element_depth[block->p[cid->instance_id].depth].input_hitbox.set_position(block->p[cid->instance_id].be_id, position);
  }

  fan::opengl::matrices_t* get_matrices(loco_t* loco, fan::opengl::cid_t* cid) {
    auto block = sb_get_block(loco, cid);
    return fan::opengl::matrices_list_GetNodeByReference(&loco->get_context()->matrices_list, *block->p[cid->instance_id].key.get_value<0>())->data.matrices_id;
  }
  void set_matrices(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
    auto block = sb_get_block(loco, cid);
    *block->p[cid->instance_id].key.get_value<0>() = n;
    loco->text.set_matrices(loco, block->p[cid->instance_id].text_id, n);
  }

  fan::opengl::viewport_t* get_viewport(loco_t* loco, fan::opengl::cid_t* cid) {
    auto block = sb_get_block(loco, cid);
    return fan::opengl::viewport_list_GetNodeByReference(&loco->get_context()->viewport_list, *block->p[cid->instance_id].key.get_value<1>())->data.viewport_id;
  }
  void set_viewport(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    auto block = sb_get_block(loco, cid);
    *block->p[cid->instance_id].key.get_value<1>() = n;
    loco->text.set_viewport(loco, block->p[cid->instance_id].text_id, n);
  }

  protected:

  static void lib_set_theme(
    loco_t* loco,
    fan::opengl::cid_t* cid,
    f32_t intensity
  ) {
    loco->button.set_theme(loco, cid, &(*loco->button.get_theme(loco, cid)), intensity);
  }

  #define dont_look_here(d_n, i) lib_set_theme( \
    d_n.loco, \
    (fan::opengl::cid_t*)d_n.element_id, \
    i \
  );

  static uint8_t mouse_move_cb(const be_t::mouse_move_data_t& mm_data) {
    if (!mm_data.changed) {
      return 1;
    }
    switch (mm_data.mouse_stage) {
      case fan_2d::graphics::gui::mouse_stage_e::inside: {
        dont_look_here(mm_data, 1.1);
        break;
      }
      case fan_2d::graphics::gui::mouse_stage_e::outside: {
        dont_look_here(mm_data, 1.0);
        break;
      }
    }
    return 1;
  }
  static uint8_t mouse_input_cb(const be_t::mouse_input_data_t& ii_data) {
    if (ii_data.key != fan::mouse_left) {
      return 1;
    }
    switch (ii_data.mouse_stage) {
      case fan_2d::graphics::gui::mouse_stage_e::inside: {
        switch (ii_data.key_state) {
          case fan::key_state::press: {
            dont_look_here(ii_data, 1.2);
            break;
          }
          case fan::key_state::release: {
            dont_look_here(ii_data, 1.1);
            break;
          }
        }
        break;
      }
      case fan_2d::graphics::gui::mouse_stage_e::outside: {
        dont_look_here(ii_data, 1.0 / 1.2);
        break;
      }
    }
    return 1;
  }

  #undef dont_look_here
};