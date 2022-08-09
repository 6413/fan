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

  static constexpr uint32_t max_instance_size = std::min(256ull, 4096 / (sizeof(instance_t) / 4));

  struct instance_properties_t {
    struct key_t : fan::masterpiece_t<
      fan::opengl::matrices_list_NodeReference_t,
      fan::opengl::viewport_list_NodeReference_t
    > {}key;

    fan::opengl::theme_list_NodeReference_t theme;
    uint32_t text_id;
    uint32_t be_id;
  };

  struct properties_t : instance_t {
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
      };
      instance_properties_t instance_properties;
    };
  };

  void push_back(loco_t* loco, uint32_t depth, fan::opengl::cid_t* cid, properties_t& p) {
    auto theme = loco->button.get_theme(loco, p.theme);
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 0.001;
    auto block = sb_push_back(loco, cid, p);
    block->p[cid->instance_id].text_id = loco->text.push_back(loco, tp);

    set_theme(loco, cid, theme, 1);

    fan_2d::graphics::gui::be_t::properties_t be_p;
    be_p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
    be_p.hitbox_rectangle.position = p.position;
    be_p.hitbox_rectangle.size = p.size;
    be_p.on_input_function = p.mouse_input_cb;
    be_p.on_mouse_event_function = p.mouse_move_cb;
    be_p.userptr[0] = loco;
    be_p.userptr[2] = p.userptr;
    be_p.cid = cid;
    if (p.disable_highlight) {
      block->p[cid->instance_id].be_id = loco->element_depth[depth].input_hitbox.push_back(
        be_p, 
        [](const be_t::mouse_input_data_t&)->uint8_t { return 1; /* continue*/ },
        [](const be_t::mouse_move_data_t&)->uint8_t { return 1; /* continue*/ }
      );
    }
    else {
      block->p[cid->instance_id].be_id = loco->element_depth[depth].input_hitbox.push_back(be_p, mouse_input_cb, mouse_move_cb);
    }
  }
  void erase(loco_t* loco, uint32_t depth, fan::opengl::cid_t* cid) {
    auto block = sb_get_block(loco, cid);
    loco->text.erase(loco, block->p[cid->instance_id].text_id);
    loco->element_depth[depth].input_hitbox.erase(block->p[cid->instance_id].be_id);

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
  T get_text(loco_t* loco, uint32_t id, T loco_t::letter_t::instance_t::* member) {
    return loco->text.get(loco, id, member);
  }
  template <typename T, typename T2>
  void set_text(loco_t* loco, uint32_t id, T loco_t::letter_t::instance_t::*member, const T2& value) {
    loco->text.set(loco, id, member, value);
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
  (loco_t*)d_n.userptr[0], \
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