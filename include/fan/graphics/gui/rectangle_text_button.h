struct button_t {

  using be_t = fan_2d::graphics::gui::be_t;

  static constexpr f32_t inactive = 1.0;
  static constexpr f32_t hover = 1.1;
  static constexpr f32_t press = 1.2;

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
    loco_t::vfi_t::shape_id_t vfi_id;
    loco_t::vfi_t::mouse_button_cb_t mouse_button_cb;
    loco_t::vfi_t::mouse_move_cb_t mouse_move_cb;
    uint64_t userptr;
  };

  struct properties_t : instance_t {
    fan::utf16_string text;
    f32_t font_size = 0.1;
    loco_t::vfi_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t&) -> void { return; };
    loco_t::vfi_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t&) -> void { return; };

    uint64_t userptr;
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

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    auto theme = loco->button.get_theme(loco, p.theme);
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 10;
    tp.viewport = p.viewport;
    tp.matrices = p.matrices;
    auto block = sb_push_back(loco, cid, p);
    block->p[cid->instance_id].text_id = loco->text.push_back(loco, tp);

    set_theme(loco, cid, theme, 1);

    loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.matrices = p.matrices;
    vfip.shape.rectangle.viewport = p.viewport;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.udata = (uint64_t)cid;
    vfip.mouse_move_cb = [] (const loco_t::mouse_move_data_t& mm_d) -> void {
      loco_t* loco = OFFSETLESS(mm_d.vfi, loco_t, vfi);
      loco_t::mouse_move_data_t mmd = mm_d;
      fan::opengl::cid_t* cid = (fan::opengl::cid_t*)mm_d.udata;
      auto block = loco->button.sb_get_block(loco, cid);
      if (mm_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
        loco->button.set_theme(loco, cid, loco->button.get_theme(loco, cid), hover);
      }
      else {
        loco->button.set_theme(loco, cid, loco->button.get_theme(loco, cid), inactive);
      }
      mmd.udata = block->p[cid->instance_id].userptr;
      block->p[cid->instance_id].mouse_move_cb(mmd);
    };
    vfip.mouse_button_cb = [](const loco_t::mouse_input_data_t& ii_d) -> void {
      loco_t* loco = OFFSETLESS(ii_d.vfi, loco_t, vfi);
      fan::opengl::cid_t* cid = (fan::opengl::cid_t*)ii_d.udata;
      auto block = loco->button.sb_get_block(loco, cid);
      if (ii_d.flag->ignore_move_focus_check == false) {
        if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::key_state::press) {
          loco->button.set_theme(loco, cid, loco->button.get_theme(loco, cid), press);
          ii_d.flag->ignore_move_focus_check = true;
        }
      }
      else {
        if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::key_state::release) {
          if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
            loco->button.set_theme(loco, cid, loco->button.get_theme(loco, cid), hover);
          }
          else {
            loco->button.set_theme(loco, cid, loco->button.get_theme(loco, cid), inactive);
          }
          ii_d.flag->ignore_move_focus_check = false;
        }
      }

      
      loco_t::mouse_input_data_t mid = ii_d;
      mid.udata = block->p[cid->instance_id].userptr;
      block->p[cid->instance_id].mouse_button_cb(mid);
    };

    block->p[cid->instance_id].vfi_id = loco->vfi.push_shape(vfip);
    block->p[cid->instance_id].mouse_button_cb = p.mouse_button_cb;
    block->p[cid->instance_id].mouse_move_cb = p.mouse_move_cb;
    block->p[cid->instance_id].userptr = p.userptr;
  }
  void erase(loco_t* loco, fan::opengl::cid_t* cid) {
    auto block = sb_get_block(loco, cid);
    instance_properties_t* p = &block->p[cid->instance_id];
    fan::print("removed text", p->text_id);
    loco->text.erase(loco, p->text_id);
    // delete vfi here
    assert(0);
    //block->
    //fan::print("deleted", loco->element_depth[p->depth].input_hitbox.m_button_data[p->be_id].properties.cid);
    //delete loco->element_depth[p->depth].input_hitbox.m_button_data[p->be_id].properties.cid;
    //loco->element_depth[p->depth].input_hitbox.erase(p->be_id);

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
    loco->vfi.set_rectangle(
      block->p[cid->instance_id].vfi_id,
      &loco_t::vfi_t::set_rectangle_t::position,
      position
    );
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

  static uint8_t mouse_move_cb(const be_t::mouse_move_data_t& mm_data) {
    if (!mm_data.changed) {
      return 1;
    }
    switch (mm_data.mouse_stage) {
      case fan_2d::graphics::gui::mouse_stage_e::inside: {
        lib_set_theme(mm_data.loco, (fan::opengl::cid_t*)mm_data.element_id, 1.1);
        break;
      }
      case fan_2d::graphics::gui::mouse_stage_e::outside: {
        lib_set_theme(mm_data.loco, (fan::opengl::cid_t*)mm_data.element_id, 1.0);
        break;
      }
    }
    return 1;
  }
  static uint8_t mouse_input_cb(const be_t::mouse_input_data_t& ii_data) {
    if (ii_data.key != fan::mouse_left) {
      return 1;
    }
    if (!ii_data.changed) {
      return 1;
    }
    switch (ii_data.mouse_stage) {
      case fan_2d::graphics::gui::mouse_stage_e::inside: {
        switch (ii_data.key_state) {
          case fan::key_state::press: {
            lib_set_theme(ii_data.loco, (fan::opengl::cid_t*)ii_data.element_id, 1.2);
            break;
          }
          case fan::key_state::release: {
            lib_set_theme(ii_data.loco, (fan::opengl::cid_t*)ii_data.element_id, 1.1);
            break;
          }
        }
        break;
      }
      case fan_2d::graphics::gui::mouse_stage_e::outside: {
        lib_set_theme(ii_data.loco, (fan::opengl::cid_t*)ii_data.element_id, 1.0 / 1.2);
        break;
      }
    }
    return 1;
  }
};