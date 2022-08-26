struct button_t {

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

  #define hardcore0_t fan::opengl::matrices_list_NodeReference_t
  #define hardcore0_n matrices
  #define hardcore1_t fan::opengl::viewport_list_NodeReference_t
  #define hardcore1_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode.h)

  struct instance_properties_t {
    struct key_t : parsed_masterpiece_t {}key;

    fan::opengl::theme_list_NodeReference_t theme;
    uint32_t text_id;
    loco_t::vfi_t::shape_id_t vfi_id;
    loco_t::vfi_t::mouse_button_cb_t mouse_button_cb;
    loco_t::vfi_t::mouse_move_cb_t mouse_move_cb;
    uint64_t userptr;
  };

  struct properties_t : instance_t {
    std::string text;
    f32_t font_size = 0.1;
    loco_t::vfi_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t&) -> void { return; };
    loco_t::vfi_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t&) -> void { return; };

    loco_t::vfi_t::iflags_t vfi_flags;

    uint64_t userptr;
    bool disable_highlight = false;

    union {
      struct {
        expand_all
        fan::opengl::theme_list_NodeReference_t theme;
        uint32_t text_id;
        uint32_t be_id;
      };
      instance_properties_t instance_properties;
    };
  };

  void push_back(fan::opengl::cid_t* cid, properties_t& p) {
    loco_t* loco = get_loco();
    auto theme = loco->button.get_theme(p.theme);
    loco_t::text_t::properties_t tp;
    tp.color = theme->button.text_color;
    tp.font_size = p.font_size;
    tp.position = p.position;
    tp.text = p.text;
    tp.position.z += p.position.z + 0.5;
    tp.viewport = p.viewport;
    tp.matrices = p.matrices;
    auto block = sb_push_back(cid, p);
    block->p[cid->instance_id].text_id = loco->text.push_back(tp);

    set_theme(cid, theme, 1);

    loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.matrices = p.matrices;
    vfip.shape.rectangle.viewport = p.viewport;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.udata = (uint64_t)cid;
    vfip.flags = p.vfi_flags;
    if (!p.disable_highlight) {
      vfip.mouse_move_cb = [](const loco_t::mouse_move_data_t& mm_d) -> void {
        loco_t* loco = OFFSETLESS(mm_d.vfi, loco_t, vfi_var_name);
        loco_t::mouse_move_data_t mmd = mm_d;
        fan::opengl::cid_t* cid = (fan::opengl::cid_t*)mm_d.udata;
        auto block = loco->button.sb_get_block(cid);
        if (mm_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
          loco->button.set_theme(cid, loco->button.get_theme(cid), hover);
        }
        else {
          loco->button.set_theme(cid, loco->button.get_theme(cid), inactive);
        }
        mmd.udata = block->p[cid->instance_id].userptr;
        block->p[cid->instance_id].mouse_move_cb(mmd);
      };
      vfip.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> void {
        loco_t* loco = OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name);
        fan::opengl::cid_t* cid = (fan::opengl::cid_t*)ii_d.udata;
        auto block = loco->button.sb_get_block(cid);
        if (ii_d.flag->ignore_move_focus_check == false) {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::key_state::press) {
            loco->button.set_theme(cid, loco->button.get_theme(cid), press);
            ii_d.flag->ignore_move_focus_check = true;
            loco->vfi.set_focus_keyboard(loco->vfi.get_focus_mouse());
          }
        }
        else {
          if (ii_d.button == fan::mouse_left && ii_d.button_state == fan::key_state::release) {
            if (ii_d.mouse_stage == loco_t::vfi_t::mouse_stage_e::inside) {
              loco->button.set_theme(cid, loco->button.get_theme(cid), hover);
            }
            else {
              loco->button.set_theme(cid, loco->button.get_theme(cid), inactive);
            }
            ii_d.flag->ignore_move_focus_check = false;
            loco->vfi.invalidate_focus_keyboard();
          }
        }

        loco_t::mouse_button_data_t mid = ii_d;
        mid.udata = block->p[cid->instance_id].userptr;
        block->p[cid->instance_id].mouse_button_cb(mid);
      };
    }

    block->p[cid->instance_id].vfi_id = loco->vfi.push_shape(vfip);
    block->p[cid->instance_id].mouse_button_cb = p.mouse_button_cb;
    block->p[cid->instance_id].mouse_move_cb = p.mouse_move_cb;
    block->p[cid->instance_id].userptr = p.userptr;
  }
  void erase(fan::opengl::cid_t* cid) {
    loco_t* loco = OFFSETLESS(this, loco_t, button);
    auto block = sb_get_block(cid);
    instance_properties_t* p = &block->p[cid->instance_id];
    loco->text.erase(p->text_id);
    loco->vfi.erase(block->p[cid->instance_id].vfi_id);
    sb_erase(cid);
  }

  void draw() {
    sb_draw();
  }

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle_box.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open() {
    sb_open();
  }
  void close() {
    sb_close();
  }

  fan_2d::graphics::gui::theme_t* get_theme(fan::opengl::theme_list_NodeReference_t nr) {
    loco_t* loco = OFFSETLESS(this, loco_t, button);
    return fan::opengl::theme_list_GetNodeByReference(&loco->get_context()->theme_list, nr)->data.theme_id;
  }
  fan_2d::graphics::gui::theme_t* get_theme(fan::opengl::cid_t* cid) {
    loco_t* loco = OFFSETLESS(this, loco_t, button);
    auto block_node = bll_block_GetNodeByReference(&blocks, *(bll_block_NodeReference_t*)&cid->block_id);
    return get_theme(block_node->data.block.p[cid->instance_id].theme);
  }
  void set_theme(fan::opengl::cid_t* cid, fan_2d::graphics::gui::theme_t* theme, f32_t intensity) {
    loco_t* loco = OFFSETLESS(this, loco_t, button);
    fan_2d::graphics::gui::theme_t t = *theme;
    t = t * intensity;
    set(cid, &instance_t::color, t.button.color);
    set(cid, &instance_t::outline_color, t.button.outline_color);
    set(cid, &instance_t::outline_size, t.button.outline_size);
    auto block = sb_get_block(cid);
    block->p[cid->instance_id].theme = theme;
    loco->text.set(block->p[cid->instance_id].text_id, 
      &loco_t::letter_t::instance_t::outline_color, t.button.text_outline_color);
    loco->text.set(block->p[cid->instance_id].text_id, 
      &loco_t::letter_t::instance_t::outline_size, t.button.text_outline_size);
  }

  template <typename T>
  T get_button(fan::opengl::cid_t* cid, T instance_t::* member) {
    loco_t* loco = OFFSETLESS(this, loco_t, button);
    return loco->button.get(cid, member);
  }
  template <typename T, typename T2>
  void set_button(fan::opengl::cid_t* cid, T instance_t::*member, const T2& value) {
    loco_t* loco = get_loco();
    loco->button.set(cid, member, value);
  }

  template <typename T>
  T get_text(fan::opengl::cid_t* cid, T loco_t::letter_t::instance_t::* member) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return loco->text.get(block->p[cid->instance_id].text_id, member);
  }
  template <typename T, typename T2>
  void set_text(fan::opengl::cid_t* cid, T loco_t::letter_t::instance_t::*member, const T2& value) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    loco->text.set(block->p[cid->instance_id].text_id, member, value);
  }

  void set_position(fan::opengl::cid_t* cid, const fan::vec3& position) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    set_text(cid, &loco_t::letter_t::instance_t::position, position + fan::vec3(0, 0, 0.5));
    set_button(cid, &instance_t::position, position);
    loco->vfi.set_rectangle(
      block->p[cid->instance_id].vfi_id,
      &loco_t::vfi_t::set_rectangle_t::position,
      position
    );
  }

  fan::opengl::matrices_t* get_matrices(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return fan::opengl::matrices_list_GetNodeByReference(&loco->get_context()->matrices_list, *block->p[cid->instance_id].key.get_value<0>())->data.matrices_id;
  }
  void set_matrices(fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    sb_set_key(cid, &properties_t::matrices, n);
    loco->text.set_matrices(block->p[cid->instance_id].text_id, n);
  }

  fan::opengl::viewport_t* get_viewport(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return fan::opengl::viewport_list_GetNodeByReference(&loco->get_context()->viewport_list, *block->p[cid->instance_id].key.get_value<1>())->data.viewport_id;
  }
  void set_viewport(fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    sb_set_key(cid, &properties_t::viewport, n);
    loco->text.set_viewport(block->p[cid->instance_id].text_id, n);
  }

  void set_theme(fan::opengl::cid_t* cid, f32_t state) {
    loco_t* loco = get_loco();
    loco->button.set_theme(cid, loco->button.get_theme(cid), state);
  }

  // gets udata from current focus
  uint64_t get_mouse_udata() {
    loco_t* loco = get_loco();
    auto udata = loco->vfi.get_mouse_udata();
    fan::opengl::cid_t* cid = (fan::opengl::cid_t*)udata;
    auto block = sb_get_block(cid);
    return block->p[cid->instance_id].userptr;
  }
};