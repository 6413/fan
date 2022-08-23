struct fgm_t {

  struct instance_t {
    f32_t z = 0;
    fan::opengl::cid_t cid;
    uint16_t shape;
  };

  struct shapes {
    static constexpr uint16_t line = 0;
    static constexpr uint16_t button = 1;
  };

  struct viewport_area{
    static constexpr uint32_t global = 0;
    static constexpr uint32_t editor = 1;
    static constexpr uint32_t types = 2;
    static constexpr uint32_t properties = 3;
  };

  struct action {
    static constexpr uint32_t move = 1 << 0;
    static constexpr uint32_t resize = 1 << 1;
  };

  struct corners_t {
    static constexpr uint32_t count = 8;
    fan::vec2 corners[count];
  };

  static constexpr fan::_vec2<fan::vec2> coordinate_system = { 
    {-1, 1}, 
    {-1, 1} 
  };
  static constexpr fan::vec2 button_size = fan::vec2(0.3, 0.08);

  f32_t line_y_offset_between_types_and_properties;

  loco_t* get_loco() {
    // ?
    return (loco_t*)((uint8_t*)OFFSETLESS(this, pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
  }

  fan::vec2 editor_size;

  fan::vec2 translate_viewport_position(const fan::vec2& value) {
    loco_t* loco = get_loco();
    fan::vec2 window_size = loco->get_window()->get_size();
    return (value + 1) / 2 * window_size;
  }
  fan::vec2 position_to_coordinates(const fan::vec2& value) {

    loco_t* loco = get_loco();
    fan::vec2 window_size = loco->get_window()->get_size();
    
    fan::vec2 ret = value / window_size;
    ret -= 2.0 / 2;
    return fan::vec2(0.5, ret.y);
  }
  static fan::vec2 scale_object_with_viewport(const fan::vec2& size, fan::opengl::viewport_t* from, fan::opengl::viewport_t* to) {
    fan::vec2 f = from->get_viewport_size();
    fan::vec2 t = to->get_viewport_size();
    return size / (t / f);
  }

  void open() {
    loco_t* loco = get_loco();

    line_y_offset_between_types_and_properties = 0.2;

    move_offset = 0;
    action_flag = 0;
    theme = fan_2d::graphics::gui::themes::deep_red();
    theme.open(loco->get_context());

    loco->get_window()->add_keys_callback(loco, [](fan::window_t*, uint16_t key, fan::key_state key_state, void* userptr) {
      //pile_t* pile = OFFSETLESS(userptr, pile_t, loco);
      //if (!pile->fgm.is_selected(pile)) {
      //  return;
      //}
      //switch(key) {
      //  case fan::key_delete: {
      //    switch(key_state) {
      //      case fan::key_state::press: {
      //        auto focused = pile->loco.button.get_mouse_move_focus_data();
      //        instance_t* instance = (instance_t*)focused.udata;
      //        switch (instance->shape) {
      //          case shapes::button: {
      //            pile->fgm.builder_button.erase(&instance->cid);
      //            pile->fgm.active.clear();
      //            pile->loco.vfi.invalidate_focus();
      //            break;
      //          }
      //        }
      //        break;
      //      }
      //    }
      //    break;
      //  }
      //  case fan::key_up: {
      //    ////auto focused = pile->loco.vfi.get_mouse_move_focus_data();
      //    //instance_t* instance = (instance_t*)pile->fgm.selected;
      //    //switch (instance->shape) {
      //    //case shapes::button: {
      //    //  instance->z++;
      //    //  break;
      //    //}
      //    //}
      //    //break;
      //  }
      //}
    });

    static auto resize_cb = [] (fan::window_t* window, void* userptr) {
      fan::vec2 window_size = window->get_size();
      //std::swap(ratio.x, ratio.y);
      fgm_t* fgm = (fgm_t*)userptr;
      pile_t* pile = OFFSETLESS(fgm, pile_t, fgm);
      fan::vec2 viewport_size = pile->fgm.translate_viewport_position(fan::vec2(1, 1));
      pile->fgm.viewport[viewport_area::global].set_viewport(
        pile->loco.get_context(),
        0,
        viewport_size,
        window_size
      );
      fan::vec2 ratio = viewport_size / viewport_size.max();
      pile->fgm.matrices[viewport_area::global].set_ortho(
        coordinate_system.x,
        coordinate_system.y,
        ratio
      );

      viewport_size = pile->fgm.translate_viewport_position(pile->fgm.editor_size);
      pile->fgm.viewport[viewport_area::editor].set_viewport(
        pile->loco.get_context(),
        0,
        viewport_size,
        pile->loco.get_window()->get_size()
      );
      ratio = viewport_size / viewport_size.max();
      pile->fgm.matrices[viewport_area::editor].set_ortho(
        coordinate_system.x,
        coordinate_system.y,
        ratio
      );

      fan::vec2 top_viewport = pile->fgm.translate_viewport_position(fan::vec2(pile->fgm.editor_size.x, -1));
      viewport_size = pile->fgm.translate_viewport_position(fan::vec2(1, pile->fgm.line_y_offset_between_types_and_properties)) - top_viewport;
      pile->fgm.viewport[viewport_area::types].set_viewport(
        pile->loco.get_context(),
        top_viewport,
        viewport_size,
        pile->loco.get_window()->get_size()
      );

      ratio = viewport_size / viewport_size.max();
      pile->fgm.matrices[viewport_area::types].set_ortho(
        coordinate_system.x,
        coordinate_system.y,
        ratio
      );

      top_viewport.y += pile->fgm.translate_viewport_position(fan::vec2(0, pile->fgm.line_y_offset_between_types_and_properties)).y;
      pile->fgm.viewport[viewport_area::properties].set_viewport(
        pile->loco.get_context(),
        top_viewport,
        viewport_size,
        pile->loco.get_window()->get_size()
      );

      ratio = viewport_size / viewport_size.max();
      pile->fgm.matrices[viewport_area::properties].set_ortho(
        coordinate_system.x,
        coordinate_system.y,
        ratio
      );

      fan::vec2 src = fan::vec2(coordinate_system[0][0], coordinate_system[1][0]);
      fan::print(pile->fgm.viewport[viewport_area::types].viewport_position.x, pile->fgm.position_to_coordinates(pile->fgm.viewport[viewport_area::types].viewport_position).x);
      fan::vec2 dst = fan::vec2(pile->fgm.position_to_coordinates(pile->fgm.viewport[viewport_area::types].viewport_position).x, coordinate_system[1][0]);

      pile->loco.line.set_line(
        &pile->fgm.line.instance[0]->cid, 
        src,
        dst
      );
      src = dst;
      dst.y += coordinate_system[0][1] - coordinate_system[0][0];
      pile->loco.line.set_line(
        &pile->fgm.line.instance[1]->cid,
        src,
        dst
      );
      src = fan::vec2(coordinate_system[0][0], coordinate_system[0][1]);
      src = fan::vec2(coordinate_system[1][1], coordinate_system[1][1]);
      pile->loco.line.set_line(
        &pile->fgm.line.instance[2]->cid,
        src,
        dst
      );
    };

    loco->get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& ws, void* userptr) {
      resize_cb(window, userptr);
    });

    // half size
    editor_size = fan::vec2(0.5, 1);

    matrices[viewport_area::global].open(loco->get_context());
    matrices[viewport_area::editor].open(loco->get_context());
    matrices[viewport_area::types].open(loco->get_context());
    matrices[viewport_area::properties].open(loco->get_context());

    viewport[viewport_area::global].open(loco->get_context());
    viewport[viewport_area::editor].open(loco->get_context());
    viewport[viewport_area::types].open(loco->get_context());
    viewport[viewport_area::properties].open(loco->get_context());

    loco_t::vfi_t::properties_t p;
    p.shape_type = loco_t::vfi_t::shape_t::always;
    p.shape.always.z = 0;
    p.mouse_button_cb = [](const loco_t::mouse_button_data_t& mb) {
      pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);
      pile->fgm.active.clear();
      pile->loco.vfi.invalidate_focus();
    };
    loco->vfi.push_shape(p);

    line.open();
    editor_button.open();
    builder_button.open();
    active.open();
    menu.open(fan::vec2(1, button_size.y * 1.5));

    line_t::properties_t lp;
    lp.viewport = &viewport[viewport_area::global];
    lp.matrices = &matrices[viewport_area::global];
    lp.color = fan::colors::white;
    // update these in resize
    line.push_back(lp);
    line.push_back(lp);
    lp.viewport = &viewport[viewport_area::global];
    line.push_back(lp);

    resize_cb(loco->get_window(), this);
    
    editor_button_t::properties_t ebp;
    ebp.matrices = &matrices[viewport_area::types];
    ebp.viewport = &viewport[viewport_area::types];
    ebp.position = fan::vec2(0, -0.8);
    ebp.size = button_size;
    ebp.theme = &theme;
    ebp.text = "button";
    editor_button.push_back(ebp);

    properties_menu_t::properties_t menup;
    menup.text = "position";
    menup.text_value = "100, 100";
    menu.push_back(menup);
  }
  void close() {
    line.close();
    editor_button.close();
    builder_button.close();
  }

  struct line_t {
    using properties_t = loco_t::line_t::properties_t;

    loco_t* get_loco() {
      return (loco_t*)((uint8_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, line), pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
    }
    pile_t* get_pile() {
      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
    }

    void open() {
      instance.open();
    }
    void close() {
      instance.close();
    }
    void push_back(properties_t& p) {
      loco_t* loco = get_loco();
      uint32_t i = instance.resize(instance.size() + 1);
      instance[i] = new instance_t;
      instance[i]->shape = shapes::line;
      loco->line.push_back(&instance[i]->cid, p);
    }
    fan::hector_t<instance_t*> instance;
  }line;

  struct editor_button_t {
    using properties_t = loco_t::button_t::properties_t;

    loco_t* get_loco() {
      return (loco_t*)((uint8_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, editor_button), pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
    }
    pile_t* get_pile() {
      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
    }

    void open() {
      instance.open();
    }
    void close() {
      instance.close();
    }
    void push_back(properties_t& p) {
      p.position.z = 1;
      loco_t* loco = get_loco();
      uint32_t i = instance.resize(instance.size() + 1);
      instance[i] = new instance_t;
      instance[i]->shape = shapes::button;
      p.userptr = (uint64_t)instance[i];
      p.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> void {
        pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi), pile_t, loco);
        instance_t* instance = (instance_t*)ii_d.udata;
        if (ii_d.button != fan::mouse_left) {
          return;
        }
        if (ii_d.button_state != fan::key_state::press) {
          pile->fgm.builder_button.release();
          if (!pile->fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position())) {
            //switch (pile->loco.focus.shape_type) {
            //  case loco_t::shapes::button: {
            //    pile->fgm.builder_button.erase(&pile->loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id);
            //  /*  pile->loco.button.set_viewport(
            //      &pile->loco,
            //      (fan::opengl::cid_t*)pile->fgm.selected_index,
            //      &pile->fgm.viewport[viewport_area::editor]
            //    );*/
            //    break;
            //  }
            //}
          }
          // pile->loco.focus.shape_type = fan::uninitialized;
          return;
        }
        if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
          return;
        }
        pile->fgm.action_flag |= action::move;
        builder_button_t::properties_t bbp;
        bbp.matrices = &pile->fgm.matrices[viewport_area::editor];
        bbp.viewport = &pile->fgm.viewport[viewport_area::editor];
        bbp.position = pile->loco.get_mouse_position(
          pile->fgm.viewport[viewport_area::editor].get_viewport_position(),
          pile->fgm.viewport[viewport_area::editor].get_viewport_size()
        );

        bbp.size = scale_object_with_viewport(button_size, &pile->fgm.viewport[viewport_area::types], &pile->fgm.viewport[viewport_area::editor]);
        //bbp.size = button_size;
        bbp.theme = &pile->fgm.theme;
        bbp.text = "button";
        bbp.font_size = scale_object_with_viewport(fan::vec2(0.2), &pile->fgm.viewport[viewport_area::types], &pile->fgm.viewport[viewport_area::editor]).x;
        //bbp.font_size = 0.2;
        pile->fgm.builder_button.push_back(bbp);
        pile->loco.button.set_theme(&instance->cid, loco_t::button_t::inactive);
        auto builder_cid = &pile->fgm.builder_button.instance[pile->fgm.builder_button.instance.size() - 1]->cid;
        auto block = pile->loco.button.sb_get_block(builder_cid);
        pile->loco.vfi.set_mouse_focus(block->p[builder_cid->instance_id].vfi_id);
        return;
      };
      loco->button.push_back(&instance[i]->cid, p);
    }
   /* void sanitize_cid(fan::opengl::cid_t* cid) {
      bool found = false;
      for (auto i : cids) {
        if (cid == i) {
          found = true;
        }
      }
      if (!found) {
        fan::throw_error("invalid cid");
      }
    }*/

    fan::hector_t<instance_t*> instance;
  }editor_button;

  struct builder_button_t {
    using properties_t = loco_t::button_t::properties_t;

    loco_t* get_loco() {
      return (loco_t*)((uint8_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, builder_button), pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
    }
    pile_t* get_pile() {
      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
    }

    void open() {
      instance.open();
    }
    void close() {
      instance.close();
    }
    corners_t get_corners(fan::opengl::cid_t* cid) {
      loco_t* loco = get_loco();
      fan::vec2 c = loco->button.get_button(cid, &loco_t::button_t::instance_t::position);
      fan::vec2 s = loco->button.get_button(cid, &loco_t::button_t::instance_t::size);
      corners_t corners;
      corners.corners[0] = c - s;
      corners.corners[1] = fan::vec2(c.x, c.y - s.y);
      corners.corners[2] = fan::vec2(c.x + s.x, c.y - s.y);
      corners.corners[3] = fan::vec2(c.x - s.x, c.y);
      corners.corners[4] = fan::vec2(c.x + s.x, c.y);
      corners.corners[5] = fan::vec2(c.x - s.x, c.y + s.y);
      corners.corners[6] = fan::vec2(c.x, c.y + s.y);
      corners.corners[7] = fan::vec2(c.x + s.x, c.y + s.y);
      return corners;
    }
    void release() {
      pile_t* pile = get_pile();
      pile->fgm.move_offset = 0;
      pile->fgm.action_flag &= ~action::move;
    }
    void push_back(properties_t& p) {
      p.position.z = 1;
      loco_t* loco = get_loco();
      pile_t* pile = get_pile();
      uint32_t i = instance.resize(instance.size() + 1);
      instance[i] = new instance_t;
      instance[i]->shape = shapes::button;
      instance[i]->z = 0;
      p.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> void {
        pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
        instance_t* instance = (instance_t*)ii_d.udata;
        if (ii_d.button != fan::mouse_left) {
          return;
        }
        if (ii_d.button_state == fan::key_state::release) {
          pile->fgm.builder_button.release();
          return;
        }
        if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
          return;
        }
        pile->fgm.action_flag |= action::move;
        auto viewport = pile->loco.button.get_viewport(&instance->cid);
        pile->fgm.move_offset =  fan::vec2(pile->loco.button.get_button(&instance->cid, &loco_t::button_t::instance_t::position)) - ii_d.position;
        pile->fgm.active.move_corners(pile->fgm.builder_button.get_corners(&instance->cid));
        return;
      };
      p.mouse_move_cb = [](const loco_t::mouse_move_data_t& ii_d) -> void {
      
        if (ii_d.flag->ignore_move_focus_check == false) {
          return;
        }
        pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
        instance_t* instance = (instance_t*)ii_d.udata;
        fan::vec3 p;
        p.x = ii_d.position.x + pile->fgm.move_offset.x;
        p.y = ii_d.position.y + pile->fgm.move_offset.y;
        p.z = instance->z;
        pile->loco.button.set_position(&instance->cid, p);
        pile->fgm.active.move_corners(pile->fgm.builder_button.get_corners(&instance->cid));
      };
      p.userptr = (uint64_t)instance[i];
      loco->button.push_back(&instance[i]->cid, p);
      pile->loco.button.set_theme(&instance[i]->cid, loco_t::button_t::inactive);
      auto builder_cid = &instance[i]->cid;
      auto block = pile->loco.button.sb_get_block(builder_cid);
      pile->loco.vfi.set_mouse_focus(block->p[builder_cid->instance_id].vfi_id);
      pile->fgm.active.move_corners(pile->fgm.builder_button.get_corners(&instance[i]->cid));

    }
    void erase(fan::opengl::cid_t* cid) {
      loco_t* loco = get_loco();
      loco->button.erase(cid);
      release();
    }

    //bool sanitize_cid(fan::opengl::cid_t* cid) {
    //  bool found = false;
    //  for (auto i : cids) {
    //    if (cid == i) {
    //      found = true;
    //    }
    //  }
    //  return found;
    //}
    fan::hector_t<instance_t*> instance;
  }builder_button;

  struct active_rectangles_t {
    using properties_t = loco_t::button_t::properties_t;

    static constexpr f32_t r_size = 0.015;

    loco_t* get_loco() {
      return (loco_t*)((uint8_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, active), pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
    }
    pile_t* get_pile() {
      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
    }

    void open() {
      instance.open();
    }
    void close() {
      instance.close();
    }
    void push_back(properties_t& p) {
      p.position.z = 1;
      loco_t* loco = get_loco();
      uint32_t i = instance.resize(instance.size() + 1);
      instance[i] = new instance_t;
      instance[i]->shape = shapes::button;
      p.vfi_flags.ignore_button = true;
      loco->button.push_back(&instance[i]->cid, p);
    }
    void push_corners(const corners_t& corners) {
      if (!instance.empty()) {
        fan::throw_error("a");
      }
      pile_t* pile = get_pile();
      active_rectangles_t::properties_t p;
      p.size = r_size;
      auto data = pile->loco.button.get_mouse_udata();
      instance_t* instance = (instance_t*)data.udata;
      switch(instance->shape) {
        case shapes::button: {
          p.viewport = pile->loco.button.get_viewport(&instance->cid);
          p.matrices = pile->loco.button.get_matrices(&instance->cid);
          p.theme = pile->loco.button.get_theme(&instance->cid);
          break;
        }
      }
      for (uint32_t i = 0; i < corners.count; i++) {
        p.position = corners.corners[i];
        push_back(p);
      }
    }
    void set_corners(const corners_t& corners) {
      pile_t* pile = get_pile();
      if (instance.empty()) {
        fan::throw_error("a");
      }
      for (uint32_t i = 0; i < corners.count; i++) {
        pile->loco.button.set_position(&instance[i]->cid, corners.corners[i]);
      }
    }

    void move_corners(const corners_t& corners) {
      pile_t* pile = get_pile();
      if (!size()) {
        pile->fgm.active.push_corners(corners);
      }
      else {
        pile->fgm.active.set_corners(corners);
      }
    }

    void clear() {
      pile_t* pile = get_pile();
      for (uint32_t i = 0; i < instance.size(); i++) {
        pile->loco.button.erase(&instance[i]->cid);
        delete instance[i];
      }
      instance.clear();
    }

    uint32_t size() const {
      return instance.size();
    }
    //void sanitize_cid(fan::opengl::cid_t* cid) {
    //  bool found = false;
    //  for (auto i : cids) {
    //    if (cid == i) {
    //      found = true;
    //    }
    //  }
    //  if (!found) {
    //    fan::throw_error("invalid cid");
    //  }
    //}

    fan::hector_t<instance_t*> instance;
  }active;

  struct properties_menu_t {

    struct properties_t {
      fan::utf16_string text_value;
      fan::utf16_string text;
    };

    loco_t* get_loco() {
      return (loco_t*)((uint8_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, menu), pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
    }
    pile_t* get_pile() {
      return OFFSETLESS(get_loco(), pile_t, loco_var_name);
    }

    void open(const fan::vec2& off) {
      instance.open();
      offset = off;
    }
    void close() {
      instance.close();
    }

    void push_back(properties_t mp) {
      loco_t::button_t::properties_t p;
      pile_t* pile = get_pile();
      p.size = pile->fgm.button_size;
      offset += p.size.y;
      p.position = fan::vec2(-1, -1) + offset;
      p.position.z = 1;
      p.theme = &pile->fgm.theme;
      p.matrices = &pile->fgm.matrices[viewport_area::properties];
      p.viewport = &pile->fgm.viewport[viewport_area::properties];
      p.text = mp.text_value;
     // p.disable_highlight = true;
      uint32_t i = instance.resize(instance.size() + 1);
      instance[i] = new i_t;
      instance[i]->shape = shapes::button;
      pile->loco.button.push_back(&instance[i]->cid, p);
      loco_t::text_t::properties_t tp;
      tp.text = mp.text;
      fan::vec2 text_size = pile->loco.text.get_text_size(tp.text, p.font_size);
      tp.position = p.position - fan::vec2(text_size.x / 1.5 + p.size.x, 0);
      tp.font_size = p.font_size;
      tp.matrices = p.matrices;
      tp.viewport = p.viewport;
      instance[i]->text_id = pile->loco.text.push_back(tp);
    }

    struct i_t : instance_t {
      uint32_t text_id;
    };

    fan::vec2 offset;
    fan::hector_t<i_t*> instance;
  }menu;

  fan::opengl::matrices_t matrices[4];
  fan::opengl::viewport_t viewport[4];

  fan_2d::graphics::gui::theme_t theme;

  uint32_t action_flag;

  fan::vec2 move_offset;
};