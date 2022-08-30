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

  static constexpr fan::vec2 button_size = fan::vec2(0.3, 0.08);

  f32_t line_y_offset_between_types_and_properties;

  loco_t* get_loco() {
    // ?
    return (loco_t*)((uint8_t*)OFFSETLESS(this, pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
  }

  // for -1 - 1 matrix
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
    fan::vec2 f = from->get_size();
    fan::vec2 t = to->get_size();
    return size / (t / f);
  }
  fan::vec2 scale_object_with_viewport(const fan::vec2& size, fan::opengl::viewport_t* from) {
    fan::vec2 f = from->get_size();
    fan::vec2 t = get_loco()->get_window()->get_size();
    return size / (f / t);
  }
  fan::vec2 translate_to_global(const fan::vec2& position) const {
    return position / viewport[viewport_area::global].get_size() * 2 - 1;
  }
  fan::vec2 get_viewport_dst(fan::opengl::viewport_t* from, fan::opengl::viewport_t* to) {
    return (from->get_size() + from->get_position()) / (to->get_size() / 2) - 1;
  }


  void invalidate_focus() {
    loco_t* loco = get_loco();
    loco->vfi.invalidate_focus_mouse();
    loco->vfi.invalidate_focus_keyboard();
  }
	
	corners_t get_corners(const fan::vec2& position, const fan::vec2& size) {
		loco_t* loco = get_loco();
		fan::vec2 c = position;
		fan::vec2 s = size;
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

  void open() {
    loco_t* loco = get_loco();

    line_y_offset_between_types_and_properties = 0.0;

    move_offset = 0;
    action_flag = 0;
    theme = fan_2d::graphics::gui::themes::deep_red();
    theme.open(loco->get_context());

    static auto resize_cb = [] (fan::window_t* window, void* userptr) {
      fan::vec2 window_size = window->get_size();
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
        fan::vec2(-1, 1),
        fan::vec2(-1, 1),
        1
      );

      fan::vec2 viewport_position = pile->fgm.translate_viewport_position(pile->fgm.editor_position - pile->fgm.editor_size);
      viewport_size = pile->fgm.translate_viewport_position(pile->fgm.editor_size + fan::vec2(-pile->fgm.properties_line_position.x / 2 - 0.1));
      pile->fgm.viewport[viewport_area::editor].set_viewport(
        pile->loco.get_context(),
        viewport_position,
        viewport_size,
        pile->loco.get_window()->get_size()
      );
      ratio = viewport_size / viewport_size.max();
      pile->fgm.matrices[viewport_area::editor].set_ortho(
        fan::vec2(-1, 1),
        fan::vec2(-1, 1),
        ratio
      );

      viewport_position = pile->fgm.translate_viewport_position(fan::vec2(pile->fgm.properties_line_position.x, -1));
      viewport_size = pile->fgm.translate_viewport_position(fan::vec2(1, pile->fgm.line_y_offset_between_types_and_properties)) - viewport_position;
      pile->fgm.viewport[viewport_area::types].set_viewport(
        pile->loco.get_context(),
        viewport_position,
        viewport_size,
        pile->loco.get_window()->get_size()
      );

      ratio = viewport_size / viewport_size.max();
      pile->fgm.matrices[viewport_area::types].set_ortho(
        fan::vec2(-1, 1),
        fan::vec2(-1, 1),
        ratio
      );

      viewport_position.y += pile->fgm.translate_viewport_position(fan::vec2(0, pile->fgm.line_y_offset_between_types_and_properties)).y;
      pile->fgm.viewport[viewport_area::properties].set_viewport(
        pile->loco.get_context(),
        viewport_position,
        viewport_size,
        pile->loco.get_window()->get_size()
      );

      ratio = viewport_size / viewport_size.max();
      pile->fgm.matrices[viewport_area::properties].set_ortho(
        fan::vec2(-1, 1),
        fan::vec2(-1, 1),
        ratio
      );

      fan::vec3 src, dst;

      src = pile->fgm.editor_position - pile->fgm.editor_size;
      dst.x = pile->fgm.editor_position.x + pile->fgm.editor_size.x;
      dst.y = src.y;

      src.z = dst.z = 10;
      pile->loco.line.set_line(
        &pile->fgm.line.instance[0]->cid,
        src,
        dst
      );

      src = dst;
      dst.y = pile->fgm.editor_position.y + pile->fgm.editor_size.y;

      src.z = dst.z = 10;
      pile->loco.line.set_line(
        &pile->fgm.line.instance[1]->cid,
        src,
        dst
      );

      src = dst;
      dst.x = pile->fgm.editor_position.x - pile->fgm.editor_size.x;

      src.z = dst.z = 10;
      pile->loco.line.set_line(
        &pile->fgm.line.instance[2]->cid,
        src,
        dst
      );

      src = dst;
      dst.y = pile->fgm.editor_position.y - pile->fgm.editor_size.y;

      src.z = dst.z = 10;
      pile->loco.line.set_line(
        &pile->fgm.line.instance[3]->cid,
        src,
        dst
      );

      src = pile->fgm.translate_to_global(
        pile->fgm.viewport[viewport_area::types].get_position()
      );
      dst.x = src.x;
      dst.y = pile->fgm.matrices[viewport_area::global].coordinates.bottom;
      src.z = dst.z = 10;
      pile->loco.line.set_line(
        &pile->fgm.line.instance[4]->cid,
        src,
        dst
      );
      src = pile->fgm.translate_to_global(
        pile->fgm.viewport[viewport_area::types].get_position() +
        fan::vec2(0, pile->fgm.viewport[viewport_area::types].get_size().y)
      );
      dst = pile->fgm.translate_to_global(
        pile->fgm.viewport[viewport_area::types].get_position() +
        pile->fgm.viewport[viewport_area::types].get_size()
      );
      src.z = dst.z = 10;
      pile->loco.line.set_line(
        &pile->fgm.line.instance[5]->cid,
        src,
        dst
      );
    };

    loco->get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& ws, void* userptr) {
      resize_cb(window, userptr);
    });

    // half size
    properties_line_position = fan::vec2(0.5, 0);
    editor_position = fan::vec2(-properties_line_position.x / 2, 0);
    editor_size = editor_position.x + 0.9;

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
      pile->fgm.invalidate_focus();
    };
    loco->vfi.push_shape(p);

    line.open();
    editor_button.open();
    builder_button.open();
    active.open();
    menu.open(fan::vec2(1.05, button_size.y * 1.5));
		viewport_resize.open();

    line_t::properties_t lp;
    lp.viewport = &viewport[viewport_area::global];
    lp.matrices = &matrices[viewport_area::global];
    lp.color = fan::colors::white;
    
    // editor window
    line.push_back(lp);
    line.push_back(lp);
    line.push_back(lp);
    line.push_back(lp);

    // properties
    line.push_back(lp);
    line.push_back(lp);

    resize_cb(loco->get_window(), this);
    
    editor_button_t::properties_t ebp;
    ebp.matrices = &matrices[viewport_area::types];
    ebp.viewport = &viewport[viewport_area::types];
    ebp.position = fan::vec2(0, matrices[viewport_area::types].coordinates.top * 0.9);
    ebp.size = button_size;
    ebp.theme = &theme;
    ebp.text = "button";
    editor_button.push_back(ebp);

    properties_menu_t::properties_t menup;
    menup.text = "ratio";
    menup.text_value = "1, 1";
    menu.push_back(menup);
	
		viewport_resize.move_corners(get_corners(editor_position, editor_size));
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
        if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
          return;
        }
        pile->fgm.action_flag |= action::move;
        builder_button_t::properties_t bbp;
        bbp.matrices = &pile->fgm.matrices[viewport_area::editor];
        bbp.viewport = &pile->fgm.viewport[viewport_area::editor];
        bbp.position = pile->loco.get_mouse_position(
          pile->fgm.viewport[viewport_area::editor].get_position(),
          pile->fgm.viewport[viewport_area::editor].get_size()
        );

        bbp.size = button_size;
        //bbp.size = button_size;
        bbp.theme = &pile->fgm.theme;
        bbp.text = "button";
        bbp.font_size = scale_object_with_viewport(fan::vec2(0.2), &pile->fgm.viewport[viewport_area::types], &pile->fgm.viewport[viewport_area::editor]).x;
        //bbp.font_size = 0.2;
        pile->fgm.builder_button.push_back(bbp);
        pile->loco.button.set_theme(&instance->cid, loco_t::button_t::inactive);
        auto builder_cid = &pile->fgm.builder_button.instance[pile->fgm.builder_button.instance.size() - 1]->cid;
        auto block = pile->loco.button.sb_get_block(builder_cid);
        pile->loco.vfi.set_focus_mouse(block->p[builder_cid->instance_id].vfi_id);
        pile->loco.vfi.feed_mouse_button(fan::mouse_left, fan::key_state::press);
        return;
      };
      loco->button.push_back(&instance[i]->cid, p);
    }

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
          if (!pile->fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position())) {
            pile->fgm.builder_button.erase(&instance->cid);
            pile->fgm.active.clear();
          }
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
      p.keyboard_cb = [](const loco_t::keyboard_data_t& kd) -> void {
        pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

        switch (kd.key) {
        case fan::key_delete: {
          switch (kd.key_state) {
          case fan::key_state::press: {
            auto udata = pile->loco.button.get_keyboard_udata();
            instance_t* instance = (instance_t*)udata;
            switch (instance->shape) {
            case shapes::button: {
              pile->fgm.builder_button.erase(&instance->cid);
              pile->fgm.active.clear();
              pile->fgm.invalidate_focus();
              break;
            }
            }
            break;
          }
          }
          break;
        }
        }
      };
      p.userptr = (uint64_t)instance[i];
      loco->button.push_back(&instance[i]->cid, p);
      pile->loco.button.set_theme(&instance[i]->cid, loco_t::button_t::inactive);
      auto builder_cid = &instance[i]->cid;
      auto block = pile->loco.button.sb_get_block(builder_cid);
      pile->loco.vfi.set_focus_mouse(block->p[builder_cid->instance_id].vfi_id);
      pile->fgm.active.move_corners(pile->fgm.builder_button.get_corners(&instance[i]->cid));

    }
    void erase(fan::opengl::cid_t* cid) {
      loco_t* loco = get_loco();
      loco->button.erase(cid);
      release();
    }
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
      pile_t* pile = get_pile();
      active_rectangles_t::properties_t p;
      p.size = r_size;
      auto data = pile->loco.button.get_mouse_udata();
      instance_t* instance = (instance_t*)data;
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
      for (uint32_t i = 0; i < corners.count; i++) {
        pile->loco.button.set_position(&instance[i]->cid, corners.corners[i]);
      }
    }

    void move_corners(const pile_t::fgm_t::corners_t& corners) {
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
    fan::hector_t<instance_t*> instance;
  }active;

	struct viewport_resize_t {
    using properties_t = loco_t::button_t::properties_t;

    static constexpr f32_t r_size = 0.015;

    loco_t* get_loco() {
      return (loco_t*)((uint8_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, viewport_resize), pile_t, fgm_var_name) + offsetof(pile_t, loco_var_name));
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
      pile_t* pile = get_pile();
      viewport_resize_t::properties_t p;
      p.size = r_size;
		  p.viewport = &pile->fgm.viewport[viewport_area::global];
			p.matrices = &pile->fgm.matrices[viewport_area::global];
			static fan_2d::graphics::gui::theme_t theme = fan_2d::graphics::gui::themes::gray();
			theme.open(pile->loco.get_context());
			p.theme = &theme;
      for (uint32_t i = 0; i < corners.count; i++) {
        p.position = corners.corners[i];
        push_back(p);
      }
    }
    void set_corners(const corners_t& corners) {
      pile_t* pile = get_pile();
      for (uint32_t i = 0; i < corners.count; i++) {
        pile->loco.button.set_position(&instance[i]->cid, corners.corners[i]);
      }
    }

    void move_corners(const pile_t::fgm_t::corners_t& corners) {
      pile_t* pile = get_pile();
      if (!size()) {
        pile->fgm.viewport_resize.push_corners(corners);
      }
      else {
        pile->fgm.viewport_resize.set_corners(corners);
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
    fan::hector_t<instance_t*> instance;
  }viewport_resize;

  struct properties_menu_t {

    struct properties_t {
      std::string text_value;
      std::string text;
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
      tp.position = p.position - fan::vec3(text_size.x / 1.5 + p.size.x, 0, 0);
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

  fan::vec2 properties_line_position;

  fan::vec2 editor_position;
  fan::vec2 editor_size;
};