struct fgm_t {
  struct viewport_area{
    static constexpr uint32_t global = 0;
    static constexpr uint32_t editor = 1;
    static constexpr uint32_t types = 2;
  };

  struct action {
    static constexpr uint32_t move = 1 << 0;
    static constexpr uint32_t resize = 1 << 1;
  };

  struct corners_t {
    static constexpr uint32_t count = 9;
    fan::vec2 corners[count];
  };

  static constexpr fan::vec2 button_size = fan::vec2(0.5, 0.1);

  fan::vec2 editor_size;

  fan::vec2 translate_viewport_position(loco_t* loco, const fan::vec2& value) {
    fan::vec2 window_size = loco->get_window()->get_size();
    return (value + 1) / 2 * window_size;
  }
  static fan::vec2 scale_object_with_viewport(const fan::vec2& size, fan::opengl::viewport_t* from, fan::opengl::viewport_t* to) {
    fan::vec2 f = from->get_viewport_size();
    fan::vec2 t = to->get_viewport_size();
    return size / (t / f);
  }

  void open(loco_t* loco) {
    move_offset = 0;
    action_flag = 0;
    theme = fan_2d::graphics::gui::themes::gray();
    theme.open(loco->get_context());

   /* loco->get_window()->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& ws, void* userptr) {
      fan::vec2 window_size = window->get_size();
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      fgm_t* fgm = (fgm_t*)userptr;
      fgm->matrices.set_ortho(
        fan::vec2(-1, 1) * ratio.x, 
        fan::vec2(-1, 1) * ratio.y
      );
      pile_t* pile = OFFSETLESS(fgm, pile_t, fgm);
      fgm->viewport[0].set_viewport(pile->loco.get_context(), 0, ws);
    });*/

    loco->get_window()->add_keys_callback(loco, [](fan::window_t*, uint16_t key, fan::key_state key_state, void* userptr) {
      pile_t* pile = OFFSETLESS(userptr, pile_t, loco);
      switch(key) {
        case fan::key_delete: {
          switch(key_state) {
            case fan::key_state::press: {
              switch (pile->loco.focus.shape_type) {
                case loco_t::shapes::button: {
                  pile->fgm.builder_button.erase(&pile->loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id);
                  break;
                }
              }
              break;
            }
          }
          break;
        }
      }
    });

    loco->get_window()->add_mouse_move_callback(loco, [](fan::window_t* window, const fan::vec2i& position, void* userptr) {
      pile_t* pile = OFFSETLESS(userptr, pile_t, loco);
      switch (pile->loco.focus.shape_type) {
        case loco_t::shapes::button: {
          if (!(pile->fgm.action_flag & action::move)) {
            return;
          }
          pile->loco.button.set_position(
            &pile->loco,
            (fan::opengl::cid_t*)pile->loco.focus.shape_id,
            pile->loco.get_mouse_position(
              pile->fgm.viewport[viewport_area::editor].get_viewport_position(),
              pile->fgm.viewport[viewport_area::editor].get_viewport_size()
            ) + pile->fgm.move_offset
          );
          if (pile->fgm.builder_button.sanitize_cid((fan::opengl::cid_t*)pile->loco.focus.shape_id)) {
            pile->fgm.active.set_corners(&pile->loco, pile->fgm.builder_button.get_corners(&pile->loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id));
          }
          break;
        }
      }
    });

    // half size
    editor_size = fan::vec2(0.5, 1);

    fan::graphics::open_matrices(
      loco->get_context(),
      &matrices,
      loco->get_window()->get_size(),
      fan::vec2(-1, 1),
      fan::vec2(-1, 1)
    );
    viewport[viewport_area::global].open(
      loco->get_context(), 
      0, 
      translate_viewport_position(loco, fan::vec2(1, 1))
    );
    viewport[viewport_area::editor].open(
      loco->get_context(), 
      0, 
      translate_viewport_position(loco, editor_size)
    );
    {
      fan::vec2 top_viewport = translate_viewport_position(loco, editor_size);
      fan::vec2 bottom_viewport = translate_viewport_position(loco, fan::vec2(-0.5, 0));
      top_viewport.y = bottom_viewport.y;
      viewport[viewport_area::types].open(
        loco->get_context(),
        top_viewport,
        bottom_viewport
      );
    }

    line.open(loco);
    editor_button.open(loco);
    builder_button.open(loco);
    active.open(loco);

    line_t::properties_t lp;
    lp.viewport = &viewport[viewport_area::editor];
    lp.matrices = &matrices;
    lp.src = fan::vec2(-1.0 , -1.0);
    lp.dst = fan::vec2(+1.0, -1.0);
    lp.color = fan::colors::white;
    line.push_back(loco, lp);
    lp.src = lp.dst;
    lp.dst.y += +2.0;
    line.push_back(loco, lp);
    lp.src = fan::vec2(-1, 1);
    lp.dst = fan::vec2(1, 1);
    lp.viewport = &viewport[viewport_area::types];
    line.push_back(loco, lp);

    editor_button_t::properties_t ebp;
    ebp.userptr = loco;
    ebp.matrices = &matrices;
    ebp.viewport = &viewport[viewport_area::types];
    ebp.position = fan::vec2(0, -0.8);
    ebp.size = button_size;
    ebp.theme = &theme;
    ebp.text = "button";
    ebp.font_size = 0.2;
    ebp.mouse_input_cb = [](const loco_t::mouse_input_data_t& ii_d) -> uint8_t {
      pile_t* pile = OFFSETLESS(ii_d.userptr, pile_t, loco);
      if (ii_d.key != fan::mouse_left) {
        return 0;
      }
      if (ii_d.key_state != fan::key_state::press) {
        pile->fgm.builder_button.release(&pile->loco);
        if (!pile->fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position())) {
          switch (pile->loco.focus.shape_type) {
            case loco_t::shapes::button: {
              pile->fgm.builder_button.erase(&pile->loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id);
            /*  pile->loco.button.set_viewport(
                &pile->loco,
                (fan::opengl::cid_t*)pile->fgm.selected_index,
                &pile->fgm.viewport[viewport_area::editor]
              );*/
              break;
            }
          }
        }
       // pile->loco.focus.shape_type = fan::uninitialized;
        return 0;
      }
      if (ii_d.mouse_stage != fan_2d::graphics::gui::mouse_stage_e::inside) {
        return 0;
      }
      pile->fgm.action_flag |= action::move;
      builder_button_t::properties_t bbp;
      bbp.matrices = &pile->fgm.matrices;
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
      pile->fgm.builder_button.push_back(&pile->loco, bbp);
      return 0;
    };
    editor_button.push_back(loco, ebp);
   
  }
  void close(loco_t* loco) {
    line.close(loco);
    editor_button.close(loco);
    builder_button.close(loco);
  }

  struct line_t {
    using properties_t = loco_t::line_t::properties_t;

    void open(loco_t* loco) {
      cids.open();
    }
    void close(loco_t* loco) {
      cids.close();
    }
    void push_back(loco_t* loco, properties_t& p) {
      cids.resize(cids.size() + 1);
      cids[cids.size() - 1] = new fan::opengl::cid_t;
      loco->line.push_back(loco, cids[cids.size() - 1], p);
    }
    fan::hector_t<fan::opengl::cid_t*> cids;
  }line;

  struct editor_button_t {
    using properties_t = loco_t::button_t::properties_t;

    void open(loco_t* loco) {
      cids.open();
    }
    void close(loco_t* loco) {
      cids.close();
    }
    void push_back(loco_t* loco, properties_t& p) {
      cids.resize(cids.size() + 1);
      cids[cids.size() - 1] = new fan::opengl::cid_t;
      loco->button.push_back(loco, cids[cids.size() - 1], p);
    }
    void sanitize_cid(fan::opengl::cid_t* cid) {
      bool found = false;
      for (auto i : cids) {
        if (cid == i) {
          found = true;
        }
      }
      if (!found) {
        fan::throw_error("invalid cid");
      }
    }
    fan::hector_t<fan::opengl::cid_t*> cids;
  }editor_button;

  struct builder_button_t {
    using properties_t = loco_t::button_t::properties_t;

    void open(loco_t* loco) {
      cids.open();
    }
    void close(loco_t* loco) {
      cids.close();
    }
    corners_t get_corners(loco_t* loco, fan::opengl::cid_t* cid) const {
      pile_t* pile = OFFSETLESS(loco, pile_t, loco);
      fan::vec2 c = loco->button.get_button(loco, cid, &loco_t::button_t::instance_t::position);
      fan::vec2 s = loco->button.get_button(loco, cid, &loco_t::button_t::instance_t::size);
      corners_t corners;
      corners.corners[0] = c - s;
      corners.corners[1] = fan::vec2(c.x, c.y - s.y);
      corners.corners[2] = fan::vec2(c.x + s.x, c.y - s.y);
      corners.corners[3] = fan::vec2(c.x - s.x, c.y);
      corners.corners[4] = fan::vec2(c.x, c.y);
      corners.corners[5] = fan::vec2(c.x + s.x, c.y);
      corners.corners[6] = fan::vec2(c.x - s.x, c.y + s.y);
      corners.corners[7] = fan::vec2(c.x, c.y + s.y);
      corners.corners[8] = fan::vec2(c.x + s.x, c.y + s.y);
      return corners;
    }
    void release(loco_t* loco) {
      pile_t* pile = OFFSETLESS(loco, pile_t, loco);
      pile->fgm.move_offset = 0;
      pile->fgm.action_flag &= ~action::move;
    }
    void push_back(loco_t* loco, properties_t& p) {
      pile_t* pile = OFFSETLESS(loco, pile_t, loco);
      cids.resize(cids.size() + 1);
      cids[cids.size() - 1] = new fan::opengl::cid_t;
      p.mouse_input_cb = [](const loco_t::mouse_input_data_t& ii_d) -> uint8_t {
        pile_t* pile = OFFSETLESS(ii_d.userptr, pile_t, loco);
        if (ii_d.key != fan::mouse_left) {
          return 0;
        }
        if (ii_d.key_state == fan::key_state::release) {
          pile->fgm.builder_button.release(&pile->loco);
          return 0;
        }
        if (ii_d.key_state == fan::key_state::press && ii_d.mouse_stage == fan_2d::graphics::gui::mouse_stage_e::outside) {
          pile->fgm.active.clear(&pile->loco);
          return 0;
        }
        if (ii_d.mouse_stage != fan_2d::graphics::gui::mouse_stage_e::inside) {
          return 0;
        }
        pile->fgm.action_flag |= action::move;
        auto viewport = pile->loco.button.get_viewport(&pile->loco, (fan::opengl::cid_t*)ii_d.element_id);
        pile->fgm.move_offset =  fan::vec2(pile->loco.button.get_button(&pile->loco, (fan::opengl::cid_t*)ii_d.element_id, &loco_t::button_t::instance_t::position)) - pile->loco.get_mouse_position(viewport->get_viewport_position(), viewport->get_viewport_size());
        if (!pile->fgm.builder_button.sanitize_cid((fan::opengl::cid_t*)ii_d.element_id)) {
          return 0;
        }
        if (!pile->fgm.active.size()) {
          pile->fgm.active.push_corners(&pile->loco, pile->fgm.builder_button.get_corners(&pile->loco, (fan::opengl::cid_t*)ii_d.element_id));
        }
        else {
          pile->fgm.active.set_corners(&pile->loco, pile->fgm.builder_button.get_corners(&pile->loco, (fan::opengl::cid_t*)ii_d.element_id));
        }
        
        return 0;
      };
      p.userptr = loco;
      loco->button.push_back(loco, cids[cids.size() - 1], p);
      loco->focus.shape_type = loco_t::shapes::button;
      loco->focus.shape_id = cids[cids.size() - 1];
      sanitize_cid((fan::opengl::cid_t*)pile->loco.focus.shape_id);
      if (!pile->fgm.active.size()) {
        pile->fgm.active.push_corners(&pile->loco, pile->fgm.builder_button.get_corners(&pile->loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id));
      }
      else {
        pile->fgm.active.set_corners(&pile->loco, pile->fgm.builder_button.get_corners(&pile->loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id));
      }
    }
    void erase(loco_t* loco, fan::opengl::cid_t* cid) {
      loco->button.erase(loco, cid);
    }

    bool sanitize_cid(fan::opengl::cid_t* cid) {
      bool found = false;
      for (auto i : cids) {
        if (cid == i) {
          found = true;
        }
      }
      return found;
    }

    fan::hector_t<fan::opengl::cid_t*> cids;
  }builder_button;

  struct active_rectangles_t {
    using properties_t = loco_t::button_t::properties_t;

    static constexpr f32_t r_size = 0.015;

    void open(loco_t* loco) {
      cids.open();
    }
    void close(loco_t* loco) {
      cids.close();
    }
    void push_back(loco_t* loco, properties_t& p) {
      cids.resize(cids.size() + 1);
      cids[cids.size() - 1] = new fan::opengl::cid_t;
      loco->button.push_back(loco, cids[cids.size() - 1], p);
    }
    void push_corners(loco_t* loco, const corners_t& corners) {
      if (!cids.empty()) {
        fan::throw_error("a");
      }
      pile_t* pile = OFFSETLESS(loco, pile_t, loco);
      active_rectangles_t::properties_t p;
      p.size = r_size;
      switch(pile->loco.focus.shape_type) {
        case loco_t::shapes::button: {
          p.viewport = loco->button.get_viewport(loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id);
          p.matrices = loco->button.get_matrices(loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id);
          p.theme = loco->button.get_theme(loco, (fan::opengl::cid_t*)pile->loco.focus.shape_id);
          break;
        }
      }
      for (uint32_t i = 0; i < corners.count; i++) {
        p.position = corners.corners[i];
        push_back(loco, p);
      }
    }
    void set_corners(loco_t* loco, const corners_t& corners) {
      if (cids.empty()) {
        fan::throw_error("a");
      }
      pile_t* pile = OFFSETLESS(loco, pile_t, loco);
      for (uint32_t i = 0; i < corners.count; i++) {
        pile->loco.button.set_position(loco, cids[i], corners.corners[i]);
      }
    }

    void clear(loco_t* loco) {
      pile_t* pile = OFFSETLESS(loco, pile_t, loco);
      for (uint32_t i = 0; i < cids.size(); i++) {
        pile->loco.button.erase(loco, cids[i]);
        delete cids[i];
      }
      cids.clear();
    }

    uint32_t size() const {
      return cids.size();
    }
    void sanitize_cid(fan::opengl::cid_t* cid) {
      bool found = false;
      for (auto i : cids) {
        if (cid == i) {
          found = true;
        }
      }
      if (!found) {
        fan::throw_error("invalid cid");
      }
    }
    fan::hector_t<fan::opengl::cid_t*> cids;
  }active;

  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport[3];

  fan_2d::graphics::gui::theme_t theme;

  uint32_t action_flag;

  fan::vec2 move_offset;
};